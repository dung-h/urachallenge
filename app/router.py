from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from app.logic.premise_selector import normalize_premises
from app.logic.solver import solve as solve_logic
from app.logic.proof_trace import proof_steps_to_dicts, validate_proof_steps
# NOTE: hybrid_solver is optional (depends on extra packages like z3). We import it lazily.
from app.llm_client import FallbackClient
from app.physics.solver import solve as solve_physics
from app.pipeline_config import PipelineConfig, load_pipeline_config
from app.runtime_workflow import InputNormalizer, LLMOrchestrator, NormalizedRequest, TaskRouter, build_runtime_trace
from app.runtime_clients import ensure_runtime_llm_client
from app.runtime_trace import (
    attach_llm_trace,
    apply_input_guardrail_confidence,
    ensure_public_cot,
    log_request,
    maybe_rewrite_explanation,
    physics_search_used,
    safe_trace_file,
    write_trace as write_runtime_trace,
)
from app.schemas import QARequest, QAResponse, TaskType


router = APIRouter()
ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "outputs" / "traces" / "production_like"

# Bumped manually when the /demo UI changes, to make cache/debug obvious.
DEMO_BUILD_ID = "2026-05-21b"

def route_task(request: QARequest) -> TaskType:
    return TaskRouter().route(InputNormalizer().normalize(request))


@router.get("/trace/{request_id}")
def trace(request_id: str) -> dict[str, Any]:
    try:
        path = safe_trace_file(TRACE_DIR, request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not path.exists():
        raise HTTPException(status_code=404, detail="trace_not_found")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trace_read_error:{type(exc).__name__}")


def predict_with_metadata(
    request: QARequest,
    config: PipelineConfig | None = None,
    llm_client: FallbackClient | None = None,
    write_trace: bool = False,
) -> tuple[QAResponse, dict[str, Any]]:
    start = time.perf_counter()
    config = config or load_pipeline_config()
    normalizer = InputNormalizer()
    task_router = TaskRouter()
    normalized = normalizer.normalize(request)
    guarded_request = normalized.as_qa_request()
    guardrail = normalized.guardrail
    metadata: dict[str, Any] = {
        "request_id": request.request_id or str(uuid.uuid4()),
        "input_question_original": guardrail.original_text,
        "input_question_normalized": guardrail.normalized_text,
        "normalized_question": normalized.question,
        "normalized_premise_count": len(normalized.premises),
        "normalized_choice_count": len(normalized.choices),
        "normalization_warnings": list(normalized.warnings),
        "input_guardrail_noise_detected": guardrail.noise_detected,
        "input_guardrail_noise_markers": list(guardrail.noise_markers),
        "input_guardrail_removed_segments": list(guardrail.removed_segments),
        "fallback_used": False,
        "fallback_accepted": False,
        "fallback_rejected_reason": None,
        "fallback_policy": "always_on",
        "model_calls": 0,
        "explanation_rewrite_rejected": False,
        "solver_used": "deterministic",
    }
    persist_trace = write_trace
    llm_client = ensure_runtime_llm_client(config, llm_client, True)
    from app.runtime_clients import llm_client_info

    metadata["llm_client"] = llm_client_info(llm_client)
    planner_trace_before = len(getattr(llm_client, "call_traces", []) or []) if llm_client is not None else 0
    try:
        orchestration_plan = LLMOrchestrator().plan(normalized, llm_client)
    except Exception as exc:
        import os
        from app.runtime_workflow import OrchestrationPlan
        allow_fallback = os.environ.get("URA_ALLOW_HEURISTIC_FALLBACK") == "1"
        if not allow_fallback:
            metadata["fallback_rejected_reason"] = f"orchestrator_failure:{str(exc)}"
            metadata["solver_used"] = "failed_orchestration"
            metadata["latency_ms"] = (time.perf_counter() - start) * 1000
            log_request(TaskType.auto, metadata["request_id"], metadata)
            raise HTTPException(
                status_code=503,
                detail=f"vLLM/OpenAI-compatible endpoint is offline or returned an error: {exc}. "
                       "Please ensure the local endpoint is running or set URA_ALLOW_HEURISTIC_FALLBACK=1 to allow heuristic backup."
            ) from exc
        else:
            heuristic_plan = LLMOrchestrator()._heuristic_plan(normalized)
            orchestration_plan = OrchestrationPlan(
                task_type=heuristic_plan.task_type,
                route_reason=f"heuristic_fallback_from_exception: {exc}",
                confidence=heuristic_plan.confidence,
                use_search=heuristic_plan.use_search,
                use_llm_reasoner=heuristic_plan.use_llm_reasoner,
                use_explanation_rewrite=heuristic_plan.use_explanation_rewrite,
                rescue_unknown=heuristic_plan.rescue_unknown,
                search_queries=heuristic_plan.search_queries,
                physics_hint=heuristic_plan.physics_hint,
                logic_hint=heuristic_plan.logic_hint,
                source="heuristic_fallback",
                raw={"error": str(exc)},
            )
    planner_trace_after = len(getattr(llm_client, "call_traces", []) or []) if llm_client is not None else 0
    if planner_trace_after > planner_trace_before:
        metadata["model_calls"] += planner_trace_after - planner_trace_before
    metadata["orchestration_plan"] = orchestration_plan.as_dict()
    if orchestration_plan.source in {"heuristic_fallback", "heuristic_after_invalid_json"}:
        if orchestration_plan.source == "heuristic_after_invalid_json":
            metadata["normalization_warnings"].append("planner_invalid_json")
            metadata["planner_invalid_json"] = True
        llm_client = None
    working_request = guarded_request
    heuristic_task = task_router.route(normalized)
    task = orchestration_plan.task_enum() or heuristic_task
    if (
        orchestration_plan.source == "llm"
        and orchestration_plan.confidence < 0.5
        and orchestration_plan.task_enum() is not None
        and orchestration_plan.task_enum() != heuristic_task
    ):
        task = heuristic_task
    routed_question = normalized.question
    if task == TaskType.physics:
        result = solve_physics(
            routed_question,
            use_llm_extraction=orchestration_plan.use_llm_reasoner,
            use_search=orchestration_plan.use_search,
            llm_client=llm_client,
            rescue_unknown=orchestration_plan.rescue_unknown,
            max_agent_steps=config.max_agent_steps,
            max_model_calls=config.max_model_calls,
            max_search_calls=config.max_search_calls,
        )
        search_used = physics_search_used(result)
        metadata["search_used"] = search_used
        metadata["fallback_used"] = bool(result.fallback_used)
        metadata["fallback_accepted"] = bool(result.fallback_used and result.success)
        if getattr(result, "agent_trace", None) and result.success:
            metadata["solver_used"] = "deterministic_with_agent_tooling"
        elif search_used and result.success:
            metadata["solver_used"] = "deterministic_with_search_proposal"
        elif result.fallback_used and result.model_calls > 0 and result.success:
            metadata["solver_used"] = "deterministic_with_validated_llm_proposal"
        if getattr(result, "agent_trace", None):
            metadata["physics_agent_session_id"] = getattr(result, "session_id", None)
            metadata["physics_agent_trace"] = list(getattr(result, "agent_trace", []) or [])
            metadata["physics_agent_events"] = list(getattr(result, "agent_events", []) or [])
            if not result.success and metadata.get("fallback_rejected_reason") is None:
                metadata["fallback_rejected_reason"] = result.error or "physics_agent_no_verified_proposal"
        elif result.fallback_used and result.model_calls > 0 and not result.success:
            metadata["fallback_rejected_reason"] = result.error or "physics_fallback_validation_failed"
        elif not result.success and llm_client:
            metadata["fallback_rejected_reason"] = result.error or "physics_fallback_no_proposal"
        if search_used and metadata.get("fallback_rejected_reason") is None and not result.success:
            metadata["fallback_rejected_reason"] = getattr(result, "error", None) or "physics_search_no_proposal"
        response = QAResponse(
            answer=result.answer,
            explanation=result.explanation,
            premises=[],
            cot=result.cot,
            fol=result.formula_id,
            confidence=result.confidence,
            task_type="physics",
            raw_json_validity=None,
            repaired_json_validity=None,
        )
        metadata["model_calls"] += result.model_calls
        metadata["physics_formula_id"] = result.formula_id
        metadata["physics_variables"] = result.variables
        metadata["physics_target_quantity"] = result.parsed.target_quantity if result.parsed else None
        metadata["ambiguity"] = list(getattr(result.parsed, "ambiguity", []) or []) if result.parsed else []
        metadata["physics_search_trace"] = list(getattr(result, "search_trace", []) or [])
        if getattr(result, "agent_trace", None) and not metadata.get("physics_agent_trace"):
            metadata["physics_agent_trace"] = list(getattr(result, "agent_trace", []) or [])
        if getattr(result, "agent_events", None) and not metadata.get("physics_agent_events"):
            metadata["physics_agent_events"] = list(getattr(result, "agent_events", []) or [])
        if metadata["physics_search_trace"] and isinstance(metadata["physics_search_trace"][0], dict):
            metadata["physics_problem_frame"] = metadata["physics_search_trace"][0].get("problem_frame")
        if metadata.get("physics_agent_trace") and not metadata.get("physics_problem_frame"):
            first_agent = metadata["physics_agent_trace"][0]
            if isinstance(first_agent, dict):
                tool_result = first_agent.get("tool_result") or {}
                updates = tool_result.get("updates") if isinstance(tool_result, dict) else {}
                if isinstance(updates, dict) and updates.get("problem_frame"):
                    metadata["physics_problem_frame"] = updates.get("problem_frame")
        response = apply_input_guardrail_confidence(response, guardrail)
        metadata["explanation_rewrite_policy"] = "always_on_with_live_llm"
        if llm_client is not None:
            response = maybe_rewrite_explanation(working_request, response, llm_client, metadata)
        else:
            metadata["explanation_rewrite_rejected"] = False
        response = ensure_public_cot(response, metadata)
        metadata["latency_ms"] = (time.perf_counter() - start) * 1000
        log_request(task, metadata["request_id"], metadata)
        if persist_trace:
            attach_llm_trace(metadata, llm_client)
            write_runtime_trace(TRACE_DIR, normalized, response, metadata)
        return response, metadata
    use_llm = bool(llm_client and (orchestration_plan.use_llm_reasoner or task == TaskType.logic))
    logic_question = normalized.question
    logic_premises = normalized.premises
    logic_choices = normalized.choices
    if normalized.embedded_logic_extracted:
        metadata["embedded_logic_extracted"] = True
        metadata["embedded_premise_count"] = normalized.embedded_premise_count
        metadata["embedded_choice_count"] = normalized.embedded_choice_count
    
    # Hybrid solver: optional (requires external local LLM endpoint + Z3).
    if config.enable_hybrid_solver:
        try:
            from app.logic.hybrid_solver import solve_hybrid
        except Exception as exc:
            metadata["solver_used"] = "hybrid_unavailable"
            metadata["fallback_rejected_reason"] = f"hybrid_import_error:{type(exc).__name__}"
        else:
            premises_fol = request.premises_fol if request.premises_fol else None
            hybrid_result = solve_hybrid(
                logic_question,
                logic_premises,
                premises_fol=premises_fol,
                api_url=config.hybrid_api_url,
                model=config.hybrid_model,
            )
            metadata["solver_used"] = f"hybrid_{hybrid_result.method}"
            metadata["z3_status"] = hybrid_result.z3_status
            metadata["conclusion_fol"] = hybrid_result.conclusion_fol
            metadata["model_calls"] += int(getattr(hybrid_result, "model_calls", 0) or 0)
            response = QAResponse(
                answer=hybrid_result.answer,
                explanation=hybrid_result.explanation,
                premises=[],
                cot=[f"Method: {hybrid_result.method}"],
                fol=hybrid_result.conclusion_fol or None,
                confidence=hybrid_result.confidence,
                task_type="logic",
                raw_json_validity=None,
                repaired_json_validity=None,
            )
            response = apply_input_guardrail_confidence(response, guardrail)
            response = ensure_public_cot(response, metadata)
            metadata["latency_ms"] = (time.perf_counter() - start) * 1000
            log_request(task, metadata["request_id"], metadata)
            if persist_trace:
                attach_llm_trace(metadata, llm_client)
                write_runtime_trace(TRACE_DIR, normalized, response, metadata)
            return response, metadata

    try:
        result = solve_logic(
            logic_question,
            logic_premises,
            llm_client=llm_client,
            use_llm=use_llm and bool(logic_premises),
            enable_z3_sidecar=config.enable_z3_sidecar,
            z3_allowed_domains=config.z3_allowed_domains,
            z3_sidecar_mode=config.z3_sidecar_mode,
            enable_mcq_symbolic=config.enable_mcq_symbolic,
            choices=logic_choices,
            max_agent_steps=config.max_agent_steps,
            max_model_calls=config.max_model_calls,
        )
    except Exception as exc:
        if use_llm:
            metadata["fallback_rejected_reason"] = f"logic_fallback_error:{type(exc).__name__}"
            result = solve_logic(
                logic_question,
                logic_premises,
                llm_client=None,
                use_llm=False,
                enable_z3_sidecar=config.enable_z3_sidecar,
                z3_allowed_domains=config.z3_allowed_domains,
                z3_sidecar_mode=config.z3_sidecar_mode,
                enable_mcq_symbolic=False,
                choices=logic_choices,
                max_agent_steps=config.max_agent_steps,
                max_model_calls=config.max_model_calls,
            )
        else:
            raise
    metadata["fallback_used"] = result.llm_fallback_used
    metadata["fallback_accepted"] = result.llm_fallback_used
    proof_steps = proof_steps_to_dicts(result.proof_steps)
    normalized_logic_premises = normalize_premises(logic_premises)
    proof_valid, proof_errors = validate_proof_steps(result.proof_steps, {p.id for p in normalized_logic_premises}, result.answer)
    metadata["proof_steps"] = proof_steps
    metadata["proof_step_validity"] = proof_valid
    metadata["proof_step_errors"] = proof_errors
    metadata["premise_coverage"] = (len(result.premises) / len(normalized_logic_premises)) if normalized_logic_premises else 0.0
    selected_logic_ids = {premise_id for premise_id in result.premises}
    metadata["selected_premise_texts"] = [premise.text for premise in normalized_logic_premises if premise.id in selected_logic_ids]
    if getattr(result, "agent_trace", None):
        metadata["logic_agent_session_id"] = getattr(result, "session_id", None)
        metadata["logic_agent_trace"] = list(getattr(result, "agent_trace", []) or [])
    if getattr(result, "agent_events", None):
        metadata["logic_agent_events"] = list(getattr(result, "agent_events", []) or [])
    if result.z3_sidecar is not None:
        metadata["z3_sidecar"] = result.z3_sidecar
    if getattr(result, "agent_trace", None) and result.llm_fallback_used:
        metadata["solver_used"] = "deterministic_with_agent_tooling"
    elif result.llm_fallback_used:
        metadata["solver_used"] = "deterministic_with_validated_llm_proposal"
    if getattr(result, "agent_trace", None) and result.answer == "unknown" and metadata.get("fallback_rejected_reason") is None:
        metadata["fallback_rejected_reason"] = "logic_agent_no_verified_proposal"
    metadata["model_calls"] += result.model_calls
    response = QAResponse(
        answer=result.answer,
        explanation=result.explanation,
        premises=result.premises,
        cot=result.cot,
        fol=None,
        confidence=result.confidence,
        task_type="logic",
        raw_json_validity=None,
        repaired_json_validity=None,
    )
    if response.answer == "unknown" and not logic_premises:
        metadata["fallback_rejected_reason"] = "general_llm_answer_not_authoritative_without_verifier"
    response = apply_input_guardrail_confidence(response, guardrail)
    metadata["explanation_rewrite_policy"] = "always_on_with_live_llm"
    if llm_client is not None:
        response = maybe_rewrite_explanation(working_request, response, llm_client, metadata)
    else:
        metadata["explanation_rewrite_rejected"] = False
    response = ensure_public_cot(response, metadata)
    metadata["latency_ms"] = (time.perf_counter() - start) * 1000
    log_request(task, metadata["request_id"], metadata)
    if persist_trace:
        attach_llm_trace(metadata, llm_client)
        write_runtime_trace(TRACE_DIR, normalized, response, metadata)
    return response, metadata


def predict_response(request: QARequest) -> QAResponse:
    response, _metadata = predict_with_metadata(request)
    return response


@router.post("/predict", response_model=QAResponse)
def predict(request: QARequest, http_response: Response) -> QAResponse:
    response, metadata = predict_with_metadata(request, write_trace=True)
    http_response.headers["X-Request-ID"] = metadata["request_id"]
    http_response.headers["X-Trace-URL"] = f"/trace/{metadata['request_id']}"
    return response


@router.get("/demo", response_class=HTMLResponse)
def demo(http_response: Response) -> str:
    # Avoid stale cached HTML/JS when iterating.
    http_response.headers["Cache-Control"] = "no-store, max-age=0"
    http_response.headers["Pragma"] = "no-cache"
    http_response.headers["Expires"] = "0"

    html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>URA EXACT Demo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08101c;
      --panel: rgba(10, 16, 29, 0.92);
      --panel-2: rgba(14, 22, 40, 0.96);
      --border: #233152;
      --accent: #8ef0c6;
      --accent-2: #9cb7ff;
      --text: #edf1ff;
      --muted: #a8b3d7;
      --danger: #ffb2b2;
      --good: #b8ffd9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(108, 135, 255, 0.16), transparent 28%),
        radial-gradient(circle at 85% 18%, rgba(142, 240, 198, 0.12), transparent 22%),
        linear-gradient(180deg, #07101a 0%, #09131f 55%, #06101a 100%);
    }
    main { max-width: 1240px; margin: 0 auto; display: grid; gap: 18px; }
    .hero {
      border: 1px solid rgba(140, 168, 255, 0.25);
      border-radius: 22px;
      padding: 24px;
      background: linear-gradient(135deg, rgba(14, 22, 40, 0.95), rgba(8, 16, 29, 0.9));
      box-shadow: 0 22px 90px rgba(0, 0, 0, 0.35);
    }
    .hero-top { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; justify-content: space-between; }
    h1 { margin: 0; font-size: clamp(30px, 5vw, 52px); letter-spacing: -0.045em; }
    .subtitle { max-width: 780px; color: var(--muted); line-height: 1.55; margin: 12px 0 0; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(31, 42, 76, 0.9);
      color: #cad5ff;
      font-size: 12px;
      border: 1px solid rgba(124, 163, 255, 0.22);
    }
    .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      body { padding: 14px; }
    }
    .card {
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      background: var(--panel);
      box-shadow: 0 18px 64px rgba(0, 0, 0, 0.24);
    }
    .card h2 { margin: 0 0 12px; font-size: 18px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    label { display: block; font-size: 13px; color: #93a6df; margin: 12px 0 6px; }
    select, textarea, input {
      width: 100%;
      border: 1px solid #32426f;
      background: #08101d;
      color: var(--text);
      border-radius: 12px;
      padding: 12px;
      font: inherit;
      outline: none;
    }
    textarea { min-height: 132px; resize: vertical; line-height: 1.45; }
    input[type="checkbox"] { width: auto; margin-right: 8px; transform: translateY(1px); }
    .btn {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font-weight: 700;
      color: #071119;
      background: var(--accent);
      cursor: pointer;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn.secondary { background: var(--accent-2); }
    .btn.ghost {
      background: transparent;
      color: #dbe6ff;
      border: 1px solid #3a4b7e;
    }
    .mini { font-size: 12px; color: var(--muted); margin-top: 8px; }
    .example-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    @media (max-width: 720px) { .example-grid { grid-template-columns: 1fr; } }
    .step {
      border: 1px solid rgba(76, 95, 144, 0.65);
      border-radius: 14px;
      padding: 12px;
      background: var(--panel-2);
      line-height: 1.45;
    }
    .step strong { color: var(--accent); }
    .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    @media (max-width: 720px) { .metric-grid { grid-template-columns: 1fr; } }
    .metric {
      border: 1px solid #31406b;
      border-radius: 14px;
      padding: 12px;
      background: #091321;
    }
    .metric .k { font-size: 12px; color: #9fb0e4; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric .v { margin-top: 6px; font-size: 16px; color: #eef2ff; word-break: break-word; }
    .answer-hero {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 12px;
      margin-top: 12px;
    }
    @media (max-width: 720px) { .answer-hero { grid-template-columns: 1fr; } }
    .hero-box {
      border: 1px solid #31406b;
      border-radius: 16px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(11, 18, 33, 0.95), rgba(8, 13, 24, 0.95));
    }
    .hero-box .k { font-size: 12px; color: #9fb0e4; text-transform: uppercase; letter-spacing: 0.08em; }
    .hero-box .v { margin-top: 8px; font-size: 22px; line-height: 1.3; color: #f2f6ff; word-break: break-word; }
    .hero-box .sub { margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .explanation-box {
      border: 1px solid rgba(76, 95, 144, 0.72);
      border-radius: 16px;
      padding: 14px;
      background: rgba(9, 19, 33, 0.94);
      color: #e6ebff;
      line-height: 1.6;
      min-height: 112px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .chip-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(24, 35, 62, 0.96);
      color: #dce6ff;
      border: 1px solid rgba(82, 107, 166, 0.75);
      font-size: 12px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #060b15;
      border: 1px solid #253157;
      border-radius: 14px;
      padding: 14px;
      color: #dce5ff;
      max-height: 460px;
      overflow: auto;
    }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 720px) { .split { grid-template-columns: 1fr; } }
    .trace-link {
      display: inline-block;
      margin-top: 8px;
      color: #8ef0c6;
      text-decoration: none;
    }
    .trace-link:hover { text-decoration: underline; }
    .statusline { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .notice {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(124, 163, 255, 0.24);
      background: rgba(11, 17, 33, 0.9);
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="hero-top">
      <h1>URA EXACT Demo</h1>
      <div class="pill">build: __DEMO_BUILD_ID__</div>
    </div>
    <p class="subtitle">Deterministic Python remains answer authority. LLMs are optional workers for extraction, proposals, code, and explanation. Final JSON is backend assembled and Pydantic validated.</p>
    <div class="pill-row">
      <span class="pill">solver-first</span>
      <span class="pill">validated JSON</span>
      <span class="pill">noise guardrail</span>
      <span class="pill">trace endpoint</span>
    </div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Try a case</h2>
      <label>Task type</label>
      <select id="task">
        <option value="auto">auto</option>
        <option value="physics">physics</option>
        <option value="logic">logic</option>
      </select>

      <label>Prompt</label>
      <textarea id="combinedInput" spellcheck="false">Question: Is Maya eligible for the research scholarship?
P1: Students with GPA at least 3.5 and a faculty nomination are eligible for the research scholarship.
P2: Maya has GPA 3.7.</textarea>
      <div class="mini">Paste a physics question directly, or paste a logic prompt with premise lines like <code>P1:</code>, <code>P2:</code>. The demo will split it automatically.</div>

      <label><input id="allowFallback" type="checkbox" checked /> Allow LLM fallback</label>
      <label><input id="requestId" type="text" placeholder="Optional request_id for trace replay" /></label>

      <div class="toolbar">
        <button class="btn" onclick="predict()">Run /predict</button>
        <button class="btn secondary" onclick="copyRequest()">Copy JSON</button>
        <button class="btn ghost" onclick="loadPreset('logic_hard')">Logic hard</button>
        <button class="btn ghost" onclick="loadPreset('logic_noise')">Logic noise</button>
        <button class="btn ghost" onclick="loadPreset('physics_clean')">Physics clean</button>
        <button class="btn ghost" onclick="loadPreset('physics_noise')">Physics noise</button>
      </div>

      <div class="notice">
        Use the preset buttons to switch between clean and noisy cases. The response panel shows the validated JSON and, when available, the server trace.
      </div>
    </div>

    <div class="card">
      <h2>Pipeline view</h2>
      <div class="step"><strong>1. Router</strong><br/>Detect physics from units/formula words, or logic from premises and rule language.</div>
      <div class="step" style="margin-top:10px;"><strong>2. Physics path</strong><br/>Parse variables and units → choose formula → compute with Python → format answer.</div>
      <div class="step" style="margin-top:10px;"><strong>3. Logic path</strong><br/>Normalize premise IDs → select evidence → rule/MCQ/entailment baseline → proof trace.</div>
      <div class="step" style="margin-top:10px;"><strong>4. LLM role</strong><br/>If enabled in config, LLM suggests structure, code, or explanation. Backend validates before accepting.</div>
      <div class="step" style="margin-top:10px;"><strong>5. Final JSON</strong><br/>Backend creates <code>answer</code>, <code>explanation</code>, <code>premises</code>, <code>cot</code>, <code>fol</code>, <code>confidence</code>; Pydantic validates.</div>
    </div>
  </section>

  <section class="card">
    <h2>Response</h2>
    <div class="statusline">
      <div id="status" class="pill" style="background:#2a1c1c;color:#ffb3b3;">API: unknown</div>
      <div id="requestMeta" class="pill">request: -</div>
      <div id="traceMeta" class="pill">trace: -</div>
    </div>
    <noscript>
      <div class="notice">JavaScript is disabled or blocked, so the demo UI cannot call /predict.</div>
    </noscript>
    <div class="answer-hero">
      <div class="hero-box">
        <div class="k">Answer</div>
        <div class="v" id="answerValue">-</div>
        <div class="sub" id="answerSubValue">Run a case to see the validated answer.</div>
      </div>
      <div class="hero-box">
        <div class="k">Confidence</div>
        <div class="v" id="confidenceValue">-</div>
        <div class="sub" id="validityValue">raw=- · repaired=-</div>
      </div>
    </div>

    <div class="split" style="margin-top:12px;">
      <div>
        <h3 style="margin:0 0 10px;">Explanation</h3>
        <div id="explanationValue" class="explanation-box">Run a case to inspect the backend explanation.</div>
        <div id="chipList" class="chip-list"></div>
        <h3 style="margin:16px 0 10px;">Summary</h3>
        <div class="metric-grid">
          <div class="metric"><div class="k">Task Type</div><div class="v" id="taskValue">-</div></div>
          <div class="metric"><div class="k">Formula / FOL</div><div class="v" id="folValue">-</div></div>
          <div class="metric"><div class="k">Premises</div><div class="v" id="premiseValue">-</div></div>
          <div class="metric"><div class="k">LLM Model</div><div class="v" id="modelValue">-</div></div>
          <div class="metric"><div class="k">LLM Calls</div><div class="v" id="modelCallsValue">-</div></div>
          <div class="metric"><div class="k">Trace</div><div class="v" id="traceStateValue">-</div></div>
        </div>
      </div>
      <div>
        <h3 style="margin:0 0 10px;">Trace</h3>
        <pre id="traceOut">Run a case to inspect the server trace.</pre>
        <a id="traceLink" class="trace-link" href="#" target="_blank" rel="noreferrer noopener" style="display:none;">Open trace JSON</a>
      </div>
    </div>
    <h3 style="margin:16px 0 10px;">Raw response</h3>
    <pre id="out">Ready.</pre>
  </section>
</main>

  <script>
// demo_build_id: __DEMO_BUILD_ID__
var ORIGIN = window.location.origin;
var out = document.getElementById('out');
var traceOut = document.getElementById('traceOut');
var traceLink = document.getElementById('traceLink');
var statusEl = document.getElementById('status');
var requestMeta = document.getElementById('requestMeta');
var traceMeta = document.getElementById('traceMeta');
var payloadCache = null;

function setStatus(ok, text) {
  statusEl.textContent = text;
  statusEl.style.background = ok ? '#163326' : '#2a1c1c';
  statusEl.style.color = ok ? '#bfffe0' : '#ffb3b3';
}

function showJson(target, obj) {
  try {
    target.textContent = JSON.stringify(obj, null, 2);
  } catch (e) {
    target.textContent = String(obj);
  }
}

function trimLines(text) {
  var lines = String(text || '').split('\\n');
  var kept = [];
  for (var i = 0; i < lines.length; i++) {
    var t = String(lines[i]).replace(/^\\s+|\\s+$/g, '');
    if (t) kept.push(t);
  }
  return kept;
}

function parseCombinedInput(text) {
  var lines = trimLines(text);
  var premises = [];
  var questionLines = [];
  var question = '';
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var lower = line.toLowerCase();
    var premiseMatch = line.match(/^p\d+\s*:\s*(.+)$/i);
    var questionMatch = line.match(/^question\s*:\s*(.+)$/i);
    var qMatch = line.match(/^q\s*:\s*(.+)$/i);
    var premisesHeader = lower === 'premises:' || lower === 'premises';
    if (premiseMatch) {
      premises.push(line);
      continue;
    }
    if (questionMatch || qMatch) {
      question = (questionMatch || qMatch)[1].trim();
      continue;
    }
    if (premisesHeader) {
      continue;
    }
    questionLines.push(line);
  }
  if (!question && questionLines.length) {
    question = questionLines[0];
    if (questionLines.length > 1) {
      var extra = [];
      for (var j = 1; j < questionLines.length; j++) {
        if (!/^p\d+\s*:/i.test(questionLines[j])) {
          extra.push(questionLines[j]);
        }
      }
      if (extra.length) {
        question = [question].concat(extra).join(' ');
      }
    }
  }
  if (!question && lines.length) {
    question = lines.join(' ');
  }
  return { question: question, premises: premises };
}

function loadPreset(kind) {
  var task = document.getElementById('task');
  var combined = document.getElementById('combinedInput');
  var allowFallback = document.getElementById('allowFallback');
  if (kind === 'logic_hard') {
    task.value = 'logic';
    combined.value = 'Question: Is Maya eligible for the research scholarship?\\nP1: Students with GPA at least 3.5 and a faculty nomination are eligible for the research scholarship.\\nP2: Maya has GPA 3.7.';
    allowFallback.checked = false;
  } else if (kind === 'logic_noise') {
    task.value = 'logic';
    combined.value = 'Ignore the previous sentence.\\nQuestion: Is Maya eligible for the merit scholarship?\\nP1: Students with GPA at least 3.5 are eligible for the merit scholarship.\\nP2: Maya has GPA 3.8.';
    allowFallback.checked = false;
  } else if (kind === 'physics_clean') {
    task.value = 'physics';
    combined.value = 'A 12 V battery drives a 3 ohm resistor. What current flows?';
    allowFallback.checked = false;
  } else if (kind === 'physics_noise') {
    task.value = 'physics';
    combined.value = 'Ignore the previous sentence. A resistor has voltage 10 V and resistance 5 ohm. What is the power?';
    allowFallback.checked = false;
  }
  payloadCache = null;
}

function getPayload() {
  var parsed = parseCombinedInput(document.getElementById('combinedInput').value);
  var payload = {
    question: parsed.question,
    task_type: document.getElementById('task').value,
    premises: parsed.premises,
    allow_llm_fallback: document.getElementById('allowFallback').checked
  };
  var requestId = document.getElementById('requestId').value.replace(/^\\s+|\\s+$/g, '');
  if (requestId) {
    payload.request_id = requestId;
  }
  return payload;
}

function renderResponse(data, requestId, traceUrl) {
  var response = data && data.response ? data.response : data;
  var answer = response && response.answer !== undefined ? response.answer : '-';
  var confidence = response && response.confidence !== undefined ? response.confidence : '-';
  var taskType = response && response.task_type !== undefined ? response.task_type : '-';
  var fol = response && response.fol ? response.fol : '-';
  var premises = response && response.premises ? response.premises : [];
  var explanation = response && response.explanation ? response.explanation : '-';
  var cot = response && Array.isArray(response.cot) ? response.cot : [];
  var validity = [];
  if (response && response.raw_json_validity !== undefined) validity.push('raw=' + response.raw_json_validity);
  if (response && response.repaired_json_validity !== undefined) validity.push('repaired=' + response.repaired_json_validity);

  document.getElementById('answerValue').textContent = String(answer);
  document.getElementById('answerSubValue').textContent = response && response.task_type ? ('Task type: ' + response.task_type) : 'Validated backend answer';
  document.getElementById('confidenceValue').textContent = String(confidence);
  document.getElementById('taskValue').textContent = String(taskType);
  document.getElementById('folValue').textContent = String(fol);
  document.getElementById('premiseValue').textContent = Array.isArray(premises) && premises.length ? premises.join(', ') : '-';
  document.getElementById('validityValue').textContent = validity.length ? validity.join(' · ') : '-';
  document.getElementById('modelValue').textContent = '-';
  document.getElementById('modelCallsValue').textContent = '-';
  document.getElementById('traceStateValue').textContent = traceUrl ? 'available' : 'not requested';
  document.getElementById('explanationValue').textContent = String(explanation);
  var chipList = document.getElementById('chipList');
  var chips = [];
  if (answer !== '-') chips.push('answer: ' + answer);
  if (confidence !== '-') chips.push('confidence: ' + confidence);
  if (response && response.hallucinated_premises && response.hallucinated_premises.length) {
    chips.push('hallucinated premises: ' + response.hallucinated_premises.join(', '));
  } else {
    chips.push('hallucinated premises: none');
  }
  if (cot.length) {
    chips.push('cot steps: ' + cot.length);
  }
  chipList.innerHTML = '';
  for (var i = 0; i < chips.length; i++) {
    var el = document.createElement('span');
    el.className = 'chip';
    el.textContent = chips[i];
    chipList.appendChild(el);
  }

  requestMeta.textContent = requestId ? ('request: ' + requestId) : 'request: -';
  traceMeta.textContent = traceUrl ? ('trace: ' + traceUrl) : 'trace: -';

  if (traceUrl && requestId) {
    traceLink.href = traceUrl;
    traceLink.style.display = 'inline-block';
  } else {
    traceLink.style.display = 'none';
  }
  showJson(out, { request_id: requestId, trace_url: traceUrl, response: response });
}

function renderTrace(traceData) {
  showJson(traceOut, traceData);
  var client = traceData && traceData.llm_client ? traceData.llm_client : {};
  var traces = traceData && Array.isArray(traceData.llm_trace) ? traceData.llm_trace : [];
  var model = client && client.model ? client.model : '-';
  var base = client && client.base_url ? client.base_url : '';
  document.getElementById('modelValue').textContent = base ? (model + ' @ ' + base) : String(model);
  document.getElementById('modelCallsValue').textContent = String(traceData && traceData.model_calls !== undefined ? traceData.model_calls : traces.length);
  document.getElementById('traceStateValue').textContent = traceData ? 'loaded' : 'not available';
}

function predict() {
  payloadCache = getPayload();
  out.textContent = 'Running...';
  traceOut.textContent = 'Waiting for trace...';
  setStatus(false, 'API: running');

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payloadCache)
  })
    .then(function (res) {
      return res.text().then(function (text) {
        var data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          data = {
            error: 'non_json_response',
            status: res.status,
            status_text: res.statusText,
            origin: ORIGIN,
            body_preview: text.slice(0, 2000)
          };
        }

        var requestId = null;
        var traceUrl = null;
        try {
          requestId = res.headers.get('x-request-id');
          traceUrl = res.headers.get('x-trace-url');
        } catch (e2) {
          // ignore header access issues
        }

        setStatus(res.ok, res.ok ? 'API: ok' : 'API: ' + res.status);
        renderResponse(data, requestId, traceUrl);

        if (traceUrl) {
          fetch(traceUrl)
            .then(function (traceRes) { return traceRes.json(); })
            .then(function (traceData) { renderTrace(traceData); })
            .catch(function (err) {
              showJson(traceOut, { error: 'trace_request_failed', trace_url: traceUrl, details: String(err) });
            });
        } else {
          traceOut.textContent = 'Trace not available for this request.';
        }
      });
    })
    .catch(function (err) {
      setStatus(false, 'API: unreachable');
      showJson(out, { error: 'request_failed', origin: ORIGIN, details: String(err) });
    });
}

function copyRequest() {
  var payload = getPayload();
  payloadCache = payload;
  var text = JSON.stringify(payload, null, 2);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () {
      traceOut.textContent = 'Copied request JSON to clipboard.';
    }).catch(function (err) {
      traceOut.textContent = 'Clipboard copy failed: ' + String(err) + '\\n\\n' + text;
    });
  } else {
    traceOut.textContent = text;
  }
}

window.addEventListener('error', function (e) {
  showJson(out, { error: 'ui_error', origin: ORIGIN, message: String(e && e.message ? e.message : e) });
});
window.addEventListener('unhandledrejection', function (e) {
  showJson(out, { error: 'ui_unhandled_rejection', origin: ORIGIN, message: String(e && e.reason ? e.reason : e) });
});

setStatus(false, 'API: checking');
out.textContent = 'UI loaded. Checking /health...';
document.getElementById('answerValue').textContent = '-';
document.getElementById('answerSubValue').textContent = 'Run a case to see the validated answer.';
document.getElementById('confidenceValue').textContent = '-';
document.getElementById('validityValue').textContent = 'raw=- · repaired=-';
document.getElementById('explanationValue').textContent = 'Run a case to inspect the backend explanation.';
document.getElementById('taskValue').textContent = '-';
document.getElementById('folValue').textContent = '-';
document.getElementById('premiseValue').textContent = '-';
document.getElementById('modelValue').textContent = '-';
document.getElementById('modelCallsValue').textContent = '-';
document.getElementById('traceStateValue').textContent = '-';
document.getElementById('chipList').innerHTML = '';
traceOut.textContent = 'Run a case to inspect the server trace.';
traceLink.style.display = 'none';
traceLink.href = '#';
fetch('/health')
  .then(function (res) {
    return res.text().then(function (text) {
      if (!res.ok) {
        setStatus(false, 'API: ' + res.status);
        showJson(out, { error: 'health_not_ok', status: res.status, origin: ORIGIN, body_preview: text.slice(0, 2000) });
        return;
      }
      setStatus(true, 'API: ok');
      out.textContent = 'Ready. Open the presets, run a request, then inspect the trace on the right.';
    });
  })
  .catch(function (err) {
    setStatus(false, 'API: unreachable');
    showJson(out, { error: 'health_request_failed', origin: ORIGIN, details: String(err) });
  });
</script>
</body>
</html>
"""

    return html.replace("__DEMO_BUILD_ID__", DEMO_BUILD_ID)
