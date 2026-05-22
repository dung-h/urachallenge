from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from app.guardrails import guardrail_prompt_text
from app.logic.premise_selector import normalize_premises
from app.logic.solver import solve as solve_logic
from app.logic.proof_trace import proof_steps_to_dicts, validate_proof_steps
from app.explanation_worker import build_explanation_trace, validate_explanation_rewrite
# NOTE: hybrid_solver is optional (depends on extra packages like z3). We import it lazily.
from app.llm_client import FallbackClient, OpenAICompatibleLLMClient, HuggingFaceLLMClient
from app.physics.solver import solve as solve_physics, solve_from_llm_suggestion, solve_from_llm_code
from app.pipeline_config import PipelineConfig, load_pipeline_config
from app.schemas import QARequest, QAResponse, TaskType


router = APIRouter()
ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "outputs" / "traces" / "production_like"

# Bumped manually when the /demo UI changes, to make cache/debug obvious.
DEMO_BUILD_ID = "2026-05-21b"

logger = logging.getLogger("ura")
uvicorn_logger = logging.getLogger("uvicorn.error")


PHYSICS_HINTS = {
    "voltage", "current", "resistance", "resistor", "ohm", "power", "capacitor", "capacitance",
    "charge", "electric field", "force", "joule", "watt", "microfarad", "coulomb",
    "lc circuit", "rlc", "inductance", "inductor", "mh", "uf", "frequency", "angular frequency",
    "rad/s", "resonant", "resonance",
}


def _looks_like_logic_prompt(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search(r"\bif\b.+\bthen\b", low, re.S)
        or "based only on the rules" in low
        or "rules state" in low
        or re.search(r"(?m)^\s*(?:all|no|some)\b", text, re.I)
        or re.search(r"(?m)^\s*(?:does|is|are|must)\b.+\?\s*$", text, re.I)
        or re.search(r"\b[A-E]\)\s+", text)
    )


def route_task(request: QARequest) -> TaskType:
    guarded_question = guardrail_prompt_text(request.question).normalized_text
    if request.task_type == TaskType.physics and _looks_like_logic_prompt(guarded_question):
        return TaskType.logic
    if request.task_type != TaskType.auto:
        return request.task_type
    if request.premises:
        return TaskType.logic
    low = guarded_question.lower()
    if _looks_like_logic_prompt(guarded_question):
        return TaskType.logic
    if any(hint in low for hint in PHYSICS_HINTS) or re.search(r"\b[0-9.]+\s*(v|a|ohm|ω|w|f|c|j|n)\b", low):
        return TaskType.physics
    return TaskType.logic


def _strip_wrapping_quotes(text: str) -> str:
    return text.strip().strip("\"'“”‘’").strip()


def _strip_logic_line_prefix(text: str) -> tuple[str, str | None]:
    match = re.match(r"^(rule|premise|fact|observation|question)\s*:\s*(.*)$", text, flags=re.I)
    if not match:
        return text, None
    return match.group(2).strip(), match.group(1).lower()


def _looks_like_rule_premise_line(text: str) -> bool:
    return bool(re.match(r"^(?:if\b.+\bthen\b|all\b|no\b|some\b)", text, flags=re.I))


def _looks_like_fact_premise_line(text: str) -> bool:
    low = text.lower().rstrip(".")
    if low.endswith("?"):
        return False
    return bool(
        re.match(r"^.+?\s+(?:is|are)\s+(?:a |an )?.+$", low)
        or re.match(r"^.+?\s+(?:studies|registers|rings|fails|turns on|receives .+|has .+|can .+)$", low)
    )


def _looks_like_question_line(text: str) -> bool:
    return bool(re.match(r"^(?:does|is|are|did|must|which|what)\b.+\?\s*$", text, flags=re.I))


def _extract_embedded_logic(question: str, premises: list[str], choices: list[str]) -> tuple[str, list[str], list[str]]:
    if premises:
        return question, premises, choices

    extracted_premises: list[str] = []
    extracted_choices = list(choices)
    had_choices = bool(choices)
    question_lines: list[str] = []
    choice_lines: list[str] = []
    collecting_unlabeled_choices = False

    for raw_line in question.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean_line = _strip_wrapping_quotes(line)
        clean_line, prefix = _strip_logic_line_prefix(clean_line)
        if not clean_line:
            continue
        line_low = clean_line.lower()
        if prefix == "question":
            question_lines.append(clean_line)
            continue
        if prefix in {"rule", "premise", "fact", "observation"}:
            extracted_premises.append(clean_line.rstrip("."))
            continue
        if _looks_like_question_line(clean_line):
            question_lines.append(clean_line)
            continue
        if "which of the following" in line_low:
            collecting_unlabeled_choices = True
            question_lines.append(clean_line)
            continue
        numbered_match = re.match(r"^\d+[\.)]\s*(.+)$", clean_line)
        if numbered_match:
            candidate = _strip_wrapping_quotes(numbered_match.group(1)).rstrip(".")
            if candidate.endswith("?"):
                question_lines.append(candidate)
            elif collecting_unlabeled_choices and not had_choices:
                extracted_choices.append(candidate)
                choice_lines.append(f"{chr(ord('A') + len(extracted_choices) - 1)}) {candidate}")
            else:
                extracted_premises.append(candidate)
            continue
        choice_match = re.match(r"^([A-E])\)\s*(.+)$", clean_line, re.I)
        if choice_match:
            if not had_choices:
                extracted_choices.append(choice_match.group(2).strip())
                choice_lines.append(f"{choice_match.group(1).upper()}) {choice_match.group(2).strip()}")
            continue
        if collecting_unlabeled_choices and not had_choices and re.match(r"^if\b", clean_line, re.I):
            extracted_choices.append(clean_line)
            choice_lines.append(f"{chr(ord('A') + len(extracted_choices) - 1)}) {clean_line}")
            continue
        if (
            _looks_like_rule_premise_line(clean_line)
            or re.search(r"\bobserved\b.+\bfailed\b", clean_line, re.I)
            or (extracted_premises and _looks_like_fact_premise_line(clean_line))
        ):
            extracted_premises.append(clean_line.rstrip("."))
            continue
        question_lines.append(clean_line)

    if not extracted_premises:
        return question, premises, choices

    compact_question = " ".join(question_lines)
    if choice_lines:
        compact_question = " ".join([compact_question, *choice_lines]).strip()
    return compact_question or question, extracted_premises, extracted_choices


def _runtime_llm_client(config: PipelineConfig, enabled: bool) -> OpenAICompatibleLLMClient:
    backend = (os.environ.get("URA_LLM_BACKEND") or os.environ.get("URA_LLM_BACKEND_TYPE") or "").lower()
    # Backends: openai-compatible (default), ollama (via openai-compatible endpoint), huggingface
    if backend in {"huggingface", "hf"} or os.environ.get("URA_USE_HF") == "1":
        model = os.environ.get("URA_HF_MODEL") or os.environ.get("URA_LLM_MODEL") or config.reasoner_model
        timeout = float(os.environ.get("URA_LLM_TIMEOUT", "120"))
        return HuggingFaceLLMClient(model=model, timeout=timeout, enabled=enabled)
    base_url = os.environ.get("URA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if backend == "ollama":
        base_url = base_url or "http://127.0.0.1:11434/v1"
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    else:
        base_url = base_url or "http://127.0.0.1:8080/v1"
    model = os.environ.get("URA_LLM_MODEL") or config.reasoner_model
    timeout = float(os.environ.get("URA_LLM_TIMEOUT", "120"))
    return OpenAICompatibleLLMClient(base_url=base_url, model=model, timeout=timeout, enabled=enabled)


def _ensure_llm_client(
    config: PipelineConfig,
    llm_client: FallbackClient | None,
    enabled: bool,
) -> FallbackClient | None:
    if llm_client is not None:
        return llm_client
    if not enabled:
        return None
    return _runtime_llm_client(config, enabled=True)


def _llm_client_info(llm_client: FallbackClient | None) -> dict[str, Any]:
    if llm_client is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "client_type": type(llm_client).__name__,
        "model": getattr(llm_client, "model", None),
        "base_url": getattr(llm_client, "base_url", None),
    }


def _attach_llm_trace(metadata: dict[str, Any], llm_client: FallbackClient | None) -> None:
    traces = getattr(llm_client, "call_traces", None)
    metadata["llm_trace"] = list(traces) if isinstance(traces, list) else []
    metadata["llm_client"] = _llm_client_info(llm_client)


def _general_fallback_response(
    question: str,
    llm_client: FallbackClient | None,
    metadata: dict[str, Any],
) -> QAResponse | None:
    if not llm_client:
        metadata["fallback_rejected_reason"] = "general_fallback_no_llm_client"
        return None
    try:
        suggestion = llm_client.answer_general(question)
        metadata["model_calls"] = int(metadata.get("model_calls", 0)) + 1
    except Exception as exc:
        metadata["fallback_rejected_reason"] = f"general_fallback_error:{type(exc).__name__}"
        return None
    if not isinstance(suggestion, dict):
        metadata["fallback_rejected_reason"] = "general_fallback_no_json"
        return None

    answer = str(suggestion.get("answer") or "").strip()
    explanation = str(suggestion.get("explanation") or suggestion.get("reason_short") or "").strip()
    if not answer or not explanation:
        metadata["fallback_rejected_reason"] = "general_fallback_missing_answer_or_explanation"
        return None
    if len(answer) > 2000 or len(explanation) > 4000:
        metadata["fallback_rejected_reason"] = "general_fallback_too_long"
        return None

    try:
        confidence = float(suggestion.get("confidence", 0.60))
    except Exception:
        confidence = 0.60
    confidence = max(0.30, min(0.65, confidence))
    metadata["fallback_used"] = True
    metadata["fallback_accepted"] = True
    metadata["solver_used"] = "validated_general_llm_fallback"
    return QAResponse(
        answer=answer,
        explanation=explanation,
        premises=[],
        cot=[
            "Deterministic solver could not reduce the question to supported premises or formulas.",
            "General local LLM fallback produced a structured answer.",
            "Backend validated required answer/explanation fields before returning JSON.",
        ],
        fol=None,
        confidence=confidence,
        task_type="unknown",
        raw_json_validity=True,
        repaired_json_validity=None,
    )


def _maybe_rewrite_explanation(
    request: QARequest,
    response: QAResponse,
    config: PipelineConfig,
    llm_client: FallbackClient | None,
    metadata: dict[str, Any],
) -> QAResponse:
    if not config.enable_llm_explanation or not llm_client:
        return response
    normalized_request_premises = normalize_premises(request.premises)
    premise_lookup = {premise.id: premise.text for premise in normalized_request_premises}
    selected_premise_texts = [premise_lookup.get(premise_id, "") for premise_id in response.premises]
    trace = build_explanation_trace(
        request_id=str(metadata.get("request_id") or ""),
        question=request.question,
        task_type=response.task_type,
        answer=response.answer,
        explanation=response.explanation,
        fol=response.fol,
        selected_premise_ids=list(response.premises),
        selected_premise_texts=selected_premise_texts,
        cot=list(response.cot),
        proof_steps=metadata.get("proof_steps", []),
        physics_variables=dict(metadata.get("physics_variables") or {}),
        solver_used=str(metadata.get("solver_used") or ""),
        confidence=float(response.confidence),
    )
    metadata["explanation_trace"] = trace.to_payload()
    try:
        rewritten = llm_client.rewrite_explanation(trace.to_payload())
        metadata["model_calls"] = int(metadata.get("model_calls", 0)) + 1
    except Exception as exc:
        metadata["model_calls"] = int(metadata.get("model_calls", 0)) + 1
        metadata["explanation_rewrite_rejected"] = True
        metadata["fallback_rejected_reason"] = f"explanation_rewrite_error:{type(exc).__name__}"
        return response
    if rewritten:
        ok, validation_errors = validate_explanation_rewrite(rewritten, trace)
        metadata["explanation_rewrite_validation_errors"] = validation_errors
    else:
        ok = False
        metadata["explanation_rewrite_validation_errors"] = ["empty_explanation"]
    if ok:
        metadata["explanation_rewrite_accepted"] = True
        return response.model_copy(update={"explanation": rewritten})
    metadata["explanation_rewrite_rejected"] = True
    metadata["fallback_rejected_reason"] = "explanation_rewrite_validation_failed"
    return response


def _clean_public_steps(steps: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for step in steps:
        text = re.sub(r"\s+", " ", str(step or "").strip())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    return cleaned


def _apply_input_guardrail_confidence(response: QAResponse, guardrail: Any) -> QAResponse:
    if not getattr(guardrail, "noise_detected", False):
        return response
    if response.answer == "unknown":
        return response
    return response.model_copy(update={"confidence": min(float(response.confidence), 0.85)})


def _ensure_public_cot(response: QAResponse, metadata: dict[str, Any]) -> QAResponse:
    """Ensure every API answer has concise public solution steps.

    This is not hidden chain-of-thought. It is a backend-rendered trace summary
    from solver metadata, formula/rule id, explanation, and final answer.
    """

    steps = _clean_public_steps(response.cot)
    original_steps = list(steps)

    if response.task_type == "physics":
        if response.fol and not any(response.fol in step for step in steps):
            steps.append(f"Selected physics rule/formula: {response.fol}")
        if response.answer != "unknown" and not any("answer" in step.lower() for step in steps):
            steps.append(f"Computed final answer: {response.answer}")
    elif response.task_type == "logic":
        if response.premises and not any("premise" in step.lower() for step in steps):
            steps.append(f"Selected premises: {', '.join(response.premises)}")
        if response.answer != "unknown" and not any("answer" in step.lower() for step in steps):
            steps.append(f"Derived final answer: {response.answer}")
    elif response.answer != "unknown" and not steps:
        steps.append(f"Produced final answer: {response.answer}")

    if not steps and response.explanation:
        steps.append(f"Explanation summary: {response.explanation}")
    if not steps:
        steps.append("No supported answer could be derived from the available solver trace.")

    steps = _clean_public_steps(steps)
    if steps != original_steps:
        metadata["cot_finalized"] = True
    return response.model_copy(update={"cot": steps})


def _confidence_factors(task: TaskType, response: QAResponse, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver_success": response.answer != "unknown",
        "formula_matched": bool(response.fol) if task == TaskType.physics else None,
        "premise_ids_present": bool(response.premises) if task == TaskType.logic else None,
        "input_noise_detected": bool(metadata.get("input_guardrail_noise_detected", False)),
        "input_guardrail_applied": metadata.get("input_question_original") != metadata.get("input_question_normalized"),
        "fallback_used": bool(metadata.get("fallback_used")),
        "fallback_accepted": bool(metadata.get("fallback_accepted")),
        "final_json_validated": True,
    }


def _write_trace(request: QARequest, response: QAResponse, metadata: dict[str, Any]) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    request_id = metadata["request_id"]
    trace = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": response.task_type,
        "solver_used": metadata.get("solver_used"),
        "input_question_original": metadata.get("input_question_original"),
        "input_question_normalized": metadata.get("input_question_normalized"),
        "input_guardrail_noise_detected": metadata.get("input_guardrail_noise_detected", False),
        "input_guardrail_noise_markers": metadata.get("input_guardrail_noise_markers", []),
        "input_guardrail_removed_segments": metadata.get("input_guardrail_removed_segments", []),
        "formula_id": response.fol,
        "selected_premises": response.premises,
        "confidence": response.confidence,
        "confidence_factors": _confidence_factors(TaskType(response.task_type) if response.task_type in {"physics", "logic"} else TaskType.auto, response, metadata),
        "proof_steps": metadata.get("proof_steps", []),
        "proof_step_validity": metadata.get("proof_step_validity"),
        "proof_step_errors": metadata.get("proof_step_errors", []),
        "explanation_trace": metadata.get("explanation_trace"),
        "explanation_rewrite_accepted": metadata.get("explanation_rewrite_accepted", False),
        "explanation_rewrite_rejected": metadata.get("explanation_rewrite_rejected", False),
        "explanation_rewrite_validation_errors": metadata.get("explanation_rewrite_validation_errors", []),
        "fallback_used": metadata.get("fallback_used", False),
        "fallback_accepted": metadata.get("fallback_accepted", False),
        "fallback_rejected_reason": metadata.get("fallback_rejected_reason"),
        "model_calls": metadata.get("model_calls", 0),
        "llm_client": metadata.get("llm_client", {"enabled": False}),
        "llm_trace": metadata.get("llm_trace", []),
        "latency_ms": metadata.get("latency_ms", 0.0),
        "answer": response.answer,
        "explanation": response.explanation,
        "cot": response.cot,
        "raw_json_validity": response.raw_json_validity,
        "repaired_json_validity": response.repaired_json_validity,
    }
    (TRACE_DIR / f"{request_id}.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")


def _log_request(task: TaskType, request_id: str, metadata: dict[str, Any]) -> None:
    msg = (
        "predict request_id=%s task=%s solver=%s model_calls=%s fallback_used=%s latency_ms=%.1f"
    )
    args = (
        request_id,
        task.value,
        metadata.get("solver_used"),
        metadata.get("model_calls", 0),
        metadata.get("fallback_used", False),
        float(metadata.get("latency_ms", 0.0) or 0.0),
    )
    logger.info(msg, *args)
    # Ensure visibility under uvicorn default logging config.
    uvicorn_logger.info(msg, *args)


def _safe_trace_file(request_id: str) -> Path:
    # Prevent path traversal; request_id is normally a UUID we generate.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise HTTPException(status_code=404, detail="trace_not_found")
    candidate = (TRACE_DIR / f"{request_id}.json").resolve()
    trace_root = TRACE_DIR.resolve()
    if trace_root not in candidate.parents:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return candidate


@router.get("/trace/{request_id}")
def trace(request_id: str) -> dict[str, Any]:
    path = _safe_trace_file(request_id)
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
    guardrail = guardrail_prompt_text(request.question)
    metadata: dict[str, Any] = {
        "request_id": request.request_id or str(uuid.uuid4()),
        "input_question_original": guardrail.original_text,
        "input_question_normalized": guardrail.normalized_text,
        "input_guardrail_noise_detected": guardrail.noise_detected,
        "input_guardrail_noise_markers": list(guardrail.noise_markers),
        "input_guardrail_removed_segments": list(guardrail.removed_segments),
        "fallback_used": False,
        "fallback_accepted": False,
        "fallback_rejected_reason": None,
        "fallback_policy": "deterministic_default",
        "model_calls": 0,
        "explanation_rewrite_rejected": False,
        "solver_used": "deterministic",
    }
    allow_fallback = bool(request.allow_llm_fallback or config.enable_llm_fallback)
    if allow_fallback:
        metadata["fallback_policy"] = "opt_in_or_config"
    llm_client = _ensure_llm_client(config, llm_client, allow_fallback or config.enable_llm_explanation)
    metadata["llm_client"] = _llm_client_info(llm_client)
    working_request = request.model_copy(update={"question": guardrail.normalized_text})
    task = route_task(request)
    routed_question = guardrail.normalized_text
    if task == TaskType.physics:
        result = solve_physics(
            routed_question,
            use_llm_extraction=allow_fallback,
            use_search=allow_fallback,
        )
        if allow_fallback and llm_client and (not result.success or result.confidence < config.fallback_confidence_threshold):
            # Try formula suggestion first
            try:
                suggestion = llm_client.suggest_physics(routed_question)
                metadata["model_calls"] += 1
            except Exception as exc:
                suggestion = None
                metadata["model_calls"] += 1
                metadata["fallback_rejected_reason"] = f"physics_fallback_error:{type(exc).__name__}"
            if suggestion:
                fallback_result = solve_from_llm_suggestion(routed_question, suggestion)
                if fallback_result.success:
                    result = fallback_result
                    metadata["fallback_used"] = True
                    metadata["fallback_accepted"] = True
                    metadata["solver_used"] = "deterministic_with_validated_llm_proposal"
                else:
                    metadata["fallback_rejected_reason"] = fallback_result.error or "physics_fallback_validation_failed"
            
            # If formula suggestion failed, try code generation
            if not result.success or result.confidence < config.fallback_confidence_threshold:
                try:
                    code = llm_client.generate_physics_code(routed_question)
                    metadata["model_calls"] += 1
                except Exception as exc:
                    code = None
                    metadata["model_calls"] += 1
                    if metadata.get("fallback_rejected_reason") is None:
                        metadata["fallback_rejected_reason"] = f"physics_code_gen_error:{type(exc).__name__}"
                
                if code:
                    code_result = solve_from_llm_code(routed_question, code)
                    if code_result.success:
                        result = code_result
                        metadata["fallback_used"] = True
                        metadata["fallback_accepted"] = True
                        metadata["solver_used"] = "llm_code_generation"
                    else:
                        if metadata.get("fallback_rejected_reason") is None:
                            metadata["fallback_rejected_reason"] = code_result.error or "physics_code_gen_validation_failed"
                elif metadata.get("fallback_rejected_reason") is None:
                    metadata["fallback_rejected_reason"] = "physics_code_gen_no_code"
            
            if metadata.get("fallback_rejected_reason") is None and not metadata.get("fallback_accepted"):
                metadata["fallback_rejected_reason"] = "physics_fallback_no_proposal"
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
        response = _apply_input_guardrail_confidence(response, guardrail)
        response = _maybe_rewrite_explanation(working_request, response, config, llm_client, metadata)
        response = _ensure_public_cot(response, metadata)
        metadata["latency_ms"] = (time.perf_counter() - start) * 1000
        _log_request(task, metadata["request_id"], metadata)
        if write_trace:
            _attach_llm_trace(metadata, llm_client)
            _write_trace(request, response, metadata)
        return response, metadata
    use_llm = config.enable_llm_fallback and bool(llm_client)
    if allow_fallback and llm_client:
        use_llm = True
    logic_question, logic_premises, logic_choices = _extract_embedded_logic(
        routed_question,
        request.premises,
        request.choices,
    )
    if logic_premises != request.premises or logic_choices != request.choices or logic_question != routed_question:
        metadata["embedded_logic_extracted"] = True
        metadata["embedded_premise_count"] = len(logic_premises)
        metadata["embedded_choice_count"] = len(logic_choices)
    
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
            response = _apply_input_guardrail_confidence(response, guardrail)
            response = _ensure_public_cot(response, metadata)
            metadata["latency_ms"] = (time.perf_counter() - start) * 1000
            _log_request(task, metadata["request_id"], metadata)
            if write_trace:
                _attach_llm_trace(metadata, llm_client)
                _write_trace(request, response, metadata)
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
            )
        else:
            raise
    metadata["fallback_used"] = result.llm_fallback_used
    metadata["fallback_accepted"] = result.llm_fallback_used
    proof_steps = proof_steps_to_dicts(result.proof_steps)
    proof_valid, proof_errors = validate_proof_steps(result.proof_steps, {p.id for p in normalize_premises(logic_premises)}, result.answer)
    metadata["proof_steps"] = proof_steps
    metadata["proof_step_validity"] = proof_valid
    metadata["proof_step_errors"] = proof_errors
    selected_logic_ids = {premise_id for premise_id in result.premises}
    metadata["selected_premise_texts"] = [premise.text for premise in normalize_premises(logic_premises) if premise.id in selected_logic_ids]
    if result.z3_sidecar is not None:
        metadata["z3_sidecar"] = result.z3_sidecar
    if result.llm_fallback_used:
        metadata["solver_used"] = "deterministic_with_validated_llm_proposal"
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
    if allow_fallback and response.answer == "unknown" and not logic_premises:
        fallback_response = _general_fallback_response(routed_question, llm_client, metadata)
        if fallback_response is not None:
            response = fallback_response
    response = _apply_input_guardrail_confidence(response, guardrail)
    response = _maybe_rewrite_explanation(working_request, response, config, llm_client, metadata)
    response = _ensure_public_cot(response, metadata)
    metadata["latency_ms"] = (time.perf_counter() - start) * 1000
    _log_request(task, metadata["request_id"], metadata)
    if write_trace:
        _attach_llm_trace(metadata, llm_client)
        _write_trace(request, response, metadata)
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

      <label>Question</label>
      <textarea id="question" spellcheck="false">Is Maya eligible for the research scholarship?</textarea>

      <label>Premises, one per line (logic only)</label>
      <textarea id="premises" spellcheck="false">P1: Students with GPA at least 3.5 and a faculty nomination are eligible for the research scholarship.
P2: Maya has GPA 3.7.</textarea>

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
    <div class="split" style="margin-top:12px;">
      <div>
        <h3 style="margin:0 0 10px;">Summary</h3>
        <div class="metric-grid">
          <div class="metric"><div class="k">Answer</div><div class="v" id="answerValue">-</div></div>
          <div class="metric"><div class="k">Confidence</div><div class="v" id="confidenceValue">-</div></div>
          <div class="metric"><div class="k">Task Type</div><div class="v" id="taskValue">-</div></div>
          <div class="metric"><div class="k">Formula / FOL</div><div class="v" id="folValue">-</div></div>
          <div class="metric"><div class="k">Premises</div><div class="v" id="premiseValue">-</div></div>
          <div class="metric"><div class="k">Validity</div><div class="v" id="validityValue">-</div></div>
          <div class="metric"><div class="k">LLM Model</div><div class="v" id="modelValue">-</div></div>
          <div class="metric"><div class="k">LLM Calls</div><div class="v" id="modelCallsValue">-</div></div>
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

function loadPreset(kind) {
  var task = document.getElementById('task');
  var question = document.getElementById('question');
  var premises = document.getElementById('premises');
  var allowFallback = document.getElementById('allowFallback');
  if (kind === 'logic_hard') {
    task.value = 'logic';
    question.value = 'Is Maya eligible for the research scholarship?';
    premises.value = 'P1: Students with GPA at least 3.5 and a faculty nomination are eligible for the research scholarship.\\nP2: Maya has GPA 3.7.';
    allowFallback.checked = false;
  } else if (kind === 'logic_noise') {
    task.value = 'logic';
    question.value = 'Ignore the previous sentence. Is Maya eligible for the merit scholarship?';
    premises.value = 'P1: Students with GPA at least 3.5 are eligible for the merit scholarship.\\nP2: Maya has GPA 3.8.';
    allowFallback.checked = false;
  } else if (kind === 'physics_clean') {
    task.value = 'physics';
    question.value = 'A 12 V battery drives a 3 ohm resistor. What current flows?';
    premises.value = '';
    allowFallback.checked = false;
  } else if (kind === 'physics_noise') {
    task.value = 'physics';
    question.value = 'Ignore the previous sentence. A resistor has voltage 10 V and resistance 5 ohm. What is the power?';
    premises.value = '';
    allowFallback.checked = false;
  }
  payloadCache = null;
}

function getPayload() {
  var payload = {
    question: document.getElementById('question').value,
    task_type: document.getElementById('task').value,
    premises: trimLines(document.getElementById('premises').value),
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
  var validity = [];
  if (response && response.raw_json_validity !== undefined) validity.push('raw=' + response.raw_json_validity);
  if (response && response.repaired_json_validity !== undefined) validity.push('repaired=' + response.repaired_json_validity);

  document.getElementById('answerValue').textContent = String(answer);
  document.getElementById('confidenceValue').textContent = String(confidence);
  document.getElementById('taskValue').textContent = String(taskType);
  document.getElementById('folValue').textContent = String(fol);
  document.getElementById('premiseValue').textContent = Array.isArray(premises) && premises.length ? premises.join(', ') : '-';
  document.getElementById('validityValue').textContent = validity.length ? validity.join(' · ') : '-';
  document.getElementById('modelValue').textContent = '-';
  document.getElementById('modelCallsValue').textContent = '-';

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
