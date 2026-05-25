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

from app.logic.premise_selector import normalize_premises
from app.logic.solver import solve as solve_logic
from app.logic.proof_trace import proof_steps_to_dicts, validate_proof_steps
from app.explanation_worker import build_explanation_trace, validate_explanation_rewrite
# NOTE: hybrid_solver is optional (depends on extra packages like z3). We import it lazily.
from app.llm_client import FallbackClient, OpenAICompatibleLLMClient, HuggingFaceLLMClient
from app.physics.solver import solve as solve_physics, solve_from_llm_suggestion, solve_from_llm_code
from app.physics.web_search import search_formula_context
from app.pipeline_config import PipelineConfig, load_pipeline_config
from app.schemas import QARequest, QAResponse, TaskType


router = APIRouter()
ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "outputs" / "traces" / "production_like"

# Bumped manually when the /demo UI changes, to make cache/debug obvious.
DEMO_BUILD_ID = "2026-05-20b"

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
    if request.task_type == TaskType.physics and _looks_like_logic_prompt(request.question):
        return TaskType.logic
    if request.task_type != TaskType.auto:
        return request.task_type
    if request.premises:
        return TaskType.logic
    low = request.question.lower()
    if _looks_like_logic_prompt(request.question):
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
    base_url = os.environ.get("URA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8080/v1"
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


def _general_fallback_response(
    request: QARequest,
    llm_client: FallbackClient | None,
    metadata: dict[str, Any],
) -> QAResponse | None:
    if not llm_client:
        metadata["fallback_rejected_reason"] = "general_fallback_no_llm_client"
        return None
    try:
        suggestion = llm_client.answer_general(request.question)
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
        "physics_search_used": metadata.get("physics_search_used", False),
        "physics_search_cache_hit": metadata.get("physics_search_cache_hit"),
        "physics_search_query": metadata.get("physics_search_query"),
        "physics_search_sources": metadata.get("physics_search_sources", []),
        "physics_search_error": metadata.get("physics_search_error"),
        "model_calls": metadata.get("model_calls", 0),
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
    metadata: dict[str, Any] = {
        "request_id": request.request_id or str(uuid.uuid4()),
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
    task = route_task(request)
    if task == TaskType.physics:
        result = solve_physics(
            request.question,
            use_llm_extraction=allow_fallback,
            use_search=allow_fallback,
        )
        if allow_fallback and llm_client and (not result.success or result.confidence < config.fallback_confidence_threshold):
            formula_context: str | None = None
            if config.enable_physics_web_search:
                try:
                    search_result = search_formula_context(request.question)
                except Exception as exc:
                    search_result = None
                    metadata["physics_search_error"] = f"{type(exc).__name__}:{exc}"
                if search_result is not None:
                    formula_context = search_result.context
                    metadata["physics_search_used"] = True
                    metadata["physics_search_cache_hit"] = search_result.cache_hit
                    metadata["physics_search_query"] = search_result.search_query
                    metadata["physics_search_cache_key"] = search_result.cache_key
                    metadata["physics_search_sources"] = search_result.sources
                else:
                    metadata["physics_search_used"] = False
            # Try formula suggestion first
            try:
                suggestion = llm_client.suggest_physics(request.question)
                metadata["model_calls"] += 1
            except Exception as exc:
                suggestion = None
                metadata["model_calls"] += 1
                metadata["fallback_rejected_reason"] = f"physics_fallback_error:{type(exc).__name__}"
            if suggestion:
                fallback_result = solve_from_llm_suggestion(request.question, suggestion)
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
                    code = llm_client.generate_physics_code(request.question, formula_context=formula_context)
                    metadata["model_calls"] += 1
                except Exception as exc:
                    code = None
                    metadata["model_calls"] += 1
                    if metadata.get("fallback_rejected_reason") is None:
                        metadata["fallback_rejected_reason"] = f"physics_code_gen_error:{type(exc).__name__}"
                
                if code:
                    code_result = solve_from_llm_code(request.question, code)
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
        response = _maybe_rewrite_explanation(request, response, config, llm_client, metadata)
        response = _ensure_public_cot(response, metadata)
        metadata["latency_ms"] = (time.perf_counter() - start) * 1000
        _log_request(task, metadata["request_id"], metadata)
        if write_trace:
            _write_trace(request, response, metadata)
        return response, metadata
    use_llm = config.enable_llm_fallback and bool(llm_client)
    if allow_fallback and llm_client:
        use_llm = True
    logic_question, logic_premises, logic_choices = _extract_embedded_logic(
        request.question,
        request.premises,
        request.choices,
    )
    if logic_premises != request.premises or logic_choices != request.choices or logic_question != request.question:
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
            metadata["model_calls"] += 1
            if "z3" in hybrid_result.method:
                metadata["model_calls"] += 1
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
            response = _ensure_public_cot(response, metadata)
            metadata["latency_ms"] = (time.perf_counter() - start) * 1000
            _log_request(task, metadata["request_id"], metadata)
            if write_trace:
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
        fallback_response = _general_fallback_response(request, llm_client, metadata)
        if fallback_response is not None:
            response = fallback_response
    response = _maybe_rewrite_explanation(request, response, config, llm_client, metadata)
    response = _ensure_public_cot(response, metadata)
    metadata["latency_ms"] = (time.perf_counter() - start) * 1000
    _log_request(task, metadata["request_id"], metadata)
    if write_trace:
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
  <title>URA EXACT Flow Demo</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, Arial, sans-serif; background: #0b1020; color: #e8ecff; }
    body { margin: 0; padding: 28px; }
    main { max-width: 1120px; margin: 0 auto; display: grid; gap: 18px; }
    .hero { border: 1px solid #29345d; border-radius: 18px; padding: 22px; background: linear-gradient(135deg, #111a36, #101827 65%, #172219); }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 52px); letter-spacing: -0.04em; }
    p { color: #aeb8df; line-height: 1.5; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } body { padding: 14px; } }
    .card { border: 1px solid #283354; border-radius: 16px; padding: 16px; background: #11182b; box-shadow: 0 20px 80px #0005; }
    label { display: block; font-size: 13px; color: #8ea0d8; margin: 10px 0 6px; }
    textarea, select, input { width: 100%; box-sizing: border-box; border: 1px solid #33416e; background: #080d1b; color: #f6f8ff; border-radius: 12px; padding: 12px; font: inherit; }
    textarea { min-height: 140px; resize: vertical; }
    button { margin-top: 12px; border: 0; border-radius: 999px; padding: 12px 18px; font-weight: 700; color: #061018; background: #7cf7c7; cursor: pointer; }
    button.secondary { background: #9db7ff; margin-left: 8px; }
    .flow { display: grid; gap: 10px; }
    .step { border: 1px solid #31406c; border-radius: 14px; padding: 12px; background: #0d1426; }
    .step strong { color: #7cf7c7; }
    pre { white-space: pre-wrap; word-break: break-word; background: #070b15; border: 1px solid #253157; border-radius: 14px; padding: 14px; color: #dce5ff; max-height: 420px; overflow: auto; }
    .pill { display: inline-block; margin: 4px 6px 0 0; padding: 5px 9px; border-radius: 999px; background: #1c294d; color: #b9c7ff; font-size: 12px; }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>URA EXACT Flow Demo</h1>
    <p>Deterministic Python remains answer authority. LLMs are optional workers for extraction, proposals, code, or explanation. Final JSON is backend assembled and Pydantic validated.</p>
    <span class="pill">solver-first</span><span class="pill">validated JSON</span><span class="pill">LLM proposal-only</span>
    <span class="pill">build: __DEMO_BUILD_ID__</span>
  </section>
  <section class="grid">
    <div class="card">
      <label>Task type</label>
      <select id="task"><option value="auto">auto</option><option value="physics">physics</option><option value="logic">logic</option></select>
      <label>Question</label>
      <textarea id="question">In a resonant RLC circuit, the measured impedance is Z=40 Ω. Determine the pure resistance R.</textarea>
      <label>Premises, one per line (logic only)</label>
      <textarea id="premises">P1: All mammals are animals.
P2: All dogs are mammals.
P3: Fido is a dog.</textarea>
      <label><input id="allowFallback" type="checkbox" checked /> Allow LLM fallback</label>
      <button onclick="runExample('physics')">Physics Example</button>
      <button class="secondary" onclick="runExample('logic')">Logic Example</button>
      <button onclick="predict()">Run /predict</button>
    </div>
    <div class="card flow">
      <div class="step"><strong>1. Router</strong><br/>Detect physics from units/formula words, or logic from premises.</div>
      <div class="step"><strong>2. Physics Path</strong><br/>parse variables and units → choose formula → compute with Python → format answer.</div>
      <div class="step"><strong>3. Logic Path</strong><br/>normalize premise IDs → select evidence → rule/MCQ/entailment baseline → proof trace.</div>
      <div class="step"><strong>4. LLM Role</strong><br/>If enabled in config, LLM suggests structure/code/explanation. Backend validates before accepting.</div>
      <div class="step"><strong>5. Final JSON</strong><br/>Backend creates `answer`, `explanation`, `premises`, `cot`, `fol`, `confidence`; Pydantic validates.</div>
    </div>
  </section>
  <section class="card">
    <h2>Response</h2>
    <div id="status" class="pill" style="background:#2a1c1c;color:#ffb3b3;">API: unknown</div>
    <noscript>
      <div class="pill" style="background:#2a1c1c;color:#ffb3b3;">JavaScript is disabled/blocked, so API status cannot update.</div>
    </noscript>
    <pre id="out">Run an example.</pre>
  </section>
</main>
<script>
// demo_build_id: __DEMO_BUILD_ID__
var ORIGIN = window.location.origin;
var out = document.getElementById('out');
var statusEl = document.getElementById('status');

function setStatus(ok, text) {
  statusEl.textContent = text;
  statusEl.style.background = ok ? '#163326' : '#2a1c1c';
  statusEl.style.color = ok ? '#bfffe0' : '#ffb3b3';
}

function show(obj) {
  try {
    out.textContent = JSON.stringify(obj, null, 2);
  } catch (e) {
    out.textContent = String(obj);
  }
}

window.addEventListener('error', function (e) {
  show({ error: 'ui_error', origin: ORIGIN, message: String(e && e.message ? e.message : e) });
});
window.addEventListener('unhandledrejection', function (e) {
  show({ error: 'ui_unhandled_rejection', origin: ORIGIN, message: String(e && e.reason ? e.reason : e) });
});

out.textContent = 'UI loaded. Checking /health...';
setStatus(false, 'API: checking');

fetch('/health')
  .then(function (res) {
    return res.text().then(function (text) {
      if (!res.ok) {
        setStatus(false, 'API: ' + res.status);
        show({ error: 'health_not_ok', status: res.status, origin: ORIGIN, body_preview: text.slice(0, 2000) });
        return;
      }
      setStatus(true, 'API: ok');
      out.textContent = 'Ready. UI origin: ' + ORIGIN;
    });
  })
  .catch(function (err) {
    setStatus(false, 'API: unreachable');
    show({ error: 'health_request_failed', origin: ORIGIN, details: String(err) });
  });

function runExample(kind) {
  document.getElementById('task').value = kind;
  if (kind === 'physics') {
    document.getElementById('question').value = 'A 12 V battery drives a 3 ohm resistor. What current flows?';
    document.getElementById('premises').value = '';
  } else {
    document.getElementById('question').value = 'Is Fido an animal?';
    // Use literal "\n" so JS can split lines reliably.
    document.getElementById('premises').value = 'P1: All mammals are animals.\nP2: All dogs are mammals.\nP3: Fido is a dog.';
  }
}
function predict() {
  // Use literal "\n" (do not embed an actual newline in the JS string).
  var premisesRaw = document.getElementById('premises').value.split('\n');
  var premises = [];
  for (var i = 0; i < premisesRaw.length; i++) {
    // Regex whitespace trim.
    var t = String(premisesRaw[i]).replace(/^\s+|\s+$/g, '');
    if (t) premises.push(t);
  }
  var payload = {
    question: document.getElementById('question').value,
    task_type: document.getElementById('task').value,
    premises: premises,
    allow_llm_fallback: document.getElementById('allowFallback').checked
  };
  out.textContent = 'Running...';

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
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
            hint: 'If the body mentions /dvwa or /juice, you opened the wrong service/port. Open the API UI at http://127.0.0.1:<api_port>/demo.',
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

        var envelope = {
          request_id: requestId,
          trace_url: traceUrl,
          response: data,
        };

        out.textContent = JSON.stringify(envelope, null, 2);
        if (traceUrl && requestId) {
          out.textContent += '\n\nTip: open ' + traceUrl + ' to see server-side trace JSON for request_id=' + requestId;
        }
      });
    })
    .catch(function (err) {
      show({ error: 'request_failed', origin: ORIGIN, details: String(err) });
    });
}
</script>
</body>
</html>
"""

    return html.replace("__DEMO_BUILD_ID__", DEMO_BUILD_ID)
