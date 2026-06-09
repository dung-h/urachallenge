from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.explanation_worker import build_explanation_trace, validate_explanation_rewrite
from app.logic.premise_selector import normalize_premises
from app.llm_client import FallbackClient
from app.runtime_workflow import NormalizedRequest, RuntimeTrace, build_runtime_trace
from app.schemas import QARequest, QAResponse, TaskType


logger = logging.getLogger("ura")
uvicorn_logger = logging.getLogger("uvicorn.error")


def attach_llm_trace(metadata: dict[str, Any], llm_client: FallbackClient | None) -> None:
    traces = getattr(llm_client, "call_traces", None)
    metadata["llm_trace"] = list(traces) if isinstance(traces, list) else []
    from app.runtime_clients import llm_client_info

    metadata["llm_client"] = llm_client_info(llm_client)


def maybe_rewrite_explanation(
    request: QARequest,
    response: QAResponse,
    llm_client: FallbackClient | None,
    metadata: dict[str, Any],
) -> QAResponse:
    if not llm_client:
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
    if not metadata.get("fallback_rejected_reason"):
        metadata["fallback_rejected_reason"] = "explanation_rewrite_validation_failed"
    return response


def clean_public_steps(steps: list[str]) -> list[str]:
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


def apply_input_guardrail_confidence(response: QAResponse, guardrail: Any) -> QAResponse:
    if not getattr(guardrail, "noise_detected", False):
        return response
    if response.answer == "unknown":
        return response
    return response.model_copy(update={"confidence": min(float(response.confidence), 0.85)})


def physics_search_used(result: Any) -> bool:
    return bool(getattr(result, "fallback_used", False) and getattr(result, "model_calls", 0) == 0)


def ensure_public_cot(response: QAResponse, metadata: dict[str, Any]) -> QAResponse:
    """Ensure every API answer has concise public solution steps."""

    steps = clean_public_steps(response.cot)
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

    steps = clean_public_steps(steps)
    if steps != original_steps:
        metadata["cot_finalized"] = True
    return response.model_copy(update={"cot": steps})


def confidence_factors(task: TaskType, response: QAResponse, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver_success": response.answer != "unknown",
        "formula_matched": bool(response.fol) if task == TaskType.physics else None,
        "premise_ids_present": bool(response.premises) if task == TaskType.logic else None,
        "premise_coverage": metadata.get("premise_coverage"),
        "proof_valid": metadata.get("proof_step_validity") if task == TaskType.logic else None,
        "unit_or_formula_verified": bool(response.fol and response.answer != "unknown") if task == TaskType.physics else None,
        "ambiguity": metadata.get("ambiguity", []),
        "input_noise_detected": bool(metadata.get("input_guardrail_noise_detected", False)),
        "input_guardrail_applied": metadata.get("input_question_original") != metadata.get("input_question_normalized"),
        "fallback_used": bool(metadata.get("fallback_used")),
        "fallback_accepted": bool(metadata.get("fallback_accepted")),
        "final_json_validated": True,
    }


def write_trace(trace_dir: Path, normalized: NormalizedRequest, response: QAResponse, metadata: dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    request_id = metadata["request_id"]
    task = TaskType(response.task_type) if response.task_type in {"physics", "logic"} else TaskType.auto
    metadata["confidence_factors"] = confidence_factors(task, response, metadata)
    trace = RuntimeTrace(**build_runtime_trace(request_id, normalized, response, metadata).model_dump())
    (trace_dir / f"{request_id}.json").write_text(json.dumps(trace.model_dump(), indent=2, sort_keys=True) + "\n")


def log_request(task: TaskType, request_id: str, metadata: dict[str, Any]) -> None:
    msg = "predict request_id=%s task=%s solver=%s model_calls=%s fallback_used=%s latency_ms=%.1f"
    args = (
        request_id,
        task.value,
        metadata.get("solver_used"),
        metadata.get("model_calls", 0),
        metadata.get("fallback_used", False),
        float(metadata.get("latency_ms", 0.0) or 0.0),
    )
    logger.info(msg, *args)
    uvicorn_logger.info(msg, *args)


def safe_trace_file(trace_dir: Path, request_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise ValueError("trace_not_found")
    candidate = (trace_dir / f"{request_id}.json").resolve()
    trace_root = trace_dir.resolve()
    if trace_root not in candidate.parents:
        raise ValueError("trace_not_found")
    return candidate
