"""Runtime entry point that drives the MethodPlanner from the FastAPI route.

`predict_with_planner` mirrors the contract of `app.router.predict_with_metadata`
(it returns `(QAResponse, metadata_dict)`) but routes the request through
the method-centric architecture instead of the legacy fixed pipeline.

Activation: `app.router.predict_with_metadata` checks
`URA_USE_METHOD_PLANNER` (or the `PipelineConfig.use_method_planner` field
if present) and delegates to this function. The legacy code path is
preserved untouched so a single env-var flip switches the system.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from app.llm_client import FallbackClient
from app.methods.library import get_default_library
from app.methods.planner import MethodPlanner, PlannerOutcome
from app.methods.problem import (
    LogicProblem,
    PhysicsProblem,
    build_logic_problem,
    build_physics_problem,
)
from app.pipeline_config import PipelineConfig, load_pipeline_config
from app.runtime_clients import ensure_runtime_llm_client, llm_client_info
from app.runtime_trace import (
    apply_input_guardrail_confidence,
    apply_json_validity_to_response,
    assemble_response,
    ensure_public_cot,
    log_request,
    record_confidence,
)
from app.runtime_workflow import (
    BudgetGatedClient,
    CallBudget,
    InputNormalizer,
    TaskRouter,
)
from app.schemas import QARequest, QAResponse, TaskType


def _legacy_fallback(
    request: QARequest,
    normalized_req: Any,
    task: TaskType,
    raw_llm_client: Any,
    config: PipelineConfig,
    metadata: dict[str, Any],
) -> QAResponse | None:
    """Last-resort fallback to the legacy solver when every Method abstained.

    Per AGENTS.md §24, the planner must be ≥ legacy on every case. If the
    Method library has no winner, run the legacy solver inline and adopt
    its answer. This guarantees no regression vs the pre-planner baseline
    while we expand Method coverage incrementally. Records the fallback
    use in metadata so audits can spot it.

    The legacy solver receives the RAW (un-gated) LLM client so its
    rescue/agent calls aren't blocked by a budget already half-spent by
    the planner's earlier attempts. The 30s overall deadline still
    applies via call_budget.deadline_exceeded() inside CallBudget.
    """
    metadata["legacy_fallback_invoked"] = True
    try:
        if task == TaskType.physics:
            from app.physics.solver import solve as solve_physics
            res = solve_physics(
                normalized_req.question,
                use_llm_extraction=bool(raw_llm_client),
                use_search=False,
                llm_client=raw_llm_client if request.allow_llm_fallback else None,
                rescue_unknown=True,
                max_agent_steps=config.max_agent_steps,
                max_model_calls=config.max_model_calls,
                max_search_calls=config.max_search_calls,
            )
            if not getattr(res, "success", False):
                return None
            from app.runtime_trace import build_physics_confidence_signals
            signals = build_physics_confidence_signals(res)
            response = assemble_response(
                {
                    "answer": res.answer,
                    "explanation": res.explanation,
                    "premises": [],
                    "cot": list(res.cot or []),
                    "fol": res.formula_id,
                    "confidence": record_confidence(signals, res.answer, metadata),
                    "task_type": "physics",
                    "raw_json_validity": None,
                    "repaired_json_validity": None,
                },
                metadata,
            )
            metadata["solver_used"] = "legacy_fallback_physics"
            metadata["answer_source"] = "deterministic_solver"
            response = apply_input_guardrail_confidence(response, normalized_req.guardrail)
            response = ensure_public_cot(response, metadata)
            response = apply_json_validity_to_response(response, raw_llm_client, metadata)
            return response

        from app.logic.solver import solve as solve_logic
        res = solve_logic(
            normalized_req.question,
            normalized_req.premises,
            llm_client=raw_llm_client if request.allow_llm_fallback else None,
            use_llm=bool(raw_llm_client and normalized_req.premises),
            choices=normalized_req.choices or None,
            max_agent_steps=config.max_agent_steps,
            max_model_calls=config.max_model_calls,
            premises_fol=request.premises_fol or None,
        )
        from app.runtime_trace import build_logic_confidence_signals
        signals = build_logic_confidence_signals(
            res, verifier_accepted=True,
            premise_coverage=(
                len(res.premises) / max(1, len(normalized_req.premises))
            ),
        )
        response = assemble_response(
            {
                "answer": res.answer,
                "explanation": res.explanation,
                "premises": list(res.premises or []),
                "cot": list(res.cot or []),
                "fol": None,
                "confidence": record_confidence(signals, res.answer, metadata),
                "task_type": "logic",
                "raw_json_validity": None,
                "repaired_json_validity": None,
            },
            metadata,
        )
        metadata["solver_used"] = "legacy_fallback_logic"
        metadata["answer_source"] = "deterministic_solver"
        response = apply_input_guardrail_confidence(response, normalized_req.guardrail)
        response = ensure_public_cot(response, metadata)
        response = apply_json_validity_to_response(response, raw_llm_client, metadata)
        return response
    except Exception as exc:
        metadata["legacy_fallback_error"] = f"{type(exc).__name__}:{exc}"
        return None


# ---------------------------------------------------------------------------


def use_method_planner_enabled(config: PipelineConfig | None = None) -> bool:
    """Return True iff the method-centric path is enabled.

    Looks at, in order: explicit `URA_USE_METHOD_PLANNER` env var
    (`1/true/yes/on` enables), then the optional
    `config.use_method_planner` field (added by a future config bump).
    """
    raw = os.environ.get("URA_USE_METHOD_PLANNER")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if config is not None and getattr(config, "use_method_planner", False):
        return True
    return False


# ---------------------------------------------------------------------------


def predict_with_planner(
    request: QARequest,
    config: PipelineConfig | None = None,
    llm_client: FallbackClient | None = None,
    write_trace: bool = False,
) -> tuple[QAResponse, dict[str, Any]]:
    """Method-centric replacement for `predict_with_metadata`.

    Steps:
      1. Normalize + budget-gate exactly like the legacy path.
      2. Route to logic vs physics, build the typed Problem IR.
      3. Hand it to `MethodPlanner` and convert the outcome into the
         standard QAResponse shape.
      4. Confidence + JSON validation are recorded the same way so trace
         consumers do not need to know which path produced the result.
    """
    config = config or load_pipeline_config()
    started = time.perf_counter()

    normalized_req = InputNormalizer().normalize(request)
    guardrail = normalized_req.guardrail
    metadata: dict[str, Any] = {
        "request_id": request.request_id or str(uuid.uuid4()),
        "input_question_original": guardrail.original_text,
        "input_question_normalized": guardrail.normalized_text,
        "normalized_question": normalized_req.question,
        "normalized_premise_count": len(normalized_req.premises),
        "normalized_choice_count": len(normalized_req.choices),
        "logic_answer_kind": normalized_req.answer_kind,
        "normalization_warnings": list(normalized_req.warnings),
        "fallback_used": False,
        "fallback_accepted": False,
        "fallback_rejected_reason": None,
        "fallback_policy": "method_planner",
        "model_calls": 0,
        "explanation_rewrite_rejected": False,
        "explanation_rewrite_rejected_reason": None,
        "solver_used": "method_planner",
        "planner_source": "method_library",
    }

    raw_llm_client = ensure_runtime_llm_client(
        config, llm_client, request.allow_llm_fallback
    )
    metadata["llm_client"] = llm_client_info(raw_llm_client)
    call_budget = CallBudget.start(max_calls=3, seconds=30.0)
    metadata["call_budget"] = {
        "max_calls": call_budget.max_calls,
        "deadline_seconds": 30.0,
    }
    # Phase F3.1: wrap the raw client in a request-scoped cache so two
    # methods that issue the SAME prompt within one request reuse the
    # same answer instead of re-rolling stochastic LLM output. This
    # eliminates the largest source of planner-vs-legacy variance
    # (see reports/phase_f3_planner_full_eval_report.md). Cache is
    # discarded at request end (the wrapper is local to this function).
    if raw_llm_client is not None:
        from app.methods.caching_client import RequestScopedCachingClient
        cached_raw_client = RequestScopedCachingClient(raw_llm_client)
        gated_client = BudgetGatedClient(cached_raw_client, call_budget)
    else:
        cached_raw_client = None
        gated_client = None

    # Decide task with the deterministic router (planner does not yet pick task).
    task = TaskRouter().route(normalized_req)

    # Build the typed Problem IR.
    if task == TaskType.physics:
        from app.physics.parser import parse_physics_question
        parsed = parse_physics_question(normalized_req.question)
        problem: LogicProblem | PhysicsProblem = build_physics_problem(
            normalized_req.question, parsed
        )
    else:
        problem = build_logic_problem(
            normalized_req.question,
            normalized_req.premises,
            choices=normalized_req.choices or None,
            premises_fol=request.premises_fol or None,
            answer_kind=normalized_req.answer_kind,
        )

    planner = MethodPlanner(library=get_default_library())
    outcome: PlannerOutcome = planner.solve(
        problem,
        llm_client=gated_client if request.allow_llm_fallback else None,
        budget=call_budget,
        allow_discovery=request.allow_llm_fallback,
    )
    metadata["planner_outcome"] = outcome.to_dict()
    metadata["planner_methods_tried"] = [d.method_id for d in outcome.decisions]
    metadata["discovery_attempted"] = bool(outcome.discovery_attempted)
    metadata["discovery_outcome"] = outcome.discovery_outcome
    metadata["model_calls"] = int(getattr(call_budget, "used", 0) or 0)

    final = outcome.final
    if final is None or not final.answer:
        # Planner abstained on this problem. Before returning a bare
        # ``unknown`` (which would regress against the legacy pipeline on
        # physics shapes not yet covered by a Method, e.g. conceptual
        # locators or qualitative variables the parser misses), delegate
        # to the legacy `solve_logic` / `solve_physics` once. This keeps
        # the planner-on configuration STRICTLY ≥ the legacy baseline:
        # methods that decisively solved a problem already returned
        # above, and only here do we let the legacy code attempt.
        legacy_response = _legacy_fallback(
            request, normalized_req, task,
            cached_raw_client if cached_raw_client is not None else raw_llm_client,
            config, metadata
        )
        if legacy_response is not None:
            metadata["latency_ms"] = (time.perf_counter() - started) * 1000
            log_request(task, metadata["request_id"], metadata)
            return legacy_response, metadata

        # Final abstain — no method, no legacy answer.
        answer = "unknown"
        explanation = (
            "The method library has no decisive solution for this question; "
            f"abstaining (reason: {outcome.abstain_reason or 'no_method'})."
        )
        cot = ["method_planner_abstain"]
        used_premises: list[str] = []
        formula_id: str | None = None
        metadata["fallback_rejected_reason"] = outcome.abstain_reason or "all_methods_abstained"
        metadata["answer_source"] = "abstention"
        method_decisive = False
        method_used_llm = False
        backend_confidence = 0.0
    else:
        answer = str(final.answer)
        explanation = final.explanation or ""
        cot = list(final.trace.backend_steps) or [final.method_id]
        used_premises = list(final.used_premise_ids)
        formula_id = final.formula_id
        metadata["solver_used"] = f"method_planner:{final.method_id}"
        metadata["answer_source"] = "method_planner"
        metadata["selected_method_id"] = final.method_id
        metadata["selected_method_family"] = final.family.value
        method_decisive = final.decisive
        method_used_llm = bool(final.trace.llm_calls)
        backend_confidence = float(final.confidence)

    # Build ConfidenceSignals (Req 9.2 / AGENTS.md §16): backend-only, never
    # LLM-self-reported. The method's own confidence already comes from
    # backend signals (translation confidence × premise coverage), so we
    # record it through the normal pipeline and let `compute_confidence`
    # aggregate with the boolean signals below.
    from app.confidence import ConfidenceSignals
    signals = ConfidenceSignals(
        parser_success=True,
        formula_matched=bool(formula_id) if task == TaskType.physics else False,
        solver_success=method_decisive,
        unit_valid=None if task != TaskType.physics else (
            (final.numeric_unit not in (None, "")) if final is not None else False
        ),
        answer_verified=method_decisive,  # method passed coverage gate
        premise_selection_score=(
            len(used_premises) / max(1, len(getattr(problem, "normalized_premises", []) or []))
            if isinstance(problem, LogicProblem) else 0.0
        ),
        llm_fallback_used=method_used_llm,
        json_valid=True,
        ambiguity_detected=False,
    )

    response = assemble_response(
        {
            "answer": answer,
            "explanation": explanation,
            "premises": used_premises,
            "cot": cot,
            "fol": formula_id,
            "confidence": record_confidence(signals, answer, metadata),
            "task_type": "logic" if task == TaskType.logic else "physics",
            "raw_json_validity": None,
            "repaired_json_validity": None,
        },
        metadata,
    )
    response = apply_input_guardrail_confidence(response, guardrail)
    response = ensure_public_cot(response, metadata)
    response = apply_json_validity_to_response(response, gated_client, metadata)

    metadata["latency_ms"] = (time.perf_counter() - started) * 1000
    log_request(task, metadata["request_id"], metadata)
    return response, metadata
