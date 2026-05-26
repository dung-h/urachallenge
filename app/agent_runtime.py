from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_kernel import run_agent_loop
from app.agent_tools import PhysicsAgentContext, execute_tool, inspect_problem
from app.physics.parser import ParsedPhysicsProblem
from app.physics.problem_frame import infer_problem_frame, search_unknown_explanation


_ALLOWED_TOOLS = {
    "inspect_problem",
    "retrieve_method_evidence",
    "extract_equation_proposals",
    "verify_and_compute_method",
    "llm_rescue",
    "finish_unknown",
}


@dataclass(frozen=True)
class PhysicsAgentOutcome:
    success: bool
    answer: str
    explanation: str
    formula_id: str | None
    session_id: str = ""
    variables: dict[str, float] = field(default_factory=dict)
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    search_trace: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    agent_events: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    error: str | None = None


def _default_next_tool(context: PhysicsAgentContext) -> str:
    if context.frame is None:
        return "inspect_problem"
    if not context.evidence:
        return "retrieve_method_evidence"
    if not context.proposals:
        return "extract_equation_proposals"
    if context.verified is None:
        return "verify_and_compute_method"
    if context.allow_llm_rescue:
        return "llm_rescue"
    return "finish_unknown"


def _planner_payload(
    context: PhysicsAgentContext,
    base_solution: dict[str, Any] | None,
    tried_tools: list[str],
    last_result: dict[str, Any] | None,
    agent_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question": context.question,
        "parsed": {
            "target_quantity": context.parsed.target_quantity,
            "formula_id": context.parsed.formula_id,
            "variables": dict(context.parsed.variables),
            "ambiguity": list(getattr(context.parsed, "ambiguity", []) or []),
        },
        "problem_frame": context.frame.__dict__ if context.frame is not None else None,
        "method_search_objective": context.objective.__dict__ if context.objective is not None else None,
        "base_solution": base_solution,
        "evidence_count": len(context.evidence),
        "proposal_count": len(context.proposals),
        "tried_tools": list(tried_tools),
        "available_tools": sorted(_ALLOWED_TOOLS),
        "last_result": last_result,
        "agent_trace": list(agent_trace[-3:]),
        "allow_llm_rescue": context.allow_llm_rescue,
    }


def _to_outcome(
    *,
    success: bool,
    answer: str,
    explanation: str,
    formula_id: str | None,
    session_id: str,
    variables: dict[str, float] | None,
    cot: list[str] | None,
    confidence: float,
    search_trace: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
    agent_events: list[dict[str, Any]],
    model_calls: int,
    error: str | None,
) -> PhysicsAgentOutcome:
    return PhysicsAgentOutcome(
        success=success,
        answer=answer,
        explanation=explanation,
        formula_id=formula_id,
        session_id=session_id,
        variables=dict(variables or {}),
        cot=list(cot or []),
        confidence=max(0.0, min(1.0, confidence)),
        search_trace=list(search_trace),
        agent_trace=list(agent_trace),
        agent_events=list(agent_events),
        model_calls=model_calls,
        error=error,
    )


def run_physics_agent(
    question: str,
    parsed: ParsedPhysicsProblem,
    *,
    llm_client: Any | None,
    base_solution: dict[str, Any] | None = None,
    allow_llm_rescue: bool = True,
    max_steps: int = 4,
    max_model_calls: int | None = None,
    max_search_calls: int = 3,
) -> PhysicsAgentOutcome:
    context = PhysicsAgentContext(
        question=question,
        parsed=parsed,
        llm_client=llm_client,
        allow_llm_rescue=allow_llm_rescue,
        max_search_calls=max_search_calls,
    )
    agent_outcome = run_agent_loop(
        llm_client=llm_client,
        allowed_tools=_ALLOWED_TOOLS,
        default_tool=_default_next_tool,
        inspect_tool=inspect_problem,
        execute_tool=execute_tool,
        build_payload=_planner_payload,
        context=context,
        base_solution=base_solution,
        planner_method_name="plan_physics_action",
        max_steps=max_steps,
        max_model_calls=max_model_calls,
    )
    if agent_outcome.success:
        return _to_outcome(
            success=True,
            answer=agent_outcome.answer,
            explanation=agent_outcome.explanation,
            formula_id=agent_outcome.formula_id,
            session_id=agent_outcome.session_id,
            variables=agent_outcome.variables,
            cot=agent_outcome.cot,
            confidence=agent_outcome.confidence,
            search_trace=list(agent_outcome.trace),
            agent_trace=list(agent_outcome.trace),
            agent_events=list(agent_outcome.events),
            model_calls=agent_outcome.model_calls,
            error=None,
        )

    frame = context.frame or infer_problem_frame(parsed, question)
    explanation = search_unknown_explanation(frame, agent_outcome.error or "physics_agent_no_verified_proposal")
    return _to_outcome(
        success=False,
        answer=agent_outcome.answer,
        explanation=explanation,
        formula_id=None,
        session_id=agent_outcome.session_id,
        variables={},
        cot=list(agent_outcome.cot),
        confidence=agent_outcome.confidence,
        search_trace=list(agent_outcome.trace),
        agent_trace=list(agent_outcome.trace),
        agent_events=list(agent_outcome.events),
        model_calls=agent_outcome.model_calls,
        error=agent_outcome.error,
    )
