from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_kernel import run_agent_loop
from app.logic.agent_tools import LogicAgentContext, execute_tool, inspect_problem
from app.logic.premise_selector import Premise


_ALLOWED_TOOLS = {
    "inspect_problem",
    "detect_contradiction",
    "derive_deterministic_solution",
    "llm_rescue",
    "finish_unknown",
}


@dataclass(frozen=True)
class LogicAgentOutcome:
    success: bool
    answer: str
    explanation: str
    premises: list[str]
    session_id: str = ""
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    agent_events: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    error: str | None = None
    solution: dict[str, Any] | None = None


def _default_next_tool(context: LogicAgentContext) -> str:
    if not context.selected_premises:
        return "inspect_problem"
    if not context.contradiction_checked:
        return "detect_contradiction"
    if not context.deterministic_checked:
        return "derive_deterministic_solution"
    if context.allow_llm_rescue and not context.llm_rescue_attempted:
        return "llm_rescue"
    return "finish_unknown"


def _planner_payload(
    context: LogicAgentContext,
    base_solution: dict[str, Any] | None,
    tried_tools: list[str],
    last_result: dict[str, Any] | None,
    agent_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question": context.question,
        "premises": [{"id": premise.id, "text": premise.text} for premise in context.premises],
        "choices": list(context.choices),
        "selected_premises": [premise.id for premise in context.selected_premises],
        "contradiction_checked": context.contradiction_checked,
        "contradiction_found": context.contradiction_found,
        "base_solution": base_solution,
        "tried_tools": list(tried_tools),
        "available_tools": sorted(_ALLOWED_TOOLS),
        "last_result": last_result,
        "agent_trace": list(agent_trace[-3:]),
        "allow_llm_rescue": context.allow_llm_rescue,
    }


def run_logic_agent(
    question: str,
    premises: list[Premise],
    *,
    llm_client: Any | None,
    base_solution: Any | None = None,
    choices: list[str] | None = None,
    allow_llm_rescue: bool = True,
    max_steps: int = 4,
) -> LogicAgentOutcome:
    context = LogicAgentContext(
        question=question,
        premises=premises,
        choices=list(choices or []),
        llm_client=llm_client,
        allow_llm_rescue=allow_llm_rescue,
        base_solution=base_solution,
    )
    agent_outcome = run_agent_loop(
        llm_client=llm_client,
        allowed_tools=_ALLOWED_TOOLS,
        default_tool=_default_next_tool,
        inspect_tool=inspect_problem,
        execute_tool=execute_tool,
        build_payload=_planner_payload,
        context=context,
        base_solution=None if base_solution is None else {
            "success": bool(getattr(base_solution, "answer", "unknown") != "unknown"),
            "answer": getattr(base_solution, "answer", "unknown"),
            "explanation": getattr(base_solution, "explanation", ""),
            "premises": list(getattr(base_solution, "premises", []) or []),
            "cot": list(getattr(base_solution, "cot", []) or []),
            "confidence": float(getattr(base_solution, "confidence", 0.0) or 0.0),
        },
        planner_method_name="plan_logic_action",
        max_steps=max_steps,
    )
    return LogicAgentOutcome(
        success=agent_outcome.success,
        answer=agent_outcome.answer,
        explanation=agent_outcome.explanation,
        premises=list(agent_outcome.solution.get("premises") if agent_outcome.solution and isinstance(agent_outcome.solution.get("premises"), list) else []),
        session_id=agent_outcome.session_id,
        cot=list(agent_outcome.cot),
        confidence=agent_outcome.confidence,
        agent_trace=list(agent_outcome.trace),
        agent_events=list(agent_outcome.events),
        model_calls=agent_outcome.model_calls,
        error=agent_outcome.error,
        solution=agent_outcome.solution,
    )
