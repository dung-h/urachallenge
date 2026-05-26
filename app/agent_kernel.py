from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    step: int
    session_id: str = ""
    tool: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "step": self.step,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.tool is not None:
            payload["tool"] = self.tool
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class AgentPolicy:
    max_steps: int = 4
    retry_planner_errors: int = 1
    retry_invalid_action: int = 1
    stop_on_unknown: bool = False


@dataclass(frozen=True)
class AgentOutcome:
    success: bool
    answer: str
    explanation: str
    formula_id: str | None
    session_id: str = ""
    variables: dict[str, float] = field(default_factory=dict)
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    error: str | None = None
    solution: dict[str, Any] | None = None


def normalize_action(action: Any, allowed_tools: set[str], default_tool: str) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {
            "tool": default_tool,
            "args": {},
            "reason": "fallback: invalid planner output",
            "stop": False,
            "confidence": 0.0,
        }
    tool = str(action.get("tool") or action.get("action") or action.get("name") or "").strip()
    if tool not in allowed_tools:
        tool = default_tool
    args = action.get("args")
    if not isinstance(args, dict):
        args = {}
    reason = str(action.get("reason") or action.get("reason_short") or "planner selected tool").strip()
    stop = bool(action.get("stop", False))
    confidence = action.get("confidence", 0.0)
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_value = 0.0
    return {
        "tool": tool,
        "args": args,
        "reason": reason,
        "stop": stop,
        "confidence": confidence_value,
    }


def run_agent_loop(
    *,
    llm_client: Any | None,
    allowed_tools: set[str],
    default_tool: Callable[[Any], str],
    inspect_tool: Callable[[Any], Any],
    execute_tool: Callable[[str, Any, dict[str, Any] | None], Any],
    build_payload: Callable[[Any, Any | None, list[str], dict[str, Any] | None, list[dict[str, Any]]], dict[str, Any]],
    context: Any,
    base_solution: dict[str, Any] | None = None,
    planner_method_name: str = "plan_action",
    max_steps: int | None = None,
    max_model_calls: int | None = None,
    policy: AgentPolicy | None = None,
) -> AgentOutcome:
    import os
    policy = policy or AgentPolicy(max_steps=max_steps or 4)
    env_max_steps = int(os.environ.get("URA_MAX_AGENT_STEPS") or "4")
    if max_steps is None:
        steps = min(policy.max_steps, env_max_steps)
    else:
        steps = min(policy.max_steps, max_steps)
    max_model_calls = max_model_calls if max_model_calls is not None else int(os.environ.get("URA_MAX_MODEL_CALLS") or "5")
    session_id = uuid.uuid4().hex
    trace: list[dict[str, Any]] = []
    events: list[AgentEvent] = []
    model_calls = 0
    tried_tools: list[str] = []
    remaining_planner_retries = policy.retry_planner_errors
    remaining_invalid_action_retries = policy.retry_invalid_action

    first_result = inspect_tool(context)
    events.append(AgentEvent(kind="tool", step=0, session_id=session_id, tool=getattr(first_result, "tool", "inspect_problem"), detail=first_result.to_trace()))
    trace.append(
        {
                "step": 0,
                "session_id": session_id,
                "action": {
                "tool": getattr(first_result, "tool", "inspect_problem"),
                "args": {},
                "reason": "initialize agent context",
                "stop": False,
                "confidence": 1.0,
            },
            "tool_result": first_result.to_trace(),
        }
    )
    tried_tools.append(getattr(first_result, "tool", "inspect_problem"))
    last_result: dict[str, Any] | None = first_result.to_trace()
    planner = getattr(llm_client, planner_method_name, None) if llm_client is not None else None

    step = 0
    while step < steps:
        fallback_tool = default_tool(context)
        if callable(planner):
            if model_calls >= max_model_calls:
                events.append(AgentEvent(kind="budget_exceeded", step=step + 1, session_id=session_id, detail={"reason": "max_model_calls", "model_calls": model_calls}))
                break
            try:
                planned_action = planner(build_payload(context, base_solution, tried_tools, last_result, trace))
                model_calls += 1
                planned_tool = None
                if isinstance(planned_action, dict):
                    planned_tool = str(planned_action.get("tool") or planned_action.get("action") or planned_action.get("name") or "").strip() or None
                events.append(AgentEvent(kind="planner", step=step + 1, session_id=session_id, detail={"status": "ok", "tool": planned_tool}))
            except Exception as exc:
                planned_action = None
                events.append(AgentEvent(kind="planner_error", step=step + 1, session_id=session_id, detail={"error": f"{type(exc).__name__}: {exc}"}))
                if remaining_planner_retries > 0:
                    remaining_planner_retries -= 1
                    events.append(
                        AgentEvent(
                            kind="retry",
                            step=step + 1,
                            session_id=session_id,
                            tool=fallback_tool,
                            detail={"reason": "planner_error", "fallback_tool": fallback_tool},
                        )
                    )
                    continue
        else:
            planned_action = None

        action = normalize_action(planned_action, allowed_tools, fallback_tool)
        if (
            action["tool"] == fallback_tool
            and isinstance(planned_action, dict)
            and str(planned_action.get("tool") or planned_action.get("action") or planned_action.get("name") or "").strip() not in allowed_tools
            and remaining_invalid_action_retries > 0
        ):
            remaining_invalid_action_retries -= 1
            events.append(
                AgentEvent(
                    kind="retry",
                    step=step + 1,
                    session_id=session_id,
                    tool=action["tool"],
                    detail={"reason": "invalid_action", "fallback_tool": fallback_tool},
                )
            )
            continue
        tool_name = action["tool"]
        tool_result = execute_tool(tool_name, context, action.get("args"))
        model_calls += int(getattr(tool_result, "model_calls", 0) or 0)
        tried_tools.append(tool_name)
        events.append(
            AgentEvent(
                kind="tool",
                step=step + 1,
                session_id=session_id,
                tool=tool_name,
                detail=tool_result.to_trace(),
            )
        )
        trace_entry = {
            "step": step + 1,
            "action": action,
            "tool_result": tool_result.to_trace(),
        }
        trace.append(trace_entry)
        last_result = tool_result.to_trace()

        if getattr(tool_result, "solution", None) is not None and getattr(tool_result, "ok", False):
            solution = tool_result.solution
            return AgentOutcome(
                success=True,
                answer=str(solution.get("answer") or "unknown"),
                explanation=str(solution.get("explanation") or ""),
                formula_id=solution.get("formula_id"),
                session_id=session_id,
                variables=solution.get("variables") if isinstance(solution.get("variables"), dict) else {},
                cot=solution.get("cot") if isinstance(solution.get("cot"), list) else [],
                confidence=float(solution.get("confidence") or 0.0),
                trace=trace,
                events=[event.to_dict() for event in events],
                model_calls=model_calls,
                error=None,
                solution=solution if isinstance(solution, dict) else None,
            )

        if tool_name == "finish_unknown" or action.get("stop"):
            events.append(AgentEvent(kind="stop", step=step + 1, session_id=session_id, tool=tool_name, detail={"reason": "finish_unknown" if tool_name == "finish_unknown" else "planner_stop"}))
            break

        step += 1

    reason = None
    details: list[str] = []
    if events and events[-1].kind == "budget_exceeded":
        reason = "budget_exceeded:max_model_calls"
    elif isinstance(last_result, dict):
        reason = str(last_result.get("error") or last_result.get("summary") or "")
        if isinstance(last_result.get("updates"), dict):
            details = [str(item) for item in last_result["updates"].get("rejected_proposals") or []]
    if not reason:
        reason = "agent_no_verified_proposal"
    fallback_error = None
    if isinstance(last_result, dict) and isinstance(last_result.get("error"), str):
        fallback_error = last_result["error"]
    return AgentOutcome(
        success=False,
        answer="unknown",
        explanation=fallback_error or reason,
        formula_id=None,
        variables={},
        cot=[f"Agent stopped after {len(trace)} steps", f"Final reason: {reason}"] + ([f"Rejected: {', '.join(details)}"] if details else []),
        confidence=0.2,
        trace=trace,
        events=[event.to_dict() for event in events],
        model_calls=model_calls,
        error=fallback_error or reason,
        session_id=session_id,
        solution=None,
    )
