from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.logic.policy_reasoner import solve_policy
from app.logic.premise_selector import Premise, normalize_premises, select_premises
from app.logic.templates import logic_explanation


@dataclass
class LogicAgentContext:
    question: str
    premises: list[Premise]
    choices: list[str] = field(default_factory=list)
    llm_client: Any | None = None
    allow_llm_rescue: bool = True
    base_solution: Any | None = None
    selected_premises: list[Premise] = field(default_factory=list)
    contradiction_checked: bool = False
    contradiction_found: bool = False
    deterministic_checked: bool = False
    llm_rescue_attempted: bool = False


@dataclass(frozen=True)
class LogicToolResult:
    tool: str
    ok: bool
    summary: str
    updates: dict[str, Any] = field(default_factory=dict)
    solution: dict[str, Any] | None = None
    model_calls: int = 0
    error: str | None = None

    def to_trace(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
        }
        if self.updates:
            payload["updates"] = dict(self.updates)
        if self.solution is not None:
            payload["solution"] = dict(self.solution)
        if self.model_calls:
            payload["model_calls"] = self.model_calls
        if self.error:
            payload["error"] = self.error
        return payload


def inspect_problem(context: LogicAgentContext) -> LogicToolResult:
    normalized = normalize_premises([premise.text for premise in context.premises])
    selected = select_premises(context.question, normalized)
    context.selected_premises = list(selected)
    return LogicToolResult(
        tool="inspect_problem",
        ok=True,
        summary=f"inspected {len(normalized)} logic premises",
        updates={
            "normalized_premise_ids": [premise.id for premise in normalized],
            "selected_premise_ids": [premise.id for premise in selected],
            "selected_premise_texts": [premise.text for premise in selected],
            "choice_count": len(context.choices),
        },
    )


def detect_contradiction(context: LogicAgentContext) -> LogicToolResult:
    context.contradiction_checked = True
    decision = solve_policy(context.question, context.premises)
    selected = context.selected_premises or context.premises
    if decision and ("contradict" in decision.rule or "conflict" in decision.rule):
        context.contradiction_found = True
        cited = [premise.id for premise in decision.premises]
        return LogicToolResult(
            tool="detect_contradiction",
            ok=True,
            summary=decision.rule,
            updates={
                "contradiction_rule": decision.rule,
                "contradiction_premise_ids": cited,
                "contradiction_cot": list(decision.cot),
            },
            solution={
                "success": True,
                "answer": "unknown",
                "explanation": logic_explanation("unknown", decision.premises, decision.rule),
                "premises": cited,
                "cot": list(decision.cot) or ["Detected contradictory or conflicting premises"],
                "confidence": min(0.78, max(0.6, decision.confidence)),
            },
        )
    return LogicToolResult(
        tool="detect_contradiction",
        ok=False,
        summary="no contradiction detected",
        updates={"selected_premise_ids": [premise.id for premise in selected]},
    )


def _base_solution_payload(context: LogicAgentContext) -> dict[str, Any]:
    base = context.base_solution
    if base is None:
        return {
            "success": False,
            "answer": "unknown",
            "explanation": "No deterministic logic base solution is available.",
            "premises": [],
            "cot": ["No deterministic base solution available"],
            "confidence": 0.0,
        }
    return {
        "success": bool(getattr(base, "answer", "unknown") != "unknown"),
        "answer": getattr(base, "answer", "unknown"),
        "explanation": getattr(base, "explanation", ""),
        "premises": list(getattr(base, "premises", []) or []),
        "cot": list(getattr(base, "cot", []) or []),
        "confidence": float(getattr(base, "confidence", 0.0) or 0.0),
    }


def derive_deterministic_solution(context: LogicAgentContext) -> LogicToolResult:
    context.deterministic_checked = True
    payload = _base_solution_payload(context)
    selected = [premise for premise in context.premises if premise.id in set(payload.get("premises") or [])]
    if payload["success"]:
        return LogicToolResult(
            tool="derive_deterministic_solution",
            ok=True,
            summary=f"deterministic solver returned {payload['answer']}",
            updates={
                "deterministic_solution": payload,
                "selected_premise_ids": [premise.id for premise in selected],
            },
            solution={
                "success": True,
                "answer": payload["answer"],
                "explanation": payload["explanation"],
                "premises": payload["premises"],
                "cot": payload["cot"],
                "confidence": payload["confidence"],
            },
        )
    return LogicToolResult(
        tool="derive_deterministic_solution",
        ok=False,
        summary="deterministic solver still returns unknown",
        updates={
            "deterministic_solution": payload,
            "selected_premise_ids": [premise.id for premise in selected],
        },
        error=payload["explanation"],
    )


def llm_rescue(context: LogicAgentContext) -> LogicToolResult:
    context.llm_rescue_attempted = True
    if not context.allow_llm_rescue:
        selected = context.selected_premises or context.premises
        return LogicToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM rescue disabled",
            error=logic_explanation("unknown", selected, "LLM rescue disabled"),
        )
    suggest_logic = getattr(context.llm_client, "suggest_logic", None) if context.llm_client is not None else None
    if not callable(suggest_logic):
        selected = context.selected_premises or context.premises
        return LogicToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM rescue unavailable",
            error=logic_explanation("unknown", selected, "logic_fallback_no_proposal"),
        )
    try:
        suggestion = suggest_logic(context.question, [f"{p.id}: {p.text}" for p in context.premises])
    except Exception as exc:
        selected = context.selected_premises or context.premises
        return LogicToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM proposal failed",
            error=logic_explanation("unknown", selected, f"llm_rescue_error:{type(exc).__name__}"),
        )

    if not isinstance(suggestion, dict):
        selected = context.selected_premises or context.premises
        return LogicToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM proposal malformed",
            error=logic_explanation("unknown", selected, "logic_fallback_no_proposal"),
        )

    candidate = str(suggestion.get("answer") or "unknown").strip().lower()
    if candidate not in {"yes", "no", "unknown"} and not candidate in {"a", "b", "c", "d", "e"}:
        candidate = "unknown"
    ids_raw = suggestion.get("used_premise_ids") or suggestion.get("premises") or []
    ids = {str(pid).upper() for pid in ids_raw if isinstance(pid, str) and re.fullmatch(r"P\d+", pid.upper())}
    normalized_ids = {p.id for p in context.premises}
    hallucinated = sorted(pid for pid in ids if pid not in normalized_ids)
    baseline_unknown_override = (getattr(context.base_solution, "answer", "unknown") == "unknown" and candidate != "unknown")
    if not ids or hallucinated or baseline_unknown_override:
        selected = [p for p in context.premises if p.id in ids] or context.selected_premises or context.premises
        return LogicToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM proposal rejected",
            updates={"llm_suggestion": suggestion, "hallucinated_premises": hallucinated},
            error=logic_explanation("unknown", selected, "logic_fallback_validation_failed"),
        )

    selected = [p for p in context.premises if p.id in ids]
    explanation = logic_explanation(candidate, selected, str(suggestion.get("reason_short") or "validated LLM fallback"))
    return LogicToolResult(
        tool="llm_rescue",
        ok=True,
        summary="LLM proposal validated by backend",
        updates={"llm_suggestion": suggestion, "validated_premises": [p.id for p in selected]},
        solution={
            "success": True,
            "answer": candidate,
            "explanation": explanation,
            "premises": [p.id for p in selected],
            "cot": [
                "Rule baseline confidence below threshold",
                "LLM reasoner suggested answer",
                "Backend validated premise IDs and normalized answer",
            ],
            "confidence": 0.72 if candidate != "unknown" else 0.68,
        },
        model_calls=1,
    )


def finish_unknown(context: LogicAgentContext, reason: str, details: list[str] | None = None) -> LogicToolResult:
    selected = context.selected_premises or context.premises
    return LogicToolResult(
        tool="finish_unknown",
        ok=False,
        summary=reason,
        error=logic_explanation("unknown", selected, reason),
        updates={"details": list(details or [])},
    )


def default_tool_sequence(context: LogicAgentContext) -> list[str]:
    if not context.selected_premises:
        return ["inspect_problem", "detect_contradiction", "derive_deterministic_solution", "llm_rescue", "finish_unknown"]
    if not context.contradiction_checked:
        return ["detect_contradiction", "derive_deterministic_solution", "llm_rescue", "finish_unknown"]
    if not context.deterministic_checked:
        return ["derive_deterministic_solution", "llm_rescue", "finish_unknown"]
    if context.allow_llm_rescue and not context.llm_rescue_attempted:
        return ["llm_rescue", "finish_unknown"]
    return ["finish_unknown"]


def execute_tool(tool_name: str, context: LogicAgentContext, args: dict[str, Any] | None = None) -> LogicToolResult:
    tool = (tool_name or "").strip()
    if tool == "inspect_problem":
        return inspect_problem(context)
    if tool == "detect_contradiction":
        return detect_contradiction(context)
    if tool == "derive_deterministic_solution":
        return derive_deterministic_solution(context)
    if tool == "llm_rescue":
        return llm_rescue(context)
    if tool == "finish_unknown":
        reason = str((args or {}).get("reason") or "logic_agent_no_verified_proposal")
        details = (args or {}).get("details")
        if not isinstance(details, list):
            details = []
        return finish_unknown(context, reason, [str(item) for item in details])
    selected = context.selected_premises or context.premises
    return LogicToolResult(
        tool=tool or "unknown",
        ok=False,
        summary="disallowed tool",
        error=logic_explanation("unknown", selected, "disallowed_tool"),
    )
