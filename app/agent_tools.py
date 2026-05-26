from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.physics.method_search import (
    MethodEquationProposal,
    MethodEvidenceSnippet,
    MethodSearchObjective,
    VerifiedMethod,
    build_objective as build_method_objective,
    extract_equation_proposals,
    retrieve_method_evidence,
    verify_and_compute_method,
)
from app.physics.parser import ParsedPhysicsProblem
from app.physics.problem_frame import ProblemFrame, infer_problem_frame, search_unknown_explanation
from app.physics.unit_converter import format_best_unit


@dataclass
class PhysicsAgentContext:
    question: str
    parsed: ParsedPhysicsProblem
    llm_client: Any | None = None
    allow_llm_rescue: bool = True
    max_search_calls: int = 3
    frame: ProblemFrame | None = None
    objective: MethodSearchObjective | None = None
    evidence: list[MethodEvidenceSnippet] = field(default_factory=list)
    proposals: list[MethodEquationProposal] = field(default_factory=list)
    verified: VerifiedMethod | None = None


@dataclass(frozen=True)
class PhysicsToolResult:
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


def inspect_problem(context: PhysicsAgentContext) -> PhysicsToolResult:
    frame = infer_problem_frame(context.parsed, context.question)
    objective = build_method_objective(context.parsed, context.question)
    context.frame = frame
    context.objective = objective
    return PhysicsToolResult(
        tool="inspect_problem",
        ok=True,
        summary=f"inferred {frame.method_family or 'unknown'} frame",
        updates={
            "problem_frame": frame.__dict__,
            "method_search_objective": objective.__dict__,
        },
    )


def retrieve_evidence(context: PhysicsAgentContext) -> PhysicsToolResult:
    if context.objective is None:
        inspect_problem(context)
    objective = context.objective
    assert objective is not None
    snippets = retrieve_method_evidence(objective, max_search_calls=context.max_search_calls)
    context.evidence = snippets
    return PhysicsToolResult(
        tool="retrieve_method_evidence",
        ok=bool(snippets),
        summary=f"retrieved {len(snippets)} evidence snippets",
        updates={
            "retrieved_evidence": [
                {
                    "source": snippet.source,
                    "title": snippet.title,
                    "url": snippet.url,
                }
                for snippet in snippets[:8]
            ]
        },
        error=None if snippets else "no_evidence_retrieved",
    )


def extract_candidates(context: PhysicsAgentContext) -> PhysicsToolResult:
    if context.objective is None:
        inspect_problem(context)
    if not context.evidence:
        retrieve_evidence(context)
    objective = context.objective
    evidence = context.evidence
    assert objective is not None
    proposals = extract_equation_proposals(objective, evidence)
    context.proposals = proposals
    return PhysicsToolResult(
        tool="extract_equation_proposals",
        ok=bool(proposals),
        summary=f"extracted {len(proposals)} equation proposals",
        updates={
            "method_proposals": [
                {
                    "method_id": proposal.method_id,
                    "method_family": proposal.method_family,
                    "expression": proposal.expression,
                    "target_unit": proposal.target_unit,
                    "assumptions": list(proposal.assumptions),
                    "blocked_formula_families": list(proposal.blocked_formula_families),
                    "confidence": proposal.confidence,
                    "evidence_source": proposal.evidence.source,
                    "evidence_title": proposal.evidence.title,
                }
                for proposal in proposals[:8]
            ]
        },
        error=None if proposals else "no_equation_proposals",
    )


def verify_candidates(context: PhysicsAgentContext) -> PhysicsToolResult:
    if context.objective is None:
        inspect_problem(context)
    if not context.proposals:
        extract_candidates(context)
    verified: VerifiedMethod | None = None
    rejected: list[str] = []
    for proposal in sorted(context.proposals, key=lambda item: item.confidence, reverse=True):
        candidate = verify_and_compute_method(context.parsed, context.question, proposal)
        if candidate is not None:
            verified = candidate
            context.verified = candidate
            break
        rejected.append(f"{proposal.method_family}:{proposal.method_id}")
    if verified is None:
        frame = context.frame or infer_problem_frame(context.parsed, context.question)
        return PhysicsToolResult(
            tool="verify_and_compute_method",
            ok=False,
            summary="no proposal verified",
            updates={"rejected_proposals": rejected},
            error=search_unknown_explanation(frame, "no_verified_method_proposal", details=rejected),
        )

    answer = format_best_unit(verified.value, verified.proposal.target_unit)
    explanation = "Search-backed method reasoning verified a retrieved equation against the question: "
    explanation += f"used {verified.proposal.expression} with variables {verified.variables}. Python computed {answer}."
    return PhysicsToolResult(
        tool="verify_and_compute_method",
        ok=True,
        summary=f"verified {verified.proposal.method_id}",
        updates={
            "accepted_method_evidence": {
                "method_id": verified.proposal.method_id,
                "method_family": verified.proposal.method_family,
                "expression": verified.proposal.expression,
                "variables": verified.variables,
                "verification_notes": verified.verification_notes,
                "evidence_source": verified.proposal.evidence.source,
                "evidence_title": verified.proposal.evidence.title,
            }
        },
        solution={
            "success": True,
            "answer": answer,
            "explanation": explanation,
            "formula_id": verified.proposal.method_id,
            "variables": dict(verified.variables),
            "cot": [
                f"Parsed target: {context.parsed.target_quantity or 'unknown'}",
                f"Retrieved method: {verified.proposal.method_id}",
                f"Verified assumptions: {', '.join(verified.proposal.assumptions)}",
                f"Computed with Python: {answer}",
            ],
            "confidence": min(0.9, max(0.75, verified.proposal.confidence)),
        },
    )


def llm_rescue(context: PhysicsAgentContext) -> PhysicsToolResult:
    if not context.allow_llm_rescue:
        frame = context.frame or infer_problem_frame(context.parsed, context.question)
        return PhysicsToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM rescue disabled",
            error=search_unknown_explanation(frame, "llm_rescue_disabled"),
        )
    suggest_physics = getattr(context.llm_client, "suggest_physics", None) if context.llm_client is not None else None
    if not callable(suggest_physics):
        frame = context.frame or infer_problem_frame(context.parsed, context.question)
        return PhysicsToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM rescue unavailable",
            error=search_unknown_explanation(frame, "physics_fallback_no_proposal"),
        )
    try:
        suggestion = suggest_physics(context.question)
    except Exception as exc:
        frame = context.frame or infer_problem_frame(context.parsed, context.question)
        return PhysicsToolResult(
            tool="llm_rescue",
            ok=False,
            summary="LLM proposal failed",
            error=search_unknown_explanation(frame, f"llm_rescue_error:{type(exc).__name__}"),
        )

    from app.physics.solver import solve_from_llm_suggestion

    result = solve_from_llm_suggestion(context.question, suggestion or {})
    if result.success:
        return PhysicsToolResult(
            tool="llm_rescue",
            ok=True,
            summary="LLM proposal verified by backend",
            solution={
                "success": True,
                "answer": result.answer,
                "explanation": result.explanation,
                "formula_id": result.formula_id,
                "variables": dict(result.variables),
                "cot": list(result.cot),
                "confidence": result.confidence,
            },
            model_calls=1,
        )
    return PhysicsToolResult(
        tool="llm_rescue",
        ok=False,
        summary="LLM proposal rejected",
        error=result.error or "physics_fallback_no_proposal",
        model_calls=1,
        updates={"llm_suggestion": suggestion or {}},
    )


def finish_unknown(context: PhysicsAgentContext, reason: str, details: list[str] | None = None) -> PhysicsToolResult:
    frame = context.frame or infer_problem_frame(context.parsed, context.question)
    return PhysicsToolResult(
        tool="finish_unknown",
        ok=False,
        summary=reason,
        error=search_unknown_explanation(frame, reason, details=details),
    )


def default_tool_sequence(context: PhysicsAgentContext) -> list[str]:
    sequence = ["inspect_problem", "retrieve_method_evidence", "extract_equation_proposals", "verify_and_compute_method"]
    if context.allow_llm_rescue:
        sequence.append("llm_rescue")
    sequence.append("finish_unknown")
    return sequence


def execute_tool(tool_name: str, context: PhysicsAgentContext, args: dict[str, Any] | None = None) -> PhysicsToolResult:
    tool = (tool_name or "").strip()
    if tool == "inspect_problem":
        return inspect_problem(context)
    if tool == "retrieve_method_evidence":
        return retrieve_evidence(context)
    if tool == "extract_equation_proposals":
        return extract_candidates(context)
    if tool == "verify_and_compute_method":
        return verify_candidates(context)
    if tool == "llm_rescue":
        return llm_rescue(context)
    if tool == "finish_unknown":
        return finish_unknown(context, str((args or {}).get("reason") or "no_verified_method_proposal"), list((args or {}).get("details") or []))
    frame = context.frame or infer_problem_frame(context.parsed, context.question)
    return PhysicsToolResult(
        tool=tool or "unknown",
        ok=False,
        summary="disallowed tool",
        error=search_unknown_explanation(frame, "disallowed_tool", details=[tool] if tool else None),
    )
