"""Method wrapper around the LLM-FOL + Z3 logic pipeline.

This is the PRIMARY logic method. It hands the problem to
``app.logic.fol_z3_pipeline.solve_fol_z3`` (LLM translates → Z3 verifies)
and packages the outcome as a ``MethodResult`` so the planner sees it
through the same interface as every other method.
"""

from __future__ import annotations

import time
from typing import Any

from app.methods.types import (
    Method,
    MethodApplicability,
    MethodFamily,
    MethodResult,
    MethodSource,
    MethodTrace,
)
from app.methods.problem import LogicProblem


class LogicFolZ3Method:
    """LLM-FOL translator + Z3 prover with a self-refinement loop.

    Applicability:
      * Any logic problem with at least one premise.
      * Higher score when conditional / quantifier / negation markers are
        present — those are the structures Z3 can reason over precisely.

    Abstains when the LLM is not available or when Z3 returns
    ``undetermined``. The planner falls through to the next method.
    """

    method_id: str = "logic.fol_z3"
    family: MethodFamily = MethodFamily.LOGIC_SYMBOLIC
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: LogicProblem) -> MethodApplicability:
        if not isinstance(problem, LogicProblem):
            return MethodApplicability(0.0, "not_logic_problem")
        if problem.premise_count == 0:
            return MethodApplicability(0.0, "no_premises")
        score = 0.5
        why_parts = ["has_premises"]
        if problem.has_conditional_marker:
            score += 0.2
            why_parts.append("conditional")
        if problem.has_quantifier_marker:
            score += 0.15
            why_parts.append("quantifier")
        if problem.has_negation_marker:
            score += 0.1
            why_parts.append("negation")
        if problem.has_unless_marker:
            score += 0.05
            why_parts.append("unless")
        score = min(1.0, score)
        # Comparison / superlative problems are better handled by a comparison
        # reasoner; we still apply but at lower preference.
        if problem.has_comparison_marker and not problem.has_conditional_marker:
            score = min(score, 0.55)
            why_parts.append("comparison_demoted")
        return MethodApplicability(score=score, why="+".join(why_parts))

    def solve(
        self,
        problem: LogicProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        trace = MethodTrace(method_id=self.method_id)
        for p in problem.normalized_premises:
            trace.inputs_seen.append(getattr(p, "id", "?"))
        started = time.perf_counter()

        if llm_client is None:
            trace.note("llm_client_unavailable")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id,
                family=self.family,
                answer=None,
                explanation="LLM-FOL+Z3 method requires an LLM client; abstaining.",
                confidence=0.0,
                trace=trace,
                abstained=True,
                abstain_reason="no_llm_client",
            )

        try:
            from app.logic.fol_z3_pipeline import solve_fol_z3, FolZ3Solution
        except Exception as exc:
            trace.note(f"pipeline_import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="FOL+Z3 pipeline unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="pipeline_unavailable",
            )

        # Soundness guard for contradictory ground premises (AGENTS.md §13.0,
        # Req 11.1): the FOL translator can silently drop a contradicting
        # premise; let the deterministic solver detect the conflict instead.
        try:
            from app.logic._fol_bridge import _premises_contain_contradiction
            if _premises_contain_contradiction(problem.normalized_premises):
                trace.note("ground_premises_contradict_skip_fol")
                trace.elapsed_ms = (time.perf_counter() - started) * 1000
                return MethodResult(
                    method_id=self.method_id, family=self.family, answer=None,
                    explanation="Ground premises are mutually contradictory; deferring to deterministic solver.",
                    confidence=0.0, trace=trace,
                    abstained=True, abstain_reason="ground_contradiction",
                )
        except Exception:
            pass  # Best-effort guard.

        payload = [
            {"id": getattr(p, "id", "?"), "text": getattr(p, "text", str(p))}
            for p in problem.normalized_premises
        ]
        try:
            result: FolZ3Solution = solve_fol_z3(
                problem.raw_question, payload, llm_client
            )
        except Exception as exc:
            trace.note(f"pipeline_error:{type(exc).__name__}:{exc}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"FOL+Z3 pipeline error: {exc}",
                confidence=0.0, trace=trace,
                error=str(exc), abstained=True, abstain_reason="pipeline_exception",
            )

        trace.llm_calls = int(getattr(result, "llm_calls", 0) or 0)
        trace.llm_roles = ["translator"] * trace.llm_calls
        for step in getattr(result, "proof_steps", []) or []:
            trace.step(str(step))

        z3_status = getattr(result, "z3_status", None)
        ans = (getattr(result, "answer", None) or "").strip()
        used_ids = list(getattr(result, "premises", []) or [])

        # Coverage gate: any normalized premise NOT in used_ids is a "drop".
        used_set = set(used_ids)
        for p in problem.normalized_premises:
            pid = getattr(p, "id", None)
            if pid and pid not in used_set:
                trace.drop(pid, "not_used_by_z3_theory")

        decisive = ans in {"yes", "no"} and z3_status in {"entailed", "contradicted"}

        # Confidence comes from backend signals (AGENTS.md §16), not the LLM.
        # Start from the translation confidence Z3 keeps, then taper for
        # missing premises (likely incomplete coverage).
        base_conf = float(getattr(result, "confidence", 0.0) or 0.0)
        coverage_ratio = (
            len(used_ids) / problem.premise_count if problem.premise_count else 0.0
        )
        confidence = base_conf * max(0.6, coverage_ratio)
        trace.elapsed_ms = (time.perf_counter() - started) * 1000

        if decisive:
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=ans, explanation=getattr(result, "explanation", "") or "",
                confidence=confidence, trace=trace,
                used_premise_ids=used_ids,
                z3_status=z3_status,
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=getattr(result, "explanation", "") or
                        "FOL+Z3 returned undetermined; abstaining.",
            confidence=confidence, trace=trace,
            used_premise_ids=used_ids, z3_status=z3_status,
            abstained=True, abstain_reason=z3_status or "non_decisive",
        )
