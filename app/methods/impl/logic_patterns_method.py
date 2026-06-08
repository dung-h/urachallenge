"""Logic Method that rewrites premises through `LogicPatternStore` before
handing them to FOL+Z3.

Sits BEFORE `LogicFolZ3Method` in the planner shortlist when at least one
pattern matches an input premise. The rewrite is purely STRUCTURAL (no LLM
call here) so it is fast: when "All birds can fly unless they are
penguins" is rewritten to "if not they are penguins, then all birds can
fly", the FOL+Z3 translator hits a shape it already handles.

If the rewrite still doesn't produce a decisive Z3 verdict, this Method
abstains and the planner falls through to the regular FOL+Z3 / BFS chain
on the ORIGINAL premise text — the rewrite is preferential, not exclusive.
"""

from __future__ import annotations

import time
from typing import Any

from app.methods.logic_patterns import (
    LogicPatternStore,
    get_default_pattern_store,
    rewrite_premises,
)
from app.methods.problem import LogicProblem
from app.methods.types import (
    Method,
    MethodApplicability,
    MethodFamily,
    MethodResult,
    MethodSource,
    MethodTrace,
)


class LogicPatternRewriteMethod:
    """Pre-rewrite logic method.

    Applicability: at least one pattern in the store matches at least one
    premise. The score is set HIGH (0.95) when any pattern fires so the
    planner runs the rewrite BEFORE the more expensive FOL+Z3 translation.
    Score is 0 otherwise so the method silently steps aside.
    """

    method_id: str = "logic.pattern_rewrite_then_fol_z3"
    family: MethodFamily = MethodFamily.LOGIC_RETRIEVAL
    source: MethodSource = MethodSource.BUILTIN

    def __init__(self, *, store: LogicPatternStore | None = None) -> None:
        self._store = store or get_default_pattern_store()

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: LogicProblem) -> MethodApplicability:
        if not isinstance(problem, LogicProblem):
            return MethodApplicability(0.0, "not_logic_problem")
        if problem.premise_count == 0:
            return MethodApplicability(0.0, "no_premises")
        # Cheap scan: any pattern matches ANY premise?
        for premise in (getattr(p, "text", "") for p in problem.normalized_premises):
            if self._store.find_match(str(premise)) is not None:
                return MethodApplicability(0.95, "pattern_match_found")
        return MethodApplicability(0.0, "no_pattern_match")

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

        # 1. Apply pattern rewrites.
        original_texts = [getattr(p, "text", "") for p in problem.normalized_premises]
        rewritten_texts, applied_log = rewrite_premises(original_texts, store=self._store)
        if not applied_log:
            trace.note("no_patterns_fired")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="No logic patterns fired; abstaining.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_pattern_fired",
            )
        for entry in applied_log:
            trace.step(
                f"pattern {entry['pattern_id']}: {entry['before'][:60]!r} -> "
                f"{entry['after'][:60]!r}"
            )

        # 2. Re-normalize so premise IDs / parse flags match the rewritten text.
        try:
            from app.logic.premise_selector import normalize_premises
            rewritten_normalized = normalize_premises(rewritten_texts)
        except Exception as exc:
            trace.note(f"renorm_error:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Could not re-normalize rewritten premises.",
                confidence=0.0, trace=trace, error=str(exc),
                abstained=True, abstain_reason="renorm_failed",
            )

        # 3. Hand off to FOL+Z3 (the same wrapped Method) on the rewritten
        #    problem. We don't recurse through the planner — direct call,
        #    bounded budget.
        if llm_client is None:
            trace.note("no_llm_client_for_fol_z3")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Pattern rewrite needs FOL+Z3 with LLM client.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_llm_client",
            )
        try:
            from app.logic.fol_z3_pipeline import solve_fol_z3
        except Exception:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="FOL+Z3 unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="fol_z3_unavailable",
            )
        payload = [
            {"id": getattr(p, "id", "?"), "text": getattr(p, "text", str(p))}
            for p in rewritten_normalized
        ]
        try:
            result = solve_fol_z3(problem.raw_question, payload, llm_client)
        except Exception as exc:
            trace.note(f"fol_z3_error:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"FOL+Z3 on rewritten premises errored: {exc}",
                confidence=0.0, trace=trace, error=str(exc),
                abstained=True, abstain_reason="fol_z3_exception",
            )
        trace.llm_calls = int(getattr(result, "llm_calls", 0) or 0)
        trace.llm_roles = ["translator"] * trace.llm_calls
        for step in getattr(result, "proof_steps", []) or []:
            trace.step(str(step))

        ans = (getattr(result, "answer", None) or "").strip()
        z3_status = getattr(result, "z3_status", None)
        used_ids = list(getattr(result, "premises", []) or [])
        decisive = ans in {"yes", "no"} and z3_status in {"entailed", "contradicted"}

        # Pattern-store stats: every applied pattern shares the outcome.
        for entry in applied_log:
            self._store.record(entry["pattern_id"], success=decisive)

        # Translate rewritten premise IDs back to original IDs (positional).
        try:
            id_map = {
                getattr(rw, "id", ""): getattr(orig, "id", "")
                for rw, orig in zip(rewritten_normalized, problem.normalized_premises)
            }
            mapped_ids = [id_map.get(pid, pid) for pid in used_ids]
        except Exception:
            mapped_ids = used_ids

        used_set = set(mapped_ids)
        for p in problem.normalized_premises:
            pid = getattr(p, "id", None)
            if pid and pid not in used_set:
                trace.drop(pid, "not_used_after_rewrite")

        base_conf = float(getattr(result, "confidence", 0.0) or 0.0)
        coverage_ratio = (
            len(mapped_ids) / problem.premise_count if problem.premise_count else 0.0
        )
        confidence = base_conf * max(0.6, coverage_ratio)
        trace.elapsed_ms = (time.perf_counter() - started) * 1000

        if decisive:
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=ans, explanation=getattr(result, "explanation", "") or "",
                confidence=confidence, trace=trace,
                used_premise_ids=mapped_ids,
                z3_status=z3_status,
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=getattr(result, "explanation", "")
                        or "Pattern-rewritten FOL+Z3 abstained.",
            confidence=confidence, trace=trace,
            used_premise_ids=mapped_ids, z3_status=z3_status,
            abstained=True, abstain_reason=z3_status or "non_decisive_after_rewrite",
        )
