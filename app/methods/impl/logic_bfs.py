"""Method wrapper around the deterministic BFS / rule-based logic solver.

Fallback after FOL+Z3 abstains. Uses ``app.logic.solver._solve_rules`` and
``solve_forward_chaining`` exactly as before, but exposes them via the
uniform Method interface so the planner can decide order, verify coverage,
and roll up audit trails.
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


class LogicBfsMethod:
    """Deterministic BFS / forward-chaining logic solver.

    Always applicable when premises exist, with moderate score so FOL+Z3 is
    preferred when available. AGENTS.md §13.1: BFS runs only as fallback,
    not as primary, when LLM+Z3 path is reachable.
    """

    method_id: str = "logic.bfs_rules"
    family: MethodFamily = MethodFamily.LOGIC_SYMBOLIC
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: LogicProblem) -> MethodApplicability:
        if not isinstance(problem, LogicProblem):
            return MethodApplicability(0.0, "not_logic_problem")
        if problem.premise_count == 0:
            return MethodApplicability(0.0, "no_premises")
        # Default 0.4 — strictly below FOL+Z3's typical 0.5+.
        return MethodApplicability(0.4, "deterministic_fallback")

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

        try:
            from app.logic.solver import _solve_rules, solve_forward_chaining
        except Exception as exc:
            trace.note(f"solver_import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="BFS solver unavailable.", confidence=0.0,
                trace=trace, abstained=True, abstain_reason="bfs_unavailable",
            )

        # Step 1: rule-based shortcuts.
        try:
            ans, support, rule, cannot_prove = _solve_rules(
                problem.raw_question, problem.normalized_premises
            )
        except Exception as exc:
            trace.note(f"_solve_rules_error:{type(exc).__name__}")
            ans, support, rule, cannot_prove = "unknown", [], "error", True

        trace.step(f"_solve_rules => {ans} via {rule}")
        used_ids = [getattr(p, "id", "?") for p in (support or [])]

        if ans in {"yes", "no"}:
            confidence = 0.7 if rule and "modus" in str(rule).lower() else 0.6
            # Coverage drops:
            used_set = set(used_ids)
            for p in problem.normalized_premises:
                pid = getattr(p, "id", None)
                if pid and pid not in used_set:
                    trace.drop(pid, "not_used_by_rule")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=ans,
                explanation=f"BFS rule '{rule}' fired on {used_ids}.",
                confidence=confidence, trace=trace,
                used_premise_ids=used_ids,
            )

        # Step 2: forward chaining.
        try:
            fc = solve_forward_chaining(
                problem.raw_question, problem.normalized_premises
            )
        except Exception as exc:
            trace.note(f"forward_chaining_error:{type(exc).__name__}")
            fc = None
        if fc is not None:
            fc_ans, fc_sup, fc_reason = fc
            trace.step(f"forward_chaining => {fc_ans} via {fc_reason}")
            if fc_ans in {"yes", "no"}:
                used_ids = [getattr(p, "id", "?") for p in (fc_sup or [])]
                used_set = set(used_ids)
                for p in problem.normalized_premises:
                    pid = getattr(p, "id", None)
                    if pid and pid not in used_set:
                        trace.drop(pid, "not_used_by_forward_chaining")
                trace.elapsed_ms = (time.perf_counter() - started) * 1000
                return MethodResult(
                    method_id=self.method_id, family=self.family,
                    answer=fc_ans,
                    explanation=f"Forward chaining: {fc_reason}.",
                    confidence=0.65, trace=trace,
                    used_premise_ids=used_ids,
                )

        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation="BFS rules and forward chaining did not reach a definite answer.",
            confidence=0.3, trace=trace,
            abstained=True, abstain_reason="bfs_no_chain",
        )
