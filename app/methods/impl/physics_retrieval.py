"""Method wrapping the retrieval-grounded physics solver.

This is the **Level-6 path** that is already in-tree:
``app.physics.retrieval_grounded_method`` searches the web (or local corpus)
for the right method, asks the LLM to pick a formula grounded in the
retrieved snippets, then re-computes with safe_eval and applies a
dimensional gate. We expose it as a Method so the planner can decide WHEN
to fire it (after hand-coded adapters abstain) instead of it being a
hidden fallback inside ``solver.py``.
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
from app.methods.problem import PhysicsProblem


class PhysicsRetrievalMethod:
    """Search the web for the solving method, ground the LLM, recompute."""

    method_id: str = "physics.retrieval_grounded"
    family: MethodFamily = MethodFamily.PHYSICS_RETRIEVAL
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        if problem.is_lookup_question:
            # Conceptual/unit-name questions are best handled by the
            # conceptual lookup helper, not the formula retrieval path.
            return MethodApplicability(0.3, "lookup_question_secondary")
        # Block retrieval when the question carries no numeric quantities AND
        # carries qualitative-change wording. The qualitative parser may have
        # missed the shape (e.g. uncovered variable like "turns" / "magnetic
        # field"), but the wording still implies a SYMBOLIC monotonic answer
        # (doubled / halved / increases / remains constant). Retrieval will
        # happily compute a numeric value here and return a wrong answer; the
        # legacy fallback inside `solve_physics` handles these cases without
        # invoking retrieval at all. Generalizes over every "qualitative shape
        # the parser missed" case (AGENTS.md §20.1: structural, no per-question
        # text).
        low = problem.raw_question.lower()
        qualitative_change_words = (
            "doubled", "tripled", "halved", "quadrupled",
            "increases", "decreases", "increased", "decreased",
            "increasing", "decreasing", "what happens to", "remains constant",
        )
        has_qual_change = any(w in low for w in qualitative_change_words)
        if has_qual_change and problem.quantity_count == 0:
            return MethodApplicability(0.05, "qualitative_change_no_numeric")
        # Block retrieval for "where is X stored" / "where does X happen"
        # conceptual questions — they have no numeric to compute, retrieval
        # would invent a formula. Same structural guard as above.
        conceptual_locator_words = (
            "where is ", "where are ", "where does ", "where do ",
            "in which ", "at which ", "at what point",
        )
        has_locator = any(w in low for w in conceptual_locator_words)
        if has_locator and problem.quantity_count == 0:
            return MethodApplicability(0.05, "conceptual_locator_no_numeric")
        if not problem.has_units and problem.target_quantity is None:
            return MethodApplicability(0.2, "no_target_or_units")
        # Retrieval is most valuable when hand-coded adapters likely don't
        # cover the topic. Lower default than adapters; the planner picks it
        # up when adapters abstain.
        return MethodApplicability(0.45, "retrieval_default")

    def solve(
        self,
        problem: PhysicsProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        trace = MethodTrace(method_id=self.method_id)
        trace.inputs_seen.append("physics.parsed")
        started = time.perf_counter()
        if llm_client is None:
            trace.note("llm_client_unavailable")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Retrieval-grounded method requires an LLM client.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_llm_client",
            )
        try:
            from app.physics.retrieval_grounded_method import (
                solve_with_retrieved_method,
            )
        except Exception as exc:
            trace.note(f"import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Retrieval module unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="module_unavailable",
            )
        try:
            grounded = solve_with_retrieved_method(
                problem.parsed, problem.raw_question, llm_client
            )
        except Exception as exc:
            trace.note(f"retrieval_error:{type(exc).__name__}:{exc}")
            grounded = None

        if grounded is None:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Retrieval-grounded method abstained.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="retrieval_abstained",
            )

        for step in getattr(grounded, "cot", []) or []:
            trace.step(str(step))
        trace.llm_calls = 1
        trace.llm_roles = ["formula_selector"]
        trace.elapsed_ms = (time.perf_counter() - started) * 1000

        return MethodResult(
            method_id=self.method_id, family=self.family,
            answer=str(grounded.answer),
            explanation=grounded.explanation,
            confidence=float(grounded.confidence),
            trace=trace,
            formula_id=grounded.formula_name,
            numeric_value=float(grounded.value),
            numeric_unit=grounded.target_unit,
        )


class PhysicsConceptualLookupMethod:
    """Wraps ``solve_conceptual_lookup`` for definitional / unit-name questions."""

    method_id: str = "physics.conceptual_lookup"
    family: MethodFamily = MethodFamily.PHYSICS_RETRIEVAL
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        if not problem.is_lookup_question:
            return MethodApplicability(0.0, "not_lookup_question")
        return MethodApplicability(0.85, "conceptual_lookup")

    def solve(
        self,
        problem: PhysicsProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        trace = MethodTrace(method_id=self.method_id)
        trace.inputs_seen.append("physics.question")
        started = time.perf_counter()
        if llm_client is None:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Lookup requires an LLM client.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_llm_client",
            )
        try:
            from app.physics.retrieval_grounded_method import (
                solve_conceptual_lookup,
            )
        except Exception as exc:
            trace.note(f"import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Lookup module unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="module_unavailable",
            )
        try:
            res = solve_conceptual_lookup(
                problem.parsed, problem.raw_question, llm_client
            )
        except Exception as exc:
            trace.note(f"lookup_error:{type(exc).__name__}:{exc}")
            res = None
        if res is None:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Lookup abstained.", confidence=0.0,
                trace=trace, abstained=True, abstain_reason="lookup_abstained",
            )
        for step in getattr(res, "cot", []) or []:
            trace.step(str(step))
        trace.llm_calls = 1
        trace.llm_roles = ["fact_extractor"]
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        return MethodResult(
            method_id=self.method_id, family=self.family,
            answer=str(res.answer), explanation=res.explanation,
            confidence=float(res.confidence), trace=trace,
        )
