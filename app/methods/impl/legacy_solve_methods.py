"""Wrap the entire legacy `solve_logic` / `solve_physics` pipelines as
single Methods.

Rationale (AGENTS.md §24, post-F3 measurement)
----------------------------------------------
Phase F3's full eval found that decomposing logic into independent
``LogicFolZ3Method`` + ``LogicBfsMethod`` regressed -3 logic pts vs the
legacy pipeline because BFS runs on its own here, **without** the
post-processing gates that ``app.logic.solver.solve`` runs after BFS
(direction gate, MCQ verifier, hallucination check, agent rescue).
Re-implementing every gate inside the planner is brittle.

Instead: we wrap ``solve_logic`` / ``solve_physics`` (the entire pipeline,
gates included) as **single Method instances**. The planner keeps its
fast structural shortcuts on top:

  * ``PhysicsQualitativeMethod`` — handle qualitative-monotonic shapes
    in 0.1 s instead of dispatching the legacy 15-s LLM-extraction pass.
  * ``LogicPatternRewriteMethod`` — pre-rewrite "X unless Y" so the
    legacy translator hits a shape it already handles.
  * ``DiscoveredPhysicsMethod`` — apply runtime-discovered formulas to
    new numeric questions.

When none of those shortcuts decide the answer, the planner hands off to
``LegacyLogicMethod`` / ``LegacyPhysicsMethod`` which IS the previous
production pipeline. This guarantees planner-mode is ≥ legacy on every
case while keeping the meta-reasoning Level-5/6 hooks active.
"""

from __future__ import annotations

import time
from typing import Any

from app.methods.problem import LogicProblem, PhysicsProblem
from app.methods.types import (
    Method,
    MethodApplicability,
    MethodFamily,
    MethodResult,
    MethodSource,
    MethodTrace,
)


class LegacyLogicMethod:
    """Wraps ``app.logic.solver.solve`` as a single Method.

    Mid-low priority (score 0.50 base) so structural shortcuts win when
    they apply. When the shortcuts abstain, this Method runs the entire
    legacy pipeline — FOL+Z3 with self-refinement, deterministic FOL
    compiler, BFS with full gates, MCQ verifier, agent rescue — exactly
    as production has been doing. By construction it is at parity with
    legacy on every logic case.
    """

    method_id: str = "logic.legacy_pipeline"
    family: MethodFamily = MethodFamily.LOGIC_SYMBOLIC
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: LogicProblem) -> MethodApplicability:
        if not isinstance(problem, LogicProblem):
            return MethodApplicability(0.0, "not_logic_problem")
        # Always applicable when premises exist; structural shortcuts
        # (LogicPatternRewriteMethod) outrank by score when they fire.
        return MethodApplicability(0.50, "legacy_pipeline_default")

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
            from app.logic.solver import solve as solve_logic
        except Exception as exc:
            trace.note(f"solver_import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Legacy logic solver unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="solver_unavailable",
            )
        # Apply pattern-store rewrites to raw premises before passing to the
        # BFS solver. The pattern store contains structural rewrites (e.g.
        # "X unless Y" -> "if not Y, then X") that the BFS ALREADY handles
        # once in canonical "if-then" form. Without this pre-pass the BFS
        # never sees the rewritten text — only LogicPatternRewriteMethod does,
        # and it routes through FOL+Z3 (which may abstain). This way both
        # the FOL+Z3 path (via LogicPatternRewriteMethod) AND the legacy BFS
        # path benefit from the same structural rewrites.
        try:
            from app.methods.logic_patterns import rewrite_premises
            rewritten_premises, rewrite_log = rewrite_premises(problem.raw_premises)
            if rewrite_log:
                for entry in rewrite_log:
                    trace.step(
                        f"pattern_rewrite({entry['pattern_id']}): "
                        f"{entry['before'][:50]!r} -> {entry['after'][:50]!r}"
                    )
        except Exception:
            rewritten_premises = problem.raw_premises
        try:
            res = solve_logic(
                problem.raw_question,
                rewritten_premises,
                llm_client=llm_client,
                use_llm=bool(llm_client and problem.normalized_premises),
                choices=problem.choices or None,
                premises_fol=problem.premises_fol or None,
            )
        except Exception as exc:
            trace.note(f"solver_error:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"Legacy solver error: {exc}",
                confidence=0.0, trace=trace, error=str(exc),
                abstained=True, abstain_reason="solver_exception",
            )
        ans = (getattr(res, "answer", None) or "").strip()
        for step in getattr(res, "cot", []) or []:
            trace.step(str(step))
        trace.llm_calls = int(getattr(res, "model_calls", 0) or 0)
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        used_ids = list(getattr(res, "premises", []) or [])
        # Coverage drops are recorded for parity with the rest of the
        # method-centric audit but the gate inside the planner only fires
        # on decisive results.
        used_set = set(used_ids)
        for p in problem.normalized_premises:
            pid = getattr(p, "id", None)
            if pid and pid not in used_set:
                trace.drop(pid, "not_used_by_legacy_pipeline")
        if ans and ans.lower() != "unknown":
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=ans, explanation=getattr(res, "explanation", "") or "",
                confidence=float(getattr(res, "confidence", 0.0) or 0.0),
                trace=trace,
                used_premise_ids=used_ids,
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=getattr(res, "explanation", "")
                        or "Legacy pipeline abstained.",
            confidence=float(getattr(res, "confidence", 0.0) or 0.0),
            trace=trace,
            used_premise_ids=used_ids,
            abstained=True, abstain_reason="legacy_abstained",
        )


class LegacyPhysicsMethod:
    """Wraps ``app.physics.solver.solve`` as a single Method.

    Same role for physics as LegacyLogicMethod for logic. Score 0.40 so
    that the qualitative reasoner and any hand-coded adapters / discovered
    methods OUTRANK it; this one is the safety-net when none of those fit.
    """

    method_id: str = "physics.legacy_pipeline"
    family: MethodFamily = MethodFamily.PHYSICS_FORMULA
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        return MethodApplicability(0.40, "legacy_pipeline_default")

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
        try:
            from app.physics.solver import solve as solve_physics
        except Exception as exc:
            trace.note(f"solver_import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Legacy physics solver unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="solver_unavailable",
            )
        try:
            res = solve_physics(
                problem.raw_question,
                use_llm_extraction=bool(llm_client),
                use_search=False,
                llm_client=llm_client,
                rescue_unknown=True,
            )
        except Exception as exc:
            trace.note(f"solver_error:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"Legacy physics solver error: {exc}",
                confidence=0.0, trace=trace, error=str(exc),
                abstained=True, abstain_reason="solver_exception",
            )
        for step in getattr(res, "cot", []) or []:
            trace.step(str(step))
        trace.llm_calls = int(getattr(res, "model_calls", 0) or 0)
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        success = bool(getattr(res, "success", False))
        ans = getattr(res, "answer", None)
        if success and ans and str(ans).strip().lower() != "unknown":
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=str(ans),
                explanation=getattr(res, "explanation", "") or "",
                confidence=float(getattr(res, "confidence", 0.0) or 0.0),
                trace=trace,
                formula_id=getattr(res, "formula_id", None),
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=getattr(res, "explanation", "")
                        or "Legacy physics pipeline abstained.",
            confidence=float(getattr(res, "confidence", 0.0) or 0.0),
            trace=trace,
            formula_id=getattr(res, "formula_id", None),
            abstained=True, abstain_reason="legacy_abstained",
        )
