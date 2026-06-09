"""Method wrapping the deterministic qualitative reasoner.

The legacy `_solve_impl` runs a separate branch (`_solve_qualitative`) when
the question is a qualitative monotonic-reasoning shape ("If R increases,
what happens to I?"). Without this wrapper, the planner sends such
questions to numeric adapters / retrieval and they wrongly attempt to
compute. This Method exposes the existing deterministic qualitative
reasoner as a uniform Method so the planner picks it first when the
qualitative parser flags the shape.
"""

from __future__ import annotations

import os
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


class PhysicsQualitativeMethod:
    """Run the deterministic qualitative monotonic reasoner.

    Applicability is gated by ``parsed.qualitative`` being populated AND the
    feature flag ``URA_ENABLE_QUALITATIVE_PARSER`` being on (which the
    upstream service sets in production). High score because qualitative
    questions have a wrong-but-plausible numeric path that retrieval will
    happily go down if we don't preempt it here.
    """

    method_id: str = "physics.qualitative_reasoner"
    family: MethodFamily = MethodFamily.PHYSICS_FORMULA
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        # Cheap structural check first: the parser must have populated the
        # qualitative field (filled by `app.physics.qualitative_parser`).
        qq = getattr(problem.parsed, "qualitative", None)
        if qq is None:
            return MethodApplicability(0.0, "no_qualitative_shape")
        # Even if parsed, only run when the upstream allows it.
        try:
            from app.physics.qualitative_parser import qualitative_parser_enabled
            if not qualitative_parser_enabled():
                return MethodApplicability(0.0, "qualitative_disabled")
        except Exception:
            return MethodApplicability(0.0, "qualitative_module_unavailable")
        # Strong score: this method is the canonical path for these questions.
        return MethodApplicability(0.95, "qualitative_shape_matched")

    def solve(
        self,
        problem: PhysicsProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        trace = MethodTrace(method_id=self.method_id)
        trace.inputs_seen.append("physics.parsed.qualitative")
        started = time.perf_counter()
        try:
            from app.physics.solver import _solve_qualitative
        except Exception as exc:
            trace.note(f"import_failed:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Qualitative reasoner unavailable.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="module_unavailable",
            )
        qq = getattr(problem.parsed, "qualitative", None)
        if qq is None:
            trace.note("no_qualitative_shape")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Question is not a qualitative monotonic shape.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_qualitative_shape",
            )
        try:
            result = _solve_qualitative(problem.parsed, qq)
        except Exception as exc:
            trace.note(f"solver_error:{type(exc).__name__}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"Qualitative reasoner error: {exc}",
                confidence=0.0, trace=trace, error=str(exc),
                abstained=True, abstain_reason="solver_exception",
            )
        for step in getattr(result, "cot", []) or []:
            trace.step(str(step))
        success = bool(getattr(result, "success", True))
        ans = getattr(result, "answer", None)
        expl = getattr(result, "explanation", "") or ""
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        if success and ans and str(ans).strip().lower() != "unknown":
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=str(ans), explanation=expl,
                confidence=confidence if confidence > 0 else 0.9,
                trace=trace,
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=expl or "Qualitative reasoner abstained.",
            confidence=confidence, trace=trace,
            abstained=True, abstain_reason="qualitative_no_answer",
        )
