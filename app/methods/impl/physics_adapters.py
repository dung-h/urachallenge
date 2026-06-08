"""Method wrappers around the existing physics adapters.

Each ``app.physics.adapters.*`` instance is wrapped in a generic
``PhysicsAdapterMethod`` so the planner can pick / re-order / score / persist
adapters uniformly with logic methods. The wrapping is deliberately thin: the
adapter still owns its IR/equation logic — we only translate
applicability + result into the Method protocol.
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


class PhysicsAdapterMethod:
    """Generic wrapper exposing a ``PhysicsAdapter`` as a ``Method``.

    Adapters expose ``can_handle(problem) -> Applicability`` and
    ``solve(problem) -> AdapterSolution``. We map applicability score directly
    and translate ``AdapterSolution`` into ``MethodResult``.
    """

    family: MethodFamily = MethodFamily.PHYSICS_FORMULA
    source: MethodSource = MethodSource.BUILTIN

    def __init__(self, adapter: Any, *, method_id: str | None = None,
                 domain_hint: str | None = None) -> None:
        self._adapter = adapter
        self.method_id = method_id or f"physics.adapter.{type(adapter).__name__}"
        self._domain_hint = domain_hint or self._infer_domain_hint()

    def _infer_domain_hint(self) -> str | None:
        """Guess the domain hint from the adapter's class name."""
        name = type(self._adapter).__name__.lower()
        if "circuit" in name:
            return "circuit"
        if "mechanic" in name:
            return "mechanics"
        if "electrostatic" in name:
            return "electrostatics"
        if "optic" in name:
            return "optics"
        if "fluid" in name:
            return "fluids"
        if "thermal" in name:
            return "thermal"
        if "measure" in name:
            return "measurement"
        return None

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        # Domain hint fast filter — saves calling adapter.can_handle when
        # the adapter is clearly out of domain.
        if self._domain_hint and self._domain_hint not in problem.domain_hints:
            # Still allow the adapter to claim non-keyword cases (e.g.
            # measurement adapter handling generic "average" prompts) — give
            # it a low probe score so the planner only consults it after
            # higher-scoring candidates abstain.
            base = 0.2
        else:
            base = 0.7
        # Defer to the adapter's own can_handle if it implements one.
        try:
            can_handle = getattr(self._adapter, "can_handle", None)
            if callable(can_handle):
                applicability = can_handle(problem.parsed)
                if hasattr(applicability, "score"):
                    score = float(getattr(applicability, "score", 0.0))
                    why = str(getattr(applicability, "why", "")) or self.method_id
                    return MethodApplicability(score=max(score, 0.0), why=why)
                if isinstance(applicability, bool):
                    return MethodApplicability(
                        score=base if applicability else 0.0,
                        why="adapter_can_handle" if applicability else "adapter_skip",
                    )
                if isinstance(applicability, (int, float)):
                    return MethodApplicability(
                        score=float(applicability), why="adapter_score"
                    )
        except Exception:
            pass
        return MethodApplicability(score=base, why=f"domain_hint:{self._domain_hint or 'generic'}")

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
            adapter_result = self._adapter.solve(problem.parsed)
        except Exception as exc:
            trace.note(f"adapter_error:{type(exc).__name__}:{exc}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=f"Adapter error: {exc}",
                confidence=0.0, trace=trace,
                error=str(exc), abstained=True, abstain_reason="adapter_exception",
            )
        # Some adapters return None to signal "I don't apply".
        if adapter_result is None:
            trace.note("adapter_returned_none")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Adapter abstained on this problem.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="adapter_abstained",
            )
        # Adapters typically return a PhysicsSolution-like object with
        # .answer / .explanation / .formula_id / .variables / .cot / .confidence
        # / .success.
        success = bool(getattr(adapter_result, "success", True))
        ans = getattr(adapter_result, "answer", None)
        expl = getattr(adapter_result, "explanation", "") or ""
        formula_id = getattr(adapter_result, "formula_id", None)
        # Confidence: trust the adapter's value if present, else infer.
        confidence = float(getattr(adapter_result, "confidence", 0.0) or 0.0)
        if confidence == 0.0 and success and ans is not None and str(ans).strip():
            confidence = 0.85
        for step in getattr(adapter_result, "cot", []) or []:
            trace.step(str(step))
        trace.elapsed_ms = (time.perf_counter() - started) * 1000

        if success and ans is not None and str(ans).strip():
            return MethodResult(
                method_id=self.method_id, family=self.family,
                answer=str(ans), explanation=expl,
                confidence=confidence, trace=trace,
                formula_id=formula_id,
            )
        return MethodResult(
            method_id=self.method_id, family=self.family, answer=None,
            explanation=expl or "Adapter did not produce a verified answer.",
            confidence=confidence, trace=trace,
            formula_id=formula_id,
            abstained=True, abstain_reason="adapter_no_verified_answer",
        )
