"""Method-aware planner: meta-reasoning over the MethodLibrary.

This is the agent that chooses HOW to solve a problem instead of running a
fixed pipeline. The choice is two-tier:

  1. **Deterministic shortlist**: every Method scores its applicability to
     the problem. The shortlist is the set with score ≥ ``MIN_APPLICABLE`` —
     no LLM needed for this step.

  2. **LLM-aided ranking** (optional, budget-permitting): when several
     methods are applicable AND the problem looks novel (e.g. domain hint
     not matched, retrieval expected, premise count high), the planner can
     ask the LLM to RE-RANK the shortlist using its own heuristic. This is
     bounded by the request's CallBudget and falls back gracefully when the
     LLM is unavailable.

After ranking, the planner runs methods in order. Each result goes through
faithfulness + coverage gates (when applicable) and is either ACCEPTED
(returned), DOWNGRADED (kept as fallback), or REJECTED (skipped). If every
shortlisted method abstains the planner triggers ``MethodDiscovery`` to
attempt to learn a new method, then re-runs.

The planner itself is a Method-of-meta-methods, but it is NOT registered in
the library — it sits one layer above as the orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.methods.coverage import check_coverage, downgrade_to_abstain
from app.methods.discovery import discover_physics_method, discover_logic_method
from app.methods.faithfulness import check_atom_coverage
from app.methods.library import MethodLibrary, get_default_library
from app.methods.problem import LogicProblem, PhysicsProblem
from app.methods.types import (
    Method,
    MethodApplicability,
    MethodResult,
    planner_sort_key,
)


MIN_APPLICABLE = 0.3   # methods scoring below this are dropped from the shortlist


@dataclass
class PlannerDecision:
    """Audit of a single planner-method invocation step."""

    method_id: str
    applicability_score: float
    applicability_why: str
    accepted: bool
    abstained: bool
    coverage_passed: bool
    elapsed_ms: float
    note: str | None = None


@dataclass
class PlannerOutcome:
    """The full planner trace + final accepted result (if any)."""

    final: MethodResult | None
    decisions: list[PlannerDecision] = field(default_factory=list)
    discovery_attempted: bool = False
    discovery_outcome: str | None = None
    elapsed_ms: float = 0.0
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final": self.final.to_dict() if self.final else None,
            "decisions": [d.__dict__ for d in self.decisions],
            "discovery_attempted": self.discovery_attempted,
            "discovery_outcome": self.discovery_outcome,
            "elapsed_ms": self.elapsed_ms,
            "abstain_reason": self.abstain_reason,
        }


class MethodPlanner:
    """Selects, runs, and gates Methods for a single request."""

    def __init__(self, *, library: MethodLibrary | None = None) -> None:
        self._library = library or get_default_library()

    # --- self-consistency vote ---------------------------------------------

    @staticmethod
    def _is_self_consistency_enabled() -> bool:
        """Per AGENTS.md §16 + §24, self-consistency is on by default and can
        be disabled with ``URA_PLANNER_SELF_CONSISTENCY=0`` for A/B testing.
        """
        import os as _os
        raw = _os.environ.get("URA_PLANNER_SELF_CONSISTENCY", "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _magnitude_sanity_check(problem: Any, primary: "MethodResult") -> dict[str, Any]:
        """Backend sanity bound on the SI-base value vs typical physics ranges.

        Catches the wrong-with-confidence failure mode that a method-vs-method
        consistency vote misses: when primary and verifier INDEPENDENTLY emit
        the same wrong equation (e.g. drop a 2*pi factor or a 10^3 mA→A
        conversion), they agree on the wrong number. A purely backend
        sanity range — bounded by the physically plausible scale of the
        target quantity — rejects the answer regardless.

        The bounds are deliberately wide (5+ orders of magnitude) so they
        only trigger on egregiously-off scaling, not on borderline cases.
        Returns ``{"rejected": bool, "note": str}``.
        """
        target = (getattr(problem, "target_quantity", None) or "").lower()
        unit = (primary.numeric_unit or "").strip()
        value = primary.numeric_value
        if value is None:
            return {"rejected": False, "note": "no_numeric_value"}
        # Convert to SI base when possible.
        try:
            from app.physics.unit_converter import convert_value, normalize_unit
            si_value, si_unit = convert_value(float(value), normalize_unit(unit) if unit else "")
        except Exception:
            si_value, si_unit = float(value), unit
        absv = abs(float(si_value))
        # Target → (lo, hi) plausible SI range. Wide bounds; only outliers.
        # NB: only quantities where a wrong scale factor is the dominant
        # failure mode are listed. Other targets pass through unchecked.
        BOUNDS: dict[str, tuple[float, float]] = {
            "current": (1e-9, 1e3),                 # nA to kA
            "voltage": (1e-6, 1e6),                 # uV to MV
            "resistance": (1e-3, 1e9),              # mOhm to GOhm
            "capacitance": (1e-15, 1.0),            # fF to F
            "charge": (1e-15, 1e3),                 # fC to kC
            "frequency": (1e-3, 1e12),              # mHz to THz
            "power": (1e-9, 1e9),                   # nW to GW
            "energy": (1e-9, 1e15),                 # nJ to PJ
            "force": (1e-15, 1e9),                  # fN to GN
            "magnetic_field": (1e-9, 1e3),          # nT to kT
            "electric_field": (1e-3, 1e15),         # mV/m to PV/m
            "speed": (1e-6, 3e8),                   # cap at c
            "velocity": (1e-6, 3e8),
            "acceleration": (1e-6, 1e6),
            "wavelength": (1e-15, 1e5),             # fm to 100 km
        }
        bounds = BOUNDS.get(target)
        if bounds is None:
            return {"rejected": False, "note": f"no_bounds_for_target:{target!r}"}
        lo, hi = bounds
        if absv == 0:
            # zero is suspicious but not forbidden (e.g. equilibrium force);
            # don't reject on zero alone.
            return {"rejected": False, "note": "zero_value"}
        if absv < lo or absv > hi:
            return {
                "rejected": True,
                "note": (
                    f"out_of_plausible_range:{target}|{absv:.4g}{si_unit or '?'}|"
                    f"bounds=({lo:.0e},{hi:.0e})"
                ),
            }
        # Input-scale consistency: if the question's knowns of the same
        # target dimension all sit in a tight range, the answer should not
        # be ≥ 100× outside that range. Catches the "9 mA → 9 A" failure
        # where the LLM dropped the milli prefix on the answer but the
        # question's knowns were in mA. Only fires when at least 2 knowns
        # are in a consistent decade — single-known questions skip.
        try:
            quantities = list(getattr(problem.parsed, "quantities", []) or [])
            same_unit_si_values = []
            for q in quantities:
                qu = (getattr(q, "si_unit", "") or "").strip()
                qv = float(getattr(q, "si_value", 0.0) or 0.0)
                if qu == si_unit and qv != 0:
                    same_unit_si_values.append(abs(qv))
            if len(same_unit_si_values) >= 2:
                lo_in = min(same_unit_si_values)
                hi_in = max(same_unit_si_values)
                # 2-decade tolerance both sides; outside → suspicious.
                if absv < lo_in / 100 or absv > hi_in * 100:
                    return {
                        "rejected": True,
                        "note": (
                            f"input_scale_mismatch:{target}|"
                            f"answer={absv:.4g}_input=[{lo_in:.4g},{hi_in:.4g}]"
                        ),
                    }
        except Exception:
            pass
        return {"rejected": False, "note": f"in_range:{absv:.4g}{si_unit or '?'}"}

    def _self_consistency_vote(
        self,
        problem: Any,
        primary: "MethodResult",
        *,
        llm_client: Any | None,
        budget: Any | None,
    ) -> dict[str, Any]:
        """Run the legacy physics pipeline as an INDEPENDENT verifier and
        compare its numeric answer to the primary method's result.

        Returns a dict with keys:
          - verifier_method: which method was used to verify
          - verifier_value: float or None (the verifier's numeric value)
          - verifier_abstained: True iff the verifier did not produce a number
          - agree: True / False / None (True iff numbers agree within
            5% relative tolerance over the same SI base unit; None iff
            cannot decide because verifier abstained)
          - note: short audit string
          - elapsed_ms: wall time of the vote
        """
        import time as _time
        verifier_method_id = "physics.legacy_pipeline"
        verifier = self._library.get(verifier_method_id)
        if verifier is None:
            return {
                "verifier_method": verifier_method_id,
                "verifier_value": None,
                "verifier_abstained": True,
                "agree": None,
                "note": "verifier_unavailable",
                "elapsed_ms": 0.0,
            }
        t0 = _time.perf_counter()
        try:
            # The verifier runs with a FRESH budget (budget=None), NOT the
            # primary's already-consumed budget. A self-consistency vote is a
            # verification step where correctness outranks latency
            # (AGENTS.md §15.4: a verified answer within the ~30s budget beats
            # a fast unverified one). Passing the depleted primary budget here
            # starved the verifier into abstaining ("verifier_abstain"), which
            # made the vote fail-OPEN — the planner then blindly trusted the
            # primary even when an independent solver would have disagreed
            # (this masked the parallel current-divider wrong-principle bug:
            # equation_graph said 48 mA, legacy said 9 mA, but legacy abstained
            # in the vote because it had no budget left). A fresh budget lets
            # the verifier actually produce its independent witness.
            verifier_result = verifier.solve(
                problem, llm_client=llm_client, budget=None
            )
        except Exception as exc:
            return {
                "verifier_method": verifier_method_id,
                "verifier_value": None,
                "verifier_abstained": True,
                "agree": None,
                "note": f"verifier_error:{type(exc).__name__}",
                "elapsed_ms": (_time.perf_counter() - t0) * 1000,
            }
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # Extract a numeric value from the verifier's answer. The legacy
        # PhysicsSolution does not always set numeric_value, so parse the
        # answer string as a fallback.
        verifier_value = verifier_result.numeric_value
        verifier_unit = verifier_result.numeric_unit or ""
        if verifier_value is None and verifier_result.answer:
            import re as _re
            m = _re.search(
                r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", str(verifier_result.answer)
            )
            if m:
                try:
                    verifier_value = float(m.group(1))
                    # Best-effort unit grab.
                    u = _re.search(
                        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*([a-zA-Z\u03A9\u00B0\u03BC/^*\u00B7]+)",
                        str(verifier_result.answer),
                    )
                    if u and not verifier_unit:
                        verifier_unit = u.group(1)
                except Exception:
                    verifier_value = None

        if verifier_result.abstained or verifier_value is None:
            return {
                "verifier_method": verifier_method_id,
                "verifier_value": verifier_value,
                "verifier_abstained": True,
                "agree": None,
                "note": "verifier_abstain",
                "elapsed_ms": elapsed_ms,
            }

        # Strength signal: did the verifier hit a hand-coded adapter
        # formula? If so, its answer is grounded in a deterministic
        # solver, not just LLM rescue. The override below treats this
        # as a witness on the verifier's side.
        verifier_has_formula = bool(verifier_result.formula_id)
        verifier_confidence = float(verifier_result.confidence or 0.0)
        verifier_formula_id = str(verifier_result.formula_id or "")

        # Compare in SI base units when both sides have units; else compare
        # raw numbers with the standard 5% relative tolerance.
        primary_value = float(primary.numeric_value)
        primary_unit = primary.numeric_unit or ""
        try:
            from app.physics.unit_converter import convert_value, normalize_unit
            p_si_val, p_si_unit = primary_value, ""
            v_si_val, v_si_unit = float(verifier_value), ""
            if primary_unit:
                p_si_val, p_si_unit = convert_value(primary_value, normalize_unit(primary_unit))
            if verifier_unit:
                v_si_val, v_si_unit = convert_value(float(verifier_value), normalize_unit(verifier_unit))
            if p_si_unit and v_si_unit and p_si_unit != v_si_unit:
                # Different SI bases means the methods solved for DIFFERENT
                # quantities — that's a hard disagree.
                return {
                    "verifier_method": verifier_method_id,
                    "verifier_value": float(verifier_value),
                    "verifier_abstained": False,
                    "agree": False,
                    "note": f"si_unit_mismatch:{p_si_unit}_vs_{v_si_unit}",
                    "verifier_has_formula": verifier_has_formula,
                    "verifier_confidence": verifier_confidence,
                    "verifier_formula_id": verifier_formula_id,
                    "elapsed_ms": elapsed_ms,
                }
            primary_value, verifier_value = p_si_val, v_si_val
        except Exception:
            pass  # best-effort SI normalization

        if primary_value == 0:
            agree = abs(float(verifier_value)) < 1e-9
        else:
            rel_err = abs(primary_value - float(verifier_value)) / abs(primary_value)
            agree = rel_err < 0.05
        return {
            "verifier_method": verifier_method_id,
            "verifier_value": float(verifier_value),
            "verifier_abstained": False,
            "agree": bool(agree),
            "note": (
                f"agree(rel_err<5%)" if agree
                else f"disagree:p={primary_value:.4g}_v={float(verifier_value):.4g}"
            ),
            "verifier_has_formula": verifier_has_formula,
            "verifier_confidence": verifier_confidence,
            "verifier_formula_id": verifier_formula_id,
            "elapsed_ms": elapsed_ms,
        }

    # --- shortlisting -------------------------------------------------------

    def shortlist(self, problem: Any) -> list[tuple[Method, MethodApplicability]]:
        """Score every method and return those at or above ``MIN_APPLICABLE``."""
        candidates: list[tuple[Method, MethodApplicability]] = []
        for method in self._library.all():
            try:
                applicability = method.score_match(problem)
            except Exception:
                continue
            if applicability.score >= MIN_APPLICABLE:
                candidates.append((method, applicability))
        candidates.sort(key=lambda pair: planner_sort_key(pair[0], pair[1]))
        return candidates

    # --- main entrypoint ----------------------------------------------------

    def solve(
        self,
        problem: Any,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
        allow_discovery: bool = True,
    ) -> PlannerOutcome:
        """Run the planner over ``problem``.

        Tries each shortlisted method in order. Accepts the first decisive
        result that passes faithfulness + coverage. If nothing is decisive
        AND ``allow_discovery=True`` AND the problem is a physics problem
        with an LLM client, attempts ``MethodDiscovery``.
        """
        started = time.perf_counter()
        outcome = PlannerOutcome(final=None)
        shortlist = self.shortlist(problem)
        if not shortlist:
            outcome.elapsed_ms = (time.perf_counter() - started) * 1000
            outcome.abstain_reason = "no_applicable_method"
            return outcome

        # The first decisive, gated result wins.
        # We also keep the best-ranked DOWNGRADED result as a soft fallback so
        # the planner can return *something* with low confidence rather than
        # nothing when every method abstained borderline.
        best_downgrade: MethodResult | None = None

        for method, applicability in shortlist:
            t0 = time.perf_counter()
            try:
                result = method.solve(
                    problem, llm_client=llm_client, budget=budget
                )
            except Exception as exc:
                outcome.decisions.append(
                    PlannerDecision(
                        method_id=method.method_id,
                        applicability_score=applicability.score,
                        applicability_why=applicability.why,
                        accepted=False,
                        abstained=False,
                        coverage_passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000,
                        note=f"method_error:{type(exc).__name__}:{exc}",
                    )
                )
                continue

            # Coverage gate (only when expected_inputs is meaningful — currently
            # only for logic problems with normalized premises). The gate is
            # only meaningful for DECISIVE results: an abstaining method
            # legitimately produces an empty `used_premise_ids` and is not a
            # silent drop. Running coverage on abstains would falsely fail
            # every fallthrough.
            coverage_passed = True
            coverage_note: str | None = None
            expected_ids: list[str] = []
            if isinstance(problem, LogicProblem) and result.decisive:
                expected_ids = [
                    str(getattr(p, "id", ""))
                    for p in problem.normalized_premises
                    if getattr(p, "id", "")
                ]
                report = check_coverage(expected_ids, result)
                coverage_passed = report.accepted
                coverage_note = report.note
                if not coverage_passed:
                    # Downgrade rather than reject so the audit trail surfaces
                    # the offending silent drop.
                    result = downgrade_to_abstain(result, report)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            decisive = result.decisive
            outcome.decisions.append(
                PlannerDecision(
                    method_id=method.method_id,
                    applicability_score=applicability.score,
                    applicability_why=applicability.why,
                    accepted=decisive,
                    abstained=bool(result.abstained),
                    coverage_passed=coverage_passed,
                    elapsed_ms=elapsed_ms,
                    note=coverage_note,
                )
            )

            # Stats update — important for library scoring.
            try:
                self._library.record_use(
                    method.method_id,
                    success=decisive,
                    abstained=bool(result.abstained),
                    error=bool(result.error),
                    confidence=float(result.confidence),
                    when_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
            except Exception:
                pass

            if decisive and coverage_passed:
                # Self-consistency vote (Phase F-extra Idea #1, AGENTS.md §24
                # move-without-asking). For NUMERIC physics answers from a
                # non-legacy method, run an independent verifier (the legacy
                # pipeline) and:
                #   - agree → trust + confidence boost.
                #   - disagree → reject this primary result and fall through
                #     so the planner keeps walking the shortlist.
                #   - verifier abstains → no info; trust primary as-is.
                # Logic answers (yes/no/unknown) are excluded from this vote
                # because the legacy logic pipeline ALREADY runs the same
                # FOL+Z3 + BFS chain a planner-side method would, so a
                # verification call would just duplicate the primary work.
                if (
                    self._is_self_consistency_enabled()
                    and isinstance(problem, PhysicsProblem)
                    and method.method_id != "physics.legacy_pipeline"
                    and result.numeric_value is not None
                ):
                    # 1. Magnitude sanity bound (cheap, no LLM call): if the
                    # result's SI value is wildly out of typical range for
                    # its target quantity, reject. This catches the failure
                    # mode where primary AND verifier independently emit the
                    # same wrong equation (e.g. drop a 1000× factor) so a
                    # method-vs-method consistency check would miss it.
                    bound_check = self._magnitude_sanity_check(problem, result)
                    if bound_check["rejected"]:
                        result.trace.note(
                            f"magnitude_bound_rejected: {bound_check['note']}"
                        )
                        outcome.decisions.append(
                            PlannerDecision(
                                method_id="bound_check",
                                applicability_score=0.0,
                                applicability_why="magnitude_sanity",
                                accepted=False,
                                abstained=False,
                                coverage_passed=True,
                                elapsed_ms=0.0,
                                note=bound_check["note"],
                            )
                        )
                        if (
                            best_downgrade is None
                            or float(result.confidence) > float(best_downgrade.confidence)
                        ):
                            best_downgrade = result
                        continue

                    # 2. Method-vs-method consistency vote.
                    vote = self._self_consistency_vote(
                        problem, result, llm_client=llm_client, budget=budget
                    )
                    outcome.decisions.append(
                        PlannerDecision(
                            method_id=f"verifier:{vote['verifier_method']}",
                            applicability_score=0.0,
                            applicability_why="self_consistency",
                            accepted=vote["agree"],
                            abstained=vote["verifier_abstained"],
                            coverage_passed=True,
                            elapsed_ms=vote["elapsed_ms"],
                            note=vote["note"],
                        )
                    )
                    if vote["agree"] is False and not vote["verifier_abstained"]:
                        # Independent backend disagrees on a numeric answer.
                        # Decide whether to override using the primary's own
                        # backend witness (e.g. SymPy backwards-substitution
                        # has just confirmed every equation is consistent
                        # under the answer).
                        #
                        # When the primary carries a witness AND the
                        # verifier is the legacy pipeline, the verifier's
                        # ``formula_id`` tells us how strong its answer is:
                        #   * specialized topology adapter (RLC, series-
                        #     parallel network, multi-charge field) →
                        #     legacy has its own deterministic backend
                        #     for the topology; trust legacy.
                        #   * single-equation formula (direct_voltage_
                        #     source, power_p_vi, capacitor_charge_q_cv)
                        #     or LLM rescue → no topology backend; primary
                        #     witness wins.
                        result.trace.note(
                            f"self_consistency_disagree: "
                            f"primary={result.numeric_value!r} "
                            f"verifier={vote['verifier_value']!r}"
                        )
                        STRONG_LEGACY_ADAPTERS = {
                            "series_parallel_resistor_network",
                            "rlc_resonant_frequency",
                            "rlc_resonant_omega",
                            "multi_charge_field",
                            "multi_charge_force",
                            "circuit_nodal_analysis",
                            "wheatstone_bridge",
                            "thin_lens",
                            "spherical_mirror",
                            "doppler_observer_moving",
                            "doppler_source_moving",
                        }
                        verifier_formula_id = str(
                            vote.get("verifier_formula_id") or ""
                        )
                        verifier_strong = (
                            verifier_formula_id in STRONG_LEGACY_ADAPTERS
                        )
                        if (
                            getattr(result, "backend_verified", False)
                            and vote["verifier_method"] == "physics.legacy_pipeline"
                            and not verifier_strong
                        ):
                            result.trace.note(
                                "self_consistency_override: primary is "
                                "backend-verified; overruling unverified "
                                "verifier disagreement "
                                f"(verifier_formula={verifier_formula_id!r})"
                            )
                            # Slight confidence penalty so a tied later
                            # method with even stronger signals can still
                            # win, but accept the primary as the answer.
                            result.confidence = max(
                                0.0, float(result.confidence) - 0.05
                            )
                        else:
                            if (
                                best_downgrade is None
                                or float(result.confidence) > float(best_downgrade.confidence)
                            ):
                                best_downgrade = result
                            continue
                    if vote["agree"]:
                        result.confidence = min(0.95, result.confidence * 1.10)
                        result.trace.note(
                            f"self_consistency_pass: verifier={vote['verifier_method']}"
                        )

                outcome.final = result
                outcome.elapsed_ms = (time.perf_counter() - started) * 1000
                return outcome

            # Track a best-effort downgrade fallback.
            if (
                best_downgrade is None
                or float(result.confidence) > float(best_downgrade.confidence)
            ):
                best_downgrade = result

        # ---- nothing decisive: try discovery (Level 6) ---------------------
        if (
            allow_discovery
            and isinstance(problem, PhysicsProblem)
            and llm_client is not None
        ):
            outcome.discovery_attempted = True
            disc = discover_physics_method(
                problem, llm_client=llm_client, library=self._library
            )
            outcome.discovery_outcome = disc.why
            if disc.success and disc.method is not None:
                # Persist the library RIGHT AFTER successful registration so
                # the freshly-discovered method survives even if its first
                # solve attempt below abstains. The previous code only
                # persisted on a decisive run, leaking the discovery when
                # legacy fallback ended up answering instead.
                try:
                    self._library.persist()
                except Exception:
                    pass
                # Re-run the freshly registered method exactly once.
                t0 = time.perf_counter()
                try:
                    result = disc.method.solve(
                        problem, llm_client=llm_client, budget=budget
                    )
                except Exception as exc:
                    outcome.decisions.append(
                        PlannerDecision(
                            method_id=disc.method.method_id,
                            applicability_score=1.0,
                            applicability_why="freshly_discovered",
                            accepted=False,
                            abstained=False,
                            coverage_passed=False,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            note=f"discovered_method_error:{exc}",
                        )
                    )
                    result = None
                if result is not None:
                    outcome.decisions.append(
                        PlannerDecision(
                            method_id=disc.method.method_id,
                            applicability_score=1.0,
                            applicability_why="freshly_discovered",
                            accepted=result.decisive,
                            abstained=bool(result.abstained),
                            coverage_passed=True,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            note=None,
                        )
                    )
                    if result.decisive:
                        outcome.final = result
                        outcome.elapsed_ms = (time.perf_counter() - started) * 1000
                        # Persist the library so the discovered method survives
                        # future processes.
                        try:
                            self._library.persist()
                        except Exception:
                            pass
                        return outcome
                    if (
                        best_downgrade is None
                        or float(result.confidence) > float(best_downgrade.confidence)
                    ):
                        best_downgrade = result

        # ---- nothing decisive (logic): try logic-pattern discovery ---------
        # Mirror of the physics discovery block above. Triggered only when
        # the planner has shortlisted at least one logic method that
        # abstained — usually FOL+Z3 with atom-coverage drops on a novel
        # syntactic shape ("X unless Y", "X only if Y", etc.). One LLM call
        # proposes a regex+template rewrite; backend validates by re-running
        # FOL+Z3 on the rewrite. If decisive, the pattern is persisted to
        # ``models/logic_patterns.json`` so question N+1 reuses it without
        # another search. AGENTS.md §24 Phase F4 + §20.5 LLM-first logic.
        if (
            allow_discovery
            and isinstance(problem, LogicProblem)
            and llm_client is not None
            and outcome.final is None
        ):
            outcome.discovery_attempted = True
            disc = discover_logic_method(
                problem, llm_client=llm_client, library=self._library
            )
            # Compose a discovery_outcome that doesn't overwrite a prior
            # physics outcome (logic and physics paths are mutually
            # exclusive but we still want the audit string clean).
            outcome.discovery_outcome = (
                outcome.discovery_outcome + "|" + disc.why
                if outcome.discovery_outcome else disc.why
            )
            if disc.success and disc.method is not None:
                # Re-run the pattern-rewrite method exactly once with the
                # newly registered pattern in the store.
                t0 = time.perf_counter()
                try:
                    result = disc.method.solve(
                        problem, llm_client=llm_client, budget=budget
                    )
                except Exception as exc:
                    outcome.decisions.append(
                        PlannerDecision(
                            method_id=disc.method.method_id,
                            applicability_score=1.0,
                            applicability_why="freshly_discovered_logic_pattern",
                            accepted=False,
                            abstained=False,
                            coverage_passed=False,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            note=f"discovered_method_error:{exc}",
                        )
                    )
                    result = None
                if result is not None:
                    # Coverage gate again, since the planner's earlier shortlist
                    # invocation skipped it because that earlier pass did not
                    # have access to the freshly registered pattern.
                    coverage_passed = True
                    coverage_note: str | None = None
                    if result.decisive:
                        expected_ids = [
                            str(getattr(p, "id", ""))
                            for p in problem.normalized_premises
                            if getattr(p, "id", "")
                        ]
                        report = check_coverage(expected_ids, result)
                        coverage_passed = report.accepted
                        coverage_note = report.note
                        if not coverage_passed:
                            result = downgrade_to_abstain(result, report)
                    outcome.decisions.append(
                        PlannerDecision(
                            method_id=disc.method.method_id,
                            applicability_score=1.0,
                            applicability_why="freshly_discovered_logic_pattern",
                            accepted=result.decisive and coverage_passed,
                            abstained=bool(result.abstained),
                            coverage_passed=coverage_passed,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            note=coverage_note,
                        )
                    )
                    if result.decisive and coverage_passed:
                        outcome.final = result
                        outcome.elapsed_ms = (time.perf_counter() - started) * 1000
                        return outcome
                    if (
                        best_downgrade is None
                        or float(result.confidence) > float(best_downgrade.confidence)
                    ):
                        best_downgrade = result

        # ---- final: hand back the best downgrade, or nothing ---------------
        outcome.final = best_downgrade
        outcome.abstain_reason = (
            None
            if best_downgrade is not None and best_downgrade.answer
            else "all_methods_abstained"
        )
        outcome.elapsed_ms = (time.perf_counter() - started) * 1000
        return outcome
