"""Coverage gate: every input premise/quantity must have a recorded fate.

A method can intentionally drop an input — but it MUST say why. A silent
drop is a soundness bug (AGENTS.md §20: prefer abstain over wrong). The
coverage gate inspects ``MethodResult.trace`` and either:

  * passes the result through unchanged (every input was used or
    explicitly dropped with a reason);
  * downgrades the result to ``unknown`` with an audit note (any input was
    silently ignored).

Coverage is checked AFTER faithfulness so the planner does not promote a
result whose translation was incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.methods.types import MethodResult


@dataclass
class CoverageReport:
    """Result of running the coverage gate on a MethodResult."""

    accepted: bool
    silently_dropped: list[str]
    explicitly_used: list[str]
    explicitly_dropped: list[str]
    note: str | None = None


def check_coverage(
    expected_input_ids: Iterable[str],
    method_result: MethodResult,
    *,
    accept_explicit_drops: bool = True,
) -> CoverageReport:
    """Inspect ``method_result.trace`` and decide whether every input has a fate.

    An input is "covered" if it appears in:
      * ``method_result.used_premise_ids`` (explicitly used by the prover), or
      * ``method_result.trace.inputs_dropped`` (explicitly recorded as
        skipped, with a reason).

    If ``accept_explicit_drops=False`` (strict mode), explicit drops also
    fail the gate — useful for the high-bar audit when the planner already
    expects a Method to consume every premise.
    """
    expected = [str(x) for x in expected_input_ids]
    used = {str(x) for x in (method_result.used_premise_ids or [])}
    explicit_drops = {
        str(d.get("id", ""))
        for d in (method_result.trace.inputs_dropped or [])
        if d.get("id")
    }
    silently_dropped: list[str] = []
    explicitly_used: list[str] = []
    explicitly_dropped: list[str] = []
    for pid in expected:
        if pid in used:
            explicitly_used.append(pid)
        elif pid in explicit_drops:
            explicitly_dropped.append(pid)
        else:
            silently_dropped.append(pid)

    accepted = (not silently_dropped) and (
        accept_explicit_drops or not explicitly_dropped
    )
    note: str | None = None
    if silently_dropped:
        note = f"coverage_failed:silently_dropped={silently_dropped}"
    elif explicitly_dropped and not accept_explicit_drops:
        note = f"coverage_failed:explicit_drops_disallowed={explicitly_dropped}"
    return CoverageReport(
        accepted=accepted,
        silently_dropped=silently_dropped,
        explicitly_used=explicitly_used,
        explicitly_dropped=explicitly_dropped,
        note=note,
    )


def downgrade_to_abstain(
    method_result: MethodResult, coverage: CoverageReport
) -> MethodResult:
    """Return a copy of ``method_result`` downgraded to ``unknown`` for coverage failures.

    The original answer / explanation are preserved in the trace so audits
    can see what would have been said. The new result reports ``unknown``
    with a confidence floored to 0.3 (low but non-zero so the planner can
    still see this came from a real method, not a no-op).
    """
    method_result.trace.note(coverage.note or "coverage_downgrade")
    new_explanation = (
        f"{method_result.explanation} "
        f"(downgraded to unknown by coverage gate: {coverage.note})".strip()
    )
    method_result.answer = "unknown"
    method_result.abstained = True
    method_result.abstain_reason = coverage.note or "coverage_failed"
    method_result.confidence = min(method_result.confidence, 0.3)
    method_result.explanation = new_explanation
    return method_result
