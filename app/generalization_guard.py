"""Generalization guard — runtime policy for Req 13.4 / Req 13.5 (Task 17.2).

This module is **policy enforcement**, not a code-path decision. It does not
look at any specific question text. It encodes the rule from
``.kiro/specs/exact-challenge-optimization/requirements.md`` Requirement 13:

    13.4  IF a corrected capability does not produce the correct answer on all 3
          of its structurally similar verification cases, THEN THE System SHALL
          return ``unknown`` for the affected cases rather than encoding an
          instance-specific answer.
    13.5  IF fewer than 3 structurally similar cases can be constructed to
          verify a correction, THEN THE System SHALL treat the correction as
          non-generalizing and return ``unknown`` for the affected cases rather
          than encoding an instance-specific answer.

The contract is that every capability the solvers were corrected to support is
registered here together with a citation of where its >=3 structurally-distinct
verification cases live. An audit test
(``tests/test_generalization_guard.py``) walks the registry, confirms the
citation files exist, counts the verification cases, and fails if any
capability has fewer than 3 (Req 13.5) or has been declared
``unverified``/``insufficient`` without the solver wiring it up to abstain
(Req 13.4).

The module also exposes :func:`requires_abstention` and :func:`enforce` helpers
so a solver can ask "is this capability sufficiently generalized to accept its
verdict, or must I downgrade to ``unknown``?" at the *capability level*. Solvers
never query this with question-specific text — only with the capability_id of
the component they are about to invoke. This is exactly the abstention
discipline AGENTS.md §20 mandates.

Importantly, this module:

* **Never** stores question text, gold answers, or per-instance overrides.
* **Never** decides an answer; it only decides whether to *withhold* one.
* Records a generalization-evidence count and a citation per capability, so
  the audit test can prove evidence ≥3 is on disk, not just claimed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

# The minimum number of structurally-distinct cases a capability MUST be
# verified on before its verdict is acceptable (Req 13.3). Below this bar the
# capability is non-generalizing and the System SHALL abstain (Req 13.5).
MIN_GENERALIZATION_CASES: int = 3


class GeneralizationStatus(str, Enum):
    """Outcome of the generalization audit for a capability.

    * ``SUFFICIENT``  — verification on >=3 structurally-distinct cases is on
      disk and passing. The solver may emit a definite verdict for this
      capability, subject to the verifier acceptance boundary (Req 3).
    * ``INSUFFICIENT`` — fewer than 3 structurally-similar cases are
      constructible OR the verification cases are not all passing. The
      affected cases SHALL return ``unknown`` (Req 13.4, Req 13.5).
    """

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class CapabilityEvidence:
    """Generalization-evidence record for a corrected solver capability.

    ``capability_id``  — stable identifier the solver code references, e.g.
                       ``"logic.multi_hop_universal_chain"``. Never a question
                       string.
    ``description``    — one-line summary of the structural reasoning the
                       capability implements.
    ``test_files``     — citation: relative paths (POSIX style) of the test
                       modules whose >=3 cases verify the capability. The audit
                       test confirms every path exists.
    ``case_count``     — claimed count of structurally-distinct verification
                       cases on disk. The audit test independently verifies the
                       claim (so the registry cannot drift from reality).
    ``status``         — the policy verdict. ``SUFFICIENT`` permits a definite
                       verdict; ``INSUFFICIENT`` forces the System to abstain
                       on cases that depend on this capability.
    ``requirement_refs`` — requirement IDs the capability was added to satisfy
                       (informational; the audit checks formatting only).
    """

    capability_id: str
    description: str
    test_files: tuple[str, ...]
    case_count: int
    status: GeneralizationStatus = GeneralizationStatus.SUFFICIENT
    requirement_refs: tuple[str, ...] = field(default_factory=tuple)

    def is_sufficient(self) -> bool:
        """True iff the capability has the evidence needed to accept verdicts.

        The two policies of Req 13.4 / Req 13.5 are encoded together: the
        capability is only ``sufficient`` when the registry status says so AND
        the case count meets the minimum bar. A capability marked
        ``SUFFICIENT`` but with ``case_count < MIN_GENERALIZATION_CASES`` is
        rejected as misconfigured rather than silently accepted.
        """

        if self.status is not GeneralizationStatus.SUFFICIENT:
            return False
        return self.case_count >= MIN_GENERALIZATION_CASES


# ---------------------------------------------------------------------------
# Registry — every corrected capability the solvers rely on for accuracy.
# ---------------------------------------------------------------------------
#
# Each entry cites the test module(s) implementing its >=3 structurally-
# distinct cases (Req 13.3). The audit test in
# ``tests/test_generalization_guard.py`` opens those files, confirms they
# exist, and counts the cases. If any of the cited tests fail in CI, the
# capability is non-generalizing and the SHALL-abstain rule of Req 13.4 fires.
#
# A capability is added here ONLY when:
#   1) the solver carries a corresponding component-level fix (Req 13.2), and
#   2) the cited tests document >=3 structurally-distinct cases (Req 13.3).
#
# An entry is set to ``INSUFFICIENT`` when fewer than 3 cases can be
# constructed (Req 13.5) — the test in `tests/test_generalization_evidence_
# insufficiency.py` exercises this path explicitly.

_CAPABILITIES: dict[str, CapabilityEvidence] = {
    # --- Logic capabilities (Task 7) -------------------------------------- #
    "logic.multi_hop_universal_chain": CapabilityEvidence(
        capability_id="logic.multi_hop_universal_chain",
        description=(
            "Multi-hop universal forward chaining (>=5 hops) via the "
            "deterministic FOL compiler and Z3 entailment."
        ),
        test_files=("tests/test_logic_capability_generalization.py",),
        case_count=3,
        requirement_refs=("4.1", "4.7", "13.3"),
    ),
    "logic.negation_scope_consequent": CapabilityEvidence(
        capability_id="logic.negation_scope_consequent",
        description=(
            "Negation is scoped to the consequent term, never to the "
            "antecedent or whole implication."
        ),
        test_files=("tests/test_logic_capability_generalization.py",),
        case_count=3,
        requirement_refs=("4.2", "4.7", "13.3"),
    ),
    "logic.contrapositive_direction": CapabilityEvidence(
        capability_id="logic.contrapositive_direction",
        description=(
            "Direction preservation for A->B: no converse, no inverse; only "
            "the logically valid contrapositive ¬B->¬A is derivable."
        ),
        test_files=("tests/test_logic_capability_generalization.py",),
        case_count=3,
        requirement_refs=("4.3", "4.7", "13.3"),
    ),
    "logic.necessary_vs_sufficient": CapabilityEvidence(
        capability_id="logic.necessary_vs_sufficient",
        description=(
            "Necessary-only premises ('requires', 'only if') do not entail a "
            "positive conclusion; the answer abstains to ``unknown``."
        ),
        test_files=("tests/test_logic_capability_generalization.py",),
        case_count=3,
        requirement_refs=("4.4", "4.7", "13.3"),
    ),
    "logic.mcq_existential_gate": CapabilityEvidence(
        capability_id="logic.mcq_existential_gate",
        description=(
            "MCQ existential options are selected only when a matching "
            "existential premise entails them."
        ),
        test_files=("tests/test_logic_capability_generalization.py",),
        case_count=3,
        requirement_refs=("4.5", "4.7", "13.3"),
    ),
    # --- Physics capabilities (Task 9) ------------------------------------ #
    "physics.multi_charge_triangle": CapabilityEvidence(
        capability_id="physics.multi_charge_triangle",
        description=(
            "Multi-charge electric-field/force at a triangle vertex via "
            "deterministic vector summation."
        ),
        test_files=("tests/test_physics_multi_charge_generalization.py",),
        case_count=3,
        requirement_refs=("5.1", "5.2", "13.3"),
    ),
    "physics.multi_charge_square": CapabilityEvidence(
        capability_id="physics.multi_charge_square",
        description=(
            "Multi-charge electric field at the fourth vertex of a square "
            "via deterministic vector summation."
        ),
        test_files=("tests/test_physics_multi_charge_generalization.py",),
        case_count=3,
        requirement_refs=("5.1", "5.2", "13.3"),
    ),
    "physics.multi_charge_rectangle": CapabilityEvidence(
        capability_id="physics.multi_charge_rectangle",
        description=(
            "Multi-charge force at a rectangle vertex via deterministic "
            "vector summation with both AB and AD sides."
        ),
        test_files=("tests/test_physics_multi_charge_generalization.py",),
        case_count=3,
        requirement_refs=("5.1", "5.2", "13.3"),
    ),
    "physics.multi_charge_collinear": CapabilityEvidence(
        capability_id="physics.multi_charge_collinear",
        description=(
            "Multi-charge force on a collinear (opposite-sides) configuration "
            "via deterministic vector summation."
        ),
        test_files=("tests/test_physics_multi_charge_generalization.py",),
        case_count=3,
        requirement_refs=("5.1", "5.2", "13.3"),
    ),
    "physics.multi_charge_midpoint": CapabilityEvidence(
        capability_id="physics.multi_charge_midpoint",
        description=(
            "Multi-charge force at the midpoint of a two-charge segment via "
            "deterministic vector summation."
        ),
        test_files=("tests/test_physics_multi_charge_generalization.py",),
        case_count=3,
        requirement_refs=("5.1", "5.2", "13.3"),
    ),
}


def registered_capabilities() -> Mapping[str, CapabilityEvidence]:
    """Return a read-only view of every registered capability.

    Returns:
        A dictionary mapping capability IDs to CapabilityEvidence records.
    """

    return dict(_CAPABILITIES)


def get_capability(capability_id: str) -> CapabilityEvidence | None:
    """Return the registered :class:`CapabilityEvidence` or ``None``.

    Args:
        capability_id: Stable identifier of the capability.

    Returns:
        The CapabilityEvidence record if found, otherwise None.
    """

    return _CAPABILITIES.get(capability_id)


def requires_abstention(capability_id: str) -> bool:
    """Return True when the System SHALL return ``unknown`` for the capability.

    Args:
        capability_id: Stable identifier of the capability.

    Returns:
        True if the System must abstain, False otherwise.
    """

    record = _CAPABILITIES.get(capability_id)
    if record is None:
        return True
    return not record.is_sufficient()


def enforce(capability_id: str, answer: str) -> str:
    """Return ``answer`` unchanged when the capability has sufficient evidence.

    Args:
        capability_id: Stable identifier of the capability.
        answer: Raw answer generated by the solver.

    Returns:
        The original answer or "unknown" if abstention is forced.
    """

    if requires_abstention(capability_id):
        return "unknown"
    return answer


def register_for_test(record: CapabilityEvidence) -> None:
    """Test-only hook to register a synthetic capability.

    Args:
        record: Synthetic CapabilityEvidence to register.
    """

    _CAPABILITIES[record.capability_id] = record


def unregister_for_test(capability_id: str) -> None:
    """Test-only hook to remove a synthetic capability registered above.

    Args:
        capability_id: Stable identifier of the capability.
    """

    _CAPABILITIES.pop(capability_id, None)


def repo_root() -> Path:
    """Locate the repository root for resolving citation paths.

    Returns:
        Path object representing the repository root.
    """

    return Path(__file__).resolve().parents[1]


def resolve_citation(rel_path: str) -> Path:
    """Resolve a registry citation path to an absolute :class:`Path`.

    Args:
        rel_path: Relative POSIX path to citation file.

    Returns:
        Absolute Path to resolved citation file.
    """

    return repo_root() / rel_path


__all__ = [
    "MIN_GENERALIZATION_CASES",
    "GeneralizationStatus",
    "CapabilityEvidence",
    "registered_capabilities",
    "get_capability",
    "requires_abstention",
    "enforce",
    "register_for_test",
    "unregister_for_test",
    "resolve_citation",
    "repo_root",
]


def _audit_invariants(records: Iterable[CapabilityEvidence]) -> None:
    """Internal sanity check run at import time to catch obvious drift."""

    for record in records:
        if record.status is GeneralizationStatus.SUFFICIENT:
            assert record.case_count >= MIN_GENERALIZATION_CASES, (
                f"capability {record.capability_id!r} is marked SUFFICIENT but "
                f"only claims {record.case_count} cases (minimum {MIN_GENERALIZATION_CASES})."
            )
        assert record.test_files, (
            f"capability {record.capability_id!r} has no citation test_files."
        )


_audit_invariants(_CAPABILITIES.values())
