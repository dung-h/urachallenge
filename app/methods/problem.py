"""Typed problem bundles passed into Method.score_match / Method.solve.

Methods do not see raw request strings; they see a structured ``LogicProblem``
or ``PhysicsProblem`` IR. Two reasons:

  1. Decouples Method authors from FastAPI request shape.
  2. Keeps ``score_match`` deterministic: applicability decisions are made
     against structural features (premise count, target unit hint, presence
     of MCQ options, ...) rather than by inspecting the raw question text
     keyword by keyword (AGENTS.md §20.1).

The IR builders here are thin: they collect what already-existing parsers
produce (``app.logic.premise_selector.normalize_premises``,
``app.physics.parser.parse_problem``) and bundle them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogicProblem:
    """Structural bundle of a logic question for Method dispatch.

    Fields are STRUCTURAL features. ``raw_question`` and ``raw_premises`` are
    kept verbatim only so deterministic methods that already consume the raw
    text (BFS, FOL translator) can reach them; ``score_match`` should NOT
    inspect them character by character.
    """

    raw_question: str
    raw_premises: list[str]
    normalized_premises: list[Any]   # list[Premise]
    choices: list[str] = field(default_factory=list)
    has_mcq_choices: bool = False
    has_negation_marker: bool = False     # any premise/question contains "not", "no", "unless"
    has_unless_marker: bool = False
    has_conditional_marker: bool = False  # "if ... then" / "all ... are"
    has_quantifier_marker: bool = False   # "all", "every", "some", "no"
    has_comparison_marker: bool = False   # "taller", "older", "more than"
    premises_fol: list[str] | None = None
    answer_kind: str | None = None        # "yesno" / "mcq" / "open"

    @property
    def premise_count(self) -> int:
        return len(self.normalized_premises)


@dataclass
class PhysicsProblem:
    """Structural bundle of a physics question for Method dispatch."""

    raw_question: str
    parsed: Any                    # ParsedPhysicsProblem
    target_quantity: str | None = None
    target_unit_hint: str | None = None
    quantity_count: int = 0
    has_units: bool = False
    domain_hints: list[str] = field(default_factory=list)  # mechanics / circuits / em / optics ...

    @property
    def is_lookup_question(self) -> bool:
        """True for definitional / unit-name lookups (no numeric target)."""
        # Mirror the check in retrieval_grounded_method, kept here to avoid an
        # import cycle. Conservative: only fires on explicit "what is the unit
        # of / called" phrasings AND when the question carries no digits.
        import re
        low = self.raw_question.lower()
        if re.search(r"\d", low):
            return False
        patterns = (
            r"\bwhat\s+is\s+the\s+(?:si\s+)?unit\s+of\b",
            r"\bunit\s+of\s+measure(?:ment)?\s+(?:of|for)\b",
            r"\bwhat\s+is\s+the\s+(?:si\s+)?unit\s+for\b",
            r"\bwhat\s+is\s+.+\s+measured\s+in\b",
            r"\bwhat\s+is\s+.+\s+called\b",
            r"\bwhat\s+do\s+we\s+call\b",
            r"\bname\s+the\s+(?:si\s+)?unit\b",
            r"\bwhat\s+is\s+the\s+name\s+of\b",
        )
        return any(re.search(p, low) for p in patterns)


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def build_logic_problem(
    question: str,
    premises: list[str],
    choices: list[str] | None = None,
    premises_fol: list[str] | None = None,
    answer_kind: str | None = None,
) -> LogicProblem:
    """Build a ``LogicProblem`` from raw inputs.

    Reuses ``app.logic.premise_selector.normalize_premises`` so premise IDs
    and parsed flags are consistent with the rest of the logic stack.
    """
    from app.logic.premise_selector import normalize_premises

    normalized = normalize_premises(premises)
    blob = (question + " " + " ".join(premises)).lower()
    return LogicProblem(
        raw_question=question,
        raw_premises=list(premises),
        normalized_premises=normalized,
        choices=list(choices or []),
        has_mcq_choices=bool(choices),
        has_negation_marker=any(tok in blob for tok in (" not ", " no ", " never ", " unless ")),
        has_unless_marker=" unless " in blob or "unless " in blob,
        has_conditional_marker=("if " in blob and " then " in blob) or " all " in blob or " every " in blob,
        has_quantifier_marker=any(tok in blob for tok in (" all ", " every ", " some ", " no ", " each ")),
        has_comparison_marker=any(tok in blob for tok in (" than ", "tallest", "oldest", "youngest", "shortest", "biggest", "smallest")),
        premises_fol=list(premises_fol) if premises_fol else None,
        answer_kind=answer_kind,
    )


def build_physics_problem(question: str, parsed: Any) -> PhysicsProblem:
    """Build a ``PhysicsProblem`` from a parser output."""
    target = getattr(parsed, "target_quantity", None)
    quantities = list(getattr(parsed, "quantities", []) or [])
    domain_hints: list[str] = []
    low = question.lower()
    if any(tok in low for tok in ("ohm", "voltage", "current", "resistor", "capacit", "inductor", "circuit")):
        domain_hints.append("circuit")
    if any(tok in low for tok in ("force", "mass", "acceleration", "velocity", "momentum", "kinetic")):
        domain_hints.append("mechanics")
    if any(tok in low for tok in ("electric field", "coulomb", "charge", "permittivity")):
        domain_hints.append("electrostatics")
    if any(tok in low for tok in ("lens", "mirror", "refract", "wavelength", "wavelen")):
        domain_hints.append("optics")
    if any(tok in low for tok in ("temperature", "kelvin", "joule", "heat")):
        domain_hints.append("thermal")
    if any(tok in low for tok in ("density", "pressure", "fluid", "buoyan")):
        domain_hints.append("fluids")
    if any(tok in low for tok in ("pendulum", "oscillat", "period", "frequency", "wave", "vibrat")):
        domain_hints.append("oscillation")
    if any(tok in low for tok in ("rotat", "torque", "angular", "moment of inertia")):
        domain_hints.append("rotation")
    if any(tok in low for tok in ("magnetic", "solenoid", "flux", " coil")):
        domain_hints.append("magnetism")
    if any(tok in low for tok in ("doppler", "sound", "decibel", "hz")):
        domain_hints.append("acoustics")
    return PhysicsProblem(
        raw_question=question,
        parsed=parsed,
        target_quantity=target,
        quantity_count=len(quantities),
        has_units=any(getattr(q, "si_unit", "") for q in quantities),
        domain_hints=domain_hints,
    )
