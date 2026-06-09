"""Qualitative-question parser (Task 20.1, Req 14.1, 14.2).

Detects qualitative-question shape via general regex/keyword classes — never a
per-question text -> answer mapping (AGENTS.md §20.1, Req 13.1, 11.5). Returns
``QualitativeQuestion`` with:

- ``input_var``: canonical symbol of the changing input (``R``, ``V``, ``I``,
  ``C``, ``Q``, ``E``, ``F``, ``r``, ``L``, ``B``, ``P``, ``d``, ``eps_r``,
  ``N``, ``f``, ``E_field``, ``U``);
- ``change_direction`` in {+1, -1, 0};
- ``target_var``: canonical symbol of the asked-about quantity;
- ``other_vars_constant``: True iff an "all-else-constant" guard is present or
  another variable is explicitly held constant;
- ``competing_effects_detected``: True iff 2+ inputs change OR net direction
  is not fixed by phrasing (Req 14.2 abstain over guess).

Returns ``None`` for numeric questions (no qualitative phrasing detected) so
the existing IR/adapter path handles them unchanged (Req 14.6).

The solver opts in via the ``URA_ENABLE_QUALITATIVE_PARSER`` env flag (Task
20.4 wires the actual solve flow). For Task 20.1 we only:
1. expose ``parse_qualitative``,
2. populate ``ParsedPhysicsProblem.qualitative`` in ``parse_physics_question``,
3. expose ``qualitative_parser_enabled()`` so downstream tasks can guard on it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


__all__ = [
    "QualitativeQuestion",
    "parse_qualitative",
    "qualitative_parser_enabled",
]


# ---------------------------------------------------------------------------
# Phrasing classes — general regex/keyword rules (never per-question text).
# ---------------------------------------------------------------------------


# Words that, when applied to a variable, indicate that variable INCREASED.
_INCREASE_RE = re.compile(
    r"\b(?:"
    r"doubled|tripled|quadrupled|"
    r"double|triple|quadruple|"
    r"increased|increases|increasing|increase|"
    r"raised|raises|raising|raise|"
    r"grown|grew|grows|growing|grow|"
    r"risen|rises|rising|rise|"
    r"twice|"
    r"two\s+times|three\s+times|four\s+times|"
    r"factor\s+of\s+\d+(?:\.\d+)?|"
    r"multiplied\s+by\s+\d+(?:\.\d+)?|"
    r"scaled\s+up|"
    r"by\s+a\s+factor\s+of\s+\d+(?:\.\d+)?"
    r")\b",
    re.IGNORECASE,
)

# Words that indicate DECREASE.
_DECREASE_RE = re.compile(
    r"\b(?:"
    r"halved|halfed|halve|"
    r"decreased|decreases|decreasing|decrease|"
    r"reduced|reduces|reducing|reduce|"
    r"lowered|lowers|lowering|lower|"
    r"falls|fallen|falling|fall|"
    r"drops|dropped|dropping|drop|"
    r"becomes\s+half|"
    r"by\s+half"
    r")\b",
    re.IGNORECASE,
)

# Words that indicate UNCHANGED / held constant.
_UNCHANGED_RE = re.compile(
    r"\b(?:"
    r"unchanged|unaltered|"
    r"kept\s+(?:constant|fixed|the\s+same|unchanged)|"
    r"keeps?\s+(?:constant|fixed|the\s+same|unchanged)|"
    r"keeping\s+(?:.+?\s+)?(?:constant|fixed|the\s+same|unchanged)|"
    r"remains?\s+(?:constant|fixed|the\s+same|unchanged)|"
    r"held\s+(?:constant|fixed|the\s+same|unchanged)|"
    r"holding\s+(?:.+?\s+)?(?:constant|fixed|the\s+same|unchanged)|"
    r"maintain(?:ing|ed|s)?\s+(?:.+?\s+)?(?:constant|fixed|the\s+same|unchanged)|"
    r"is\s+kept\s+constant|are\s+kept\s+constant|"
    r"is\s+(?:constant|fixed|unchanged|the\s+same)|"
    r"are\s+(?:constant|fixed|unchanged|the\s+same)|"
    r"do(?:es)?\s+not\s+change|did\s+not\s+change|"
    r"no\s+change|"
    r"stays?\s+(?:the\s+same|constant|fixed|unchanged)"
    r")\b",
    re.IGNORECASE,
)


# Bare modifier "constant <var>" / "<var> constant" / "fixed <var>" — captured
# separately so the "<var> ... constant" pattern works as a leading-modifier
# unchanged signal even without a copula.
_BARE_CONSTANT_BEFORE_RE = re.compile(
    r"(?:with\s+|at\s+|under\s+)?"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(?:constant|fixed|unchanged|same|equal)\s*$",
    re.IGNORECASE,
)


# All-else-constant guards.
_ALL_ELSE_CONSTANT_RE = re.compile(
    r"\b(?:"
    r"with\s+all\s+else\s+(?:constant|fixed|the\s+same|unchanged)|"
    r"all\s+else\s+(?:held\s+)?(?:constant|fixed|unchanged|the\s+same)|"
    r"all\s+other(?:\s+(?:variables|quantities|things|values|inputs|parameters))?\s+"
    r"(?:held\s+|kept\s+)?(?:constant|fixed|unchanged|the\s+same)|"
    r"keeping\s+(?:the\s+rest|the\s+others|everything\s+else)\s+"
    r"(?:constant|fixed|the\s+same|unchanged)|"
    r"holding\s+(?:everything\s+else|all\s+others?)\s+"
    r"(?:constant|fixed|the\s+same|unchanged)"
    r")\b",
    re.IGNORECASE,
)


# Domain-level "implicit conservation" cues. These are general physics
# phrasings (NOT per-question text) that pin a particular variable to
# constant by physics convention:
#
#   * "disconnected from the battery / source / supply" or "isolated"
#     applied to a charged capacitor: charge Q is conserved while the
#     capacitor sits in isolation. In Q = C * V terms, Q is held constant,
#     so V is determined by C alone (V = Q/C) and E = Q^2/(2C).
#
# When such a cue is present, the corresponding canonical symbol is added
# to the parser's "explicitly held constant" set so the downstream reasoner
# can satisfy its all-else-constant precondition without an explicit "Q is
# constant" sentence in the question. The cue is a *phrasing* rule, never a
# question-text match (AGENTS.md §20.1, Req 13.1).
_DISCONNECTED_CAPACITOR_RE = re.compile(
    r"\b(?:"
    r"disconnect(?:ed|ing|s)?\s+(?:from|of)\s+(?:the\s+)?(?:battery|source|supply|circuit|emf|voltage\s+source)|"
    r"isolated\s+(?:from|capacitor)|"
    r"removed\s+from\s+(?:the\s+)?(?:battery|source|supply)|"
    r"detached\s+from\s+(?:the\s+)?(?:battery|source|supply)"
    r")\b",
    re.IGNORECASE,
)


# Variable phrase to canonical symbol map. Order matters: longer / more specific
# phrases first so "electric field energy" wins over "electric field" or "energy".
# These cover the vocabulary in `app/physics/formulas.py`: V, I, R, P, C, Q, E,
# F, r (distance), L, B, d (plate distance), eps_r, N (turns), f, E_field, U.
_VAR_PHRASES: tuple[tuple[str, str], ...] = (
    ("electric field energy", "E"),
    ("magnetic field energy", "E"),
    ("electrostatic energy", "E"),
    ("stored energy", "E"),
    ("electric energy", "E"),
    ("magnetic energy", "E"),
    ("potential energy", "U"),
    ("potential difference", "V"),
    ("electromotive force", "V"),
    ("electric field strength", "E_field"),
    ("electric field intensity", "E_field"),
    ("electric field", "E_field"),
    ("magnetic field", "B"),
    ("number of turns", "N"),
    ("distance between the plates", "d"),
    ("distance between the two plates", "d"),
    ("plate distance", "d"),
    ("plate separation", "d"),
    ("dielectric constant", "eps_r"),
    ("relative permittivity", "eps_r"),
    ("voltage", "V"),
    ("current", "I"),
    ("resistance", "R"),
    ("power", "P"),
    ("energy", "E"),
    ("capacitance", "C"),
    ("charge", "Q"),
    ("force", "F"),
    ("distance", "r"),
    ("separation", "r"),
    ("radius", "r"),
    ("frequency", "f"),
    ("inductance", "L"),
    ("emf", "V"),
)


_VAR_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p, _ in _VAR_PHRASES) + r")(?:s|es)?\b",
    re.IGNORECASE,
)


def _phrase_to_var(phrase: str) -> str | None:
    """Map a matched variable phrase (lowercased text, possibly with a trailing
    plural ``s``/``es``) to its canonical symbol."""
    p = phrase.strip().lower()
    for kw, var in _VAR_PHRASES:
        if kw == p:
            return var
    # Strip a trailing English plural marker and retry.
    for suffix in ("es", "s"):
        if p.endswith(suffix):
            stem = p[: -len(suffix)]
            for kw, var in _VAR_PHRASES:
                if kw == stem:
                    return var
    return None


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualitativeQuestion:
    """Parsed qualitative-question shape (Req 14.1, 14.2).

    The reasoner downstream (Task 20.4) consumes this to dispatch to the
    monotonic-relation registry and produce a canonical direction-of-change
    label, or to abstain to ``unknown`` when the abstain flags are set.
    """

    input_var: str
    """Canonical symbol of the changing input quantity (e.g. ``R``)."""

    change_direction: int
    """+1 increase, -1 decrease, 0 unchanged."""

    target_var: str
    """Canonical symbol of the asked-about quantity (e.g. ``P``)."""

    other_vars_constant: bool
    """True iff the question carries an all-else-constant guard or an explicit
    "X is kept constant" mention for a variable other than the input/target.
    When False, the downstream reasoner SHALL yield ``unknown`` (Req 14.2)."""

    competing_effects_detected: bool
    """True iff 2+ inputs change OR the net direction is not fixed by phrasing.
    The downstream reasoner SHALL yield ``unknown`` (Req 14.2)."""

    held_constant_vars: tuple[str, ...] = ()
    """Canonical symbols of variables explicitly mentioned as held constant
    (e.g. ``("V",)`` when "voltage is unchanged" is in the question, or
    ``("Q",)`` when the disconnected-capacitor cue is present). Empty when
    the all-else-constant guard is the only signal. Used by the reasoner to
    disambiguate between formulas that share (target, input) but differ in
    which "other" variable is the natural constant (e.g. P=V^2/R vs
    P=I^2*R both relate P to R; the right pick depends on whether V or I
    is held constant)."""

    change_factor: float | None = None
    """The numeric factor by which the input changes (e.g. 2.0 for doubled,
    0.5 for halved, 3.0 for tripled). None if only direction is known without
    specific factor (e.g. "increased" without specifying by how much).
    Used by the reasoner to compute exact output factor based on formula
    relationship (linear, squared, inverse, inverse squared)."""


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


# Target-phrase patterns: capture the noun phrase that names the asked-about
# quantity in standard "what happens to / how does ... change" phrasings. The
# captured group is intentionally permissive (any non-newline characters up to
# a sentinel) so that real-world surface forms like "D₂", "lamp X", "branch B12"
# don't break ASCII-letter character classes. The captured phrase is then
# scanned for a canonical variable name in `_VAR_PHRASES`.
_TARGET_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        # "what happens to <quantity>"
        r"what\s+happens\s+(?:to|with)\s+(?:the\s+)?"
        r"(.{1,120}?)"
        r"(?=\?|\.|,|;|\bif\b|\bwhen\b|\bafter\b|$)",
        # "how does/will/do <quantity> change/vary/compare/respond/behave"
        r"how\s+(?:does|will|do)\s+(?:the\s+)?"
        r"(.{1,120}?)"
        r"\s+(?:change|vary|compare|behave|differ|respond)\b",
        # "how many times does <quantity> change/increase/decrease"
        r"how\s+many\s+times\s+(?:does|will|do|the)\s+(?:the\s+)?"
        r"(.{1,120}?)"
        r"\s+(?:change|increase|decrease|grow|shrink)\b",
        # "how is <quantity> affected/altered/changed"
        r"how\s+is\s+(?:the\s+)?"
        r"(.{1,120}?)"
        r"\s+(?:affected|altered|changed)\b",
        # "find/calculate/determine the <quantity> after / when ..."
        r"(?:find|calculate|determine|compute)\s+(?:the\s+)?"
        r"(.{1,120}?)"
        r"\s+(?:after\s+(?:it\s+)?(?:is|has\s+been)|when\s+|if\s+)",
    )
)


def _detect_target_var(text: str) -> str | None:
    """Find target variable via general "what happens to / how does ... change" phrasing.

    Iterates the captured noun phrase, choosing the variable whose canonical
    phrase appears in it. Returns the canonical symbol or ``None``.
    """
    for pattern in _TARGET_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        phrase = m.group(1).strip().lower()
        # Prefer the longest matching variable phrase inside the captured noun.
        best: tuple[int, str] | None = None
        for kw, var in _VAR_PHRASES:
            if re.search(rf"\b{re.escape(kw)}\b", phrase):
                if best is None or len(kw) > best[0]:
                    best = (len(kw), var)
        if best is not None:
            return best[1]
    return None


# Imperative leading-verb patterns: "double X", "halving X", "increase the X".
_LEADING_INCREASE = re.compile(
    r"^(?:doubl|tripl|quadrupl|increas|rais|grow|ris|expand|amplif)\w*$",
    re.IGNORECASE,
)
_LEADING_DECREASE = re.compile(
    r"^(?:halv|halv|halfed|decreas|reduc|lower|fall|drop|shrink|diminish)\w*$",
    re.IGNORECASE,
)
_LEADING_VERB_RE = re.compile(
    r"\b(\w+)\s+(?:the\s+|its\s+|their\s+|a\s+|an\s+)?$",
    re.IGNORECASE,
)


def _direction_near(text_low: str, start: int, end: int) -> int | None:
    """Determine the direction of change attached to a variable mention.

    Strategy chosen after considering proximity-scan failure modes (a stray
    direction token leaking from a previous clause to the next variable):

    0. First check if this variable is inside a prepositional phrase that
       describes something else (e.g. "from a point charge", "between two
       charges"). These are descriptive noun phrases, not subjects being
       changed. Return None to skip attaching any direction.

    1. Tight after-scan up to the next clause boundary (comma, semicolon,
       period, or a clause connective like "if / when / while / although /
       and / but"), capped at 40 chars. Looks for trailing-verb forms like
       "X is doubled / X are halved / X increases / X remains constant".
       Direction precedence: unchanged > increase > decrease so an explicit
       "is unchanged" never gets shadowed by a stray increase token.

    2. Only if (1) finds nothing, do a tight leading-modifier before-scan
       (~30 chars, also clause-bounded) for:
       - "constant / fixed / unchanged X" (sets direction 0), or
       - imperative phrasings: "double X / halving X / increase the X".

    Returns +1 / -1 / 0 / None.
    """
    # Step 0: Check if variable is inside a prepositional phrase (descriptive, not subject)
    # Patterns like "from a point charge", "between two charges", "of the charge"
    # indicate the variable is part of a description, not the thing being changed.
    before_full = text_low[max(0, start - 60) : start]
    before = _truncate_at_clause_boundary_reverse(before_full)[-30:]
    
    # Prepositional phrases that indicate descriptive context (not subject of change)
    # Match: "from a point ", "from the ", "between two ", "between the ", "of the ", "of a "
    prep_phrase_re = re.compile(
        r"\b(?:from|between|of|on|at|in|by|with|near|around)\s+"
        r"(?:a|an|the|two|three|four|some|these|those|this|that)?\s*"
        r"(?:point|test|small|large|single|isolated|positive|negative)?\s*$",
        re.IGNORECASE
    )
    if prep_phrase_re.search(before):
        # This variable is likely part of a prepositional phrase describing something
        # e.g., "distance from a point [charge]" - charge here is descriptive
        return None
    
    after_full = text_low[end : end + 60]
    after = _truncate_at_clause_boundary(after_full)[:40]
    if _UNCHANGED_RE.search(after):
        return 0
    if _INCREASE_RE.search(after):
        return 1
    if _DECREASE_RE.search(after):
        return -1

    # Leading bare modifier: "constant capacitance", "fixed voltage", etc.
    if _BARE_CONSTANT_BEFORE_RE.search(before):
        return 0

    # Imperative leading verb: "double X", "halving X", "increase the X".
    m_lead = _LEADING_VERB_RE.search(before)
    if m_lead:
        verb = m_lead.group(1)
        if _LEADING_INCREASE.match(verb):
            return 1
        if _LEADING_DECREASE.match(verb):
            return -1
    return None


# Clause-boundary tokens that terminate the proximity window so a direction
# token from a different clause cannot bind to this variable.
_CLAUSE_BOUNDARY_AFTER_RE = re.compile(
    r"(?:[,;.?!]|\band\b|\bbut\b|\bif\b|\bwhen\b|\bwhile\b|\bthough\b|\balthough\b|\bwhereas\b|\bgiven\b|\bsince\b)",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_BEFORE_RE = re.compile(
    r"(?:[,;.?!]|\band\b|\bbut\b|\bif\b|\bwhen\b|\bwhile\b|\bthough\b|\balthough\b|\bwhereas\b|\bgiven\b|\bsince\b)",
    re.IGNORECASE,
)


def _truncate_at_clause_boundary(after_text: str) -> str:
    """Cut ``after_text`` at the first clause boundary so we don't leak into
    the next clause."""
    m = _CLAUSE_BOUNDARY_AFTER_RE.search(after_text)
    if m is None:
        return after_text
    return after_text[: m.start()]


def _truncate_at_clause_boundary_reverse(before_text: str) -> str:
    """Cut ``before_text`` at the LAST clause boundary so we don't leak from
    the previous clause."""
    last = -1
    for m in _CLAUSE_BOUNDARY_BEFORE_RE.finditer(before_text):
        last = m.end()
    if last < 0:
        return before_text
    return before_text[last:]


def _detect_changes(text: str) -> list[tuple[str, int, int, float | None]]:
    """Scan for ``<variable phrase>`` occurrences with a nearby direction token.

    Returns a list of ``(canonical_var, direction, start_position, factor)``,
    deduplicated by ``(var, direction)`` keeping the earliest occurrence so
    e.g. "voltage is doubled ... voltage is doubled" contributes once.
    
    The factor is the numeric multiplier (e.g. 2.0 for doubled, 0.5 for halved,
    3.0 for tripled) or None if only direction is known.
    """
    low = text.lower()
    found: list[tuple[str, int, int, float | None]] = []
    for m in _VAR_PHRASE_RE.finditer(low):
        phrase = m.group(1)
        var = _phrase_to_var(phrase)
        if var is None:
            continue
        d = _direction_near(low, m.start(), m.end())
        if d is None:
            continue
        # Extract factor from nearby text
        factor = _extract_change_factor(low, m.start(), m.end())
        found.append((var, d, m.start(), factor))
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int, int, float | None]] = []
    for var, direction, pos, factor in sorted(found, key=lambda x: x[2]):
        key = (var, direction)
        if key in seen:
            continue
        seen.add(key)
        out.append((var, direction, pos, factor))
    return out


# Factor extraction patterns
_FACTOR_PATTERNS: tuple[tuple[re.Pattern, float | None], ...] = (
    # Exact named factors
    (re.compile(r"\bdoubled\b", re.I), 2.0),
    (re.compile(r"\btwice\b", re.I), 2.0),
    (re.compile(r"\btripled\b", re.I), 3.0),
    (re.compile(r"\bquadrupled\b", re.I), 4.0),
    (re.compile(r"\bhalved\b", re.I), 0.5),
    (re.compile(r"\bhalfed\b", re.I), 0.5),  # common misspelling
    (re.compile(r"\bbecomes?\s+half\b", re.I), 0.5),
    (re.compile(r"\bby\s+half\b", re.I), 0.5),
    # "N times" patterns
    (re.compile(r"\btwo\s+times\b", re.I), 2.0),
    (re.compile(r"\bthree\s+times\b", re.I), 3.0),
    (re.compile(r"\bfour\s+times\b", re.I), 4.0),
    (re.compile(r"\bfive\s+times\b", re.I), 5.0),
    (re.compile(r"\bten\s+times\b", re.I), 10.0),
    # Numeric factor patterns - these return None as sentinel to extract the number
    (re.compile(r"\bby\s+a\s+factor\s+of\s+(\d+(?:\.\d+)?)\b", re.I), None),
    (re.compile(r"\bmultiplied\s+by\s+(\d+(?:\.\d+)?)\b", re.I), None),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s+times\b", re.I), None),
)


def _extract_change_factor(text_low: str, start: int, end: int) -> float | None:
    """Extract the numeric factor of change from text near a variable mention.
    
    Returns the factor (e.g. 2.0 for doubled, 0.5 for halved) or None if
    only direction is known without a specific factor.
    """
    # Check in window around the variable mention
    window_start = max(0, start - 40)
    window_end = min(len(text_low), end + 60)
    window = text_low[window_start:window_end]
    
    for pattern, fixed_factor in _FACTOR_PATTERNS:
        m = pattern.search(window)
        if m:
            if fixed_factor is not None:
                return fixed_factor
            # Pattern has a capture group for the numeric value
            try:
                return float(m.group(1))
            except (IndexError, ValueError):
                continue
    
    return None


def _has_all_else_constant_guard(text: str) -> bool:
    """Checks if an all-else-constant guard is present in the text."""
    return bool(_ALL_ELSE_CONSTANT_RE.search(text))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_qualitative(question: str) -> QualitativeQuestion | None:
    """Detect qualitative-question shape from general phrasing rules.

    Returns ``None`` when:
    - the question is empty,
    - no "what happens to X / how does X change" target phrasing is present
      that resolves to a known canonical variable, or
    - no change-direction phrasing is attached to a recognized variable.

    These cases are numeric (or out of scope), so the existing IR/adapter
    path handles them unchanged (Req 14.6).

    When change-direction phrasing IS detected:
    - 2+ distinct inputs change ⇒ ``competing_effects_detected=True``,
    - target_var would equal input_var (no actual input change) ⇒ ``None``,
    - no all-else-constant guard and no explicit "X is kept constant" for a
      variable other than the input/target ⇒ ``other_vars_constant=False``.

    In both abstain cases above the downstream reasoner SHALL yield
    ``unknown`` (Req 14.2).

    Args:
        question: The raw physics problem question text.

    Returns:
        A QualitativeQuestion instance if qualitative patterns are found, or None.
    """
    if not question or not question.strip():
        return None

    target_var = _detect_target_var(question)
    if target_var is None:
        return None

    changes = _detect_changes(question)
    if not changes:
        return None

    actual_inputs: list[tuple[str, int, float | None]] = []
    explicit_constants: list[str] = []
    for var, direction, _pos, factor in changes:
        if direction == 0:
            explicit_constants.append(var)
        elif var != target_var:
            actual_inputs.append((var, direction, factor))
        # If var == target_var with non-zero direction, ignore — that's the
        # target's stated change (e.g. "the energy increases"), not an input
        # change driving the target.

    if not actual_inputs:
        return None

    # Conservation-cue handling: if the question mentions a capacitor being
    # disconnected/isolated from a source, charge Q is conserved while the
    # capacitor sits isolated. Treat 'Q' as an explicitly held-constant
    # variable so the downstream all-else-constant guard is satisfied.
    # This is a *general* physics-phrasing rule, never a per-question
    # text match (AGENTS.md §20.1, Req 13.1).
    if _DISCONNECTED_CAPACITOR_RE.search(question) and "capacit" in question.lower():
        if "Q" not in explicit_constants:
            explicit_constants.append("Q")

    # 2+ distinct input changes ⇒ competing effects (Req 14.2).
    if len(actual_inputs) >= 2:
        first_var, first_dir, first_factor = actual_inputs[0]
        return QualitativeQuestion(
            input_var=first_var,
            change_direction=first_dir,
            target_var=target_var,
            other_vars_constant=False,
            competing_effects_detected=True,
            held_constant_vars=tuple(explicit_constants),
            change_factor=first_factor,
        )

    input_var, direction, change_factor = actual_inputs[0]

    has_guard = _has_all_else_constant_guard(question)
    other_constant_explicit = any(
        v not in {input_var, target_var} for v in explicit_constants
    )
    other_vars_constant = bool(has_guard or other_constant_explicit)

    return QualitativeQuestion(
        input_var=input_var,
        change_direction=direction,
        target_var=target_var,
        other_vars_constant=other_vars_constant,
        competing_effects_detected=False,
        held_constant_vars=tuple(explicit_constants),
        change_factor=change_factor,
    )


def qualitative_parser_enabled() -> bool:
    """Feature flag checking if the qualitative parser feature is enabled.

    Returns:
        True if URA_ENABLE_QUALITATIVE_PARSER env var is set to '1', 'true', 'yes', or 'on'.
    """
    raw = os.environ.get("URA_ENABLE_QUALITATIVE_PARSER", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}

