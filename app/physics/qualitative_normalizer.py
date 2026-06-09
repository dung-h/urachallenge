"""Canonical-label normalizer for qualitative physics answers (Task 20.3, Req 14.3).

Single source of truth for mapping a free-text qualitative answer phrase to a
canonical label drawn from a small closed set. The mapping is a **general
phrasing-equivalence rule set** β€” it is NEVER a per-question text -> answer
mapping (AGENTS.md Β§20.1, Req 11.5, Req 13.1).

The same normalizer is applied symmetrically to BOTH the gold answer and the
System's produced qualitative answer (Req 14.3). The qualitative reasoner
(:mod:`app.physics.qualitative_reasoner`), the qualitative branch in
:func:`app.physics.solver.solve`, and the qualitative-equivalence pathway in
:mod:`app.eval.scorers` all consume this module so every label production
funnels through one rule set.

Canonical label set (closed)
----------------------------

The closed canonical set is::

    {"increases", "decreases", "unchanged", "halved", "doubled",
     "maximum", "minimum", "unknown"}

Most rule classes collapse to ``increases`` / ``decreases`` / ``unchanged``
because those are the labels the deterministic monotonic-sign reasoner
produces. The four magnitude / extremum labels (``halved``, ``doubled``,
``maximum``, ``minimum``) round-trip exact gold tokens that the dataset uses
in some rows, but they are *equivalent* to a directional label for scoring
purposes:

* ``doubled`` ~ ``increases``
* ``halved`` ~ ``decreases``
* ``maximum`` / ``minimum`` are kept as canonical extremum labels but have no
  directional equivalent (the reasoner abstains for extremum rows).

The :func:`labels_equivalent` helper encodes this equivalence so the scorer
treats a gold ``"Doubled"`` as equal to a produced ``"increases"`` when both
went through the same rule set (Req 14.3).

Evaluation order
----------------

Rule classes are checked in this order so that compound phrasings resolve to
the **answer-bearing** label rather than a premise mention or a magnitude
qualifier:

1. ``unchanged`` β€” "unchanged", "no change", "remains the same",
   "do(es) not change", "remains constant", "constant", "same", "equal".
2. ``maximum`` / ``minimum`` β€” direct extremum tokens
   ("maximum", "max", "minimum", "min", "peak").
3. ``increases`` β€” "increase(s/d/ing)", "raises", "higher", "rises",
   "grows", "more", "brighter", "tripled", "quadrupled".
4. ``decreases`` β€” "decrease(s/d/ing)", "reduced", "lower", "dimmer",
   "diminished", "falls", "drops", "less".
5. ``doubled`` β€” "doubled", "twice", "two times", "by 2 times".
6. ``halved`` β€” "halved", "halfed", "becomes half", "by half",
   "to one half", "to 1/2".

Order matters: "remains the same" (which contains "same") resolves to
``unchanged``; "Resistance decreases -> current increases" reports the
*answer* label ``increases`` because direction words are checked before the
magnitude words. The ``doubled`` / ``halved`` classes only fire when a phrase
mentions the magnitude *without* a direction word (so "Doubled" alone β†’
``doubled`` but "Increase by 2 times" β†’ ``increases``). This matches both
how the dataset's gold answers are written and the directional labels the
deterministic reasoner produces.

Unmatched phrases return ``"unknown"`` β€” the scorer treats this as an
abstention, not a wrong guess (Req 11.1).

This module is intentionally free of LLM dependencies and consumes no
external state.
"""

from __future__ import annotations

import re
from typing import Final, Literal


__all__ = [
    "CanonicalLabel",
    "CANONICAL_LABELS",
    "ABSTENTION_LABEL",
    "DIRECTIONAL_EQUIVALENCE",
    "normalize_qualitative_label",
    "labels_equivalent",
    "is_qualitative_gold",
]


CanonicalLabel = Literal[
    "increases",
    "decreases",
    "unchanged",
    "halved",
    "doubled",
    "maximum",
    "minimum",
    "unknown",
]


# Closed canonical set. The non-abstention members are the labels the scorer
# treats as a possible match (Req 14.3). The ``unknown`` label is the
# abstention marker; it never counts as a correct match against a non-
# ``unknown`` gold (Req 11.1).
CANONICAL_LABELS: Final[frozenset[str]] = frozenset(
    {
        "increases",
        "decreases",
        "unchanged",
        "halved",
        "doubled",
        "maximum",
        "minimum",
    }
)

ABSTENTION_LABEL: Final[str] = "unknown"


# Equivalence collapsing magnitude labels onto their directional counterpart
# for symmetric scoring. The reasoner produces directional labels (sign Γ—
# direction); the dataset's gold answers sometimes use magnitude tokens.
# Treating ``doubled`` ~ ``increases`` and ``halved`` ~ ``decreases`` lets
# the scorer accept directional answers against magnitude golds (and vice
# versa) without losing the literal label distinction in the canonical set.
DIRECTIONAL_EQUIVALENCE: Final[dict[str, str]] = {
    "doubled": "increases",
    "halved": "decreases",
}


# ---------------------------------------------------------------------------
# Phrasing-class regex tables (general rules; never per-question text).
# ---------------------------------------------------------------------------

_UNCHANGED_PATTERNS: Final[tuple[str, ...]] = (
    r"\bunchanged\b",
    r"\bno\s+change\b",
    r"\bremains?\s+(?:the\s+)?same\b",
    r"\bremains?\s+constant\b",
    r"\bdo(?:es)?\s+not\s+change\b",
    r"\bdoesn'?t\s+change\b",
    r"\bdon'?t\s+change\b",
    r"\bunaltered\b",
    r"\bconstant\b",
    r"\bequal\b",
    r"\bsame\b",
)

_MAXIMUM_PATTERNS: Final[tuple[str, ...]] = (
    r"\bmaximum\b",
    r"\bmaximal\b",
    r"\bmax\b",
    r"\bpeak\b",
    r"\bgreatest\b",
    r"\blargest\b",
)

_MINIMUM_PATTERNS: Final[tuple[str, ...]] = (
    r"\bminimum\b",
    r"\bminimal\b",
    r"\bmin\b",
    r"\bsmallest\b",
    r"\bleast\b",
    r"\blowest\b",
)

_INCREASES_PATTERNS: Final[tuple[str, ...]] = (
    r"\bincreas\w*",
    r"\braised?\b",
    r"\braises\b",
    r"\braising\b",
    r"\bhigher\b",
    r"\bbrighter\b",
    r"\btripl\w*",
    r"\bquadrupl\w*",
    r"\brise\w*",
    r"\bgrow\w*",
    r"\bgrew\b",
    r"\bgrown\b",
    r"\bmore\b",
    r"\bgreater\b",
    r"\bstronger\b",
)

_DECREASES_PATTERNS: Final[tuple[str, ...]] = (
    r"\bdecreas\w*",
    r"\breduc\w*",
    r"\blower\b",
    r"\blowered\b",
    r"\blowering\b",
    r"\bdimmer\b",
    r"\bdiminish\w*",
    r"\bfall\w*",
    r"\bdrop\w*",
    r"\bless\b",
    r"\bweaker\b",
)

_DOUBLED_PATTERNS: Final[tuple[str, ...]] = (
    r"\bdoubled\b",
    r"\btwice\b",
    r"\btwo\s+times\b",
    r"\bby\s+two\s+times\b",
    r"\bby\s+2\s+times\b",
    r"\bby\s+a\s+factor\s+of\s+2\b",
)

_HALVED_PATTERNS: Final[tuple[str, ...]] = (
    r"\bhalved\b",
    r"\bhalfed\b",  # spelling variant observed in dataset
    r"\bbecomes?\s+half\b",
    r"\bby\s+half\b",
    r"\bto\s+one\s+half\b",
    r"\bto\s+a\s+half\b",
    r"\bto\s+1/2\b",
)


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compiles a tuple of regex strings into ignore-case Patterns."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_UNCHANGED_RES = _compile(_UNCHANGED_PATTERNS)
_MAXIMUM_RES = _compile(_MAXIMUM_PATTERNS)
_MINIMUM_RES = _compile(_MINIMUM_PATTERNS)
_INCREASES_RES = _compile(_INCREASES_PATTERNS)
_DECREASES_RES = _compile(_DECREASES_PATTERNS)
_DOUBLED_RES = _compile(_DOUBLED_PATTERNS)
_HALVED_RES = _compile(_HALVED_PATTERNS)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Checks if the text matches any of the compiled Patterns."""
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_qualitative_label(text: object) -> CanonicalLabel:
    """Map a free-text qualitative phrase to a canonical label.

    Returns one of the closed canonical labels:
    ``"increases"`` | ``"decreases"`` | ``"unchanged"`` | ``"halved"`` |
    ``"doubled"`` | ``"maximum"`` | ``"minimum"`` | ``"unknown"``.

    The mapping is a single, general phrasing-equivalence rule set. It is
    NEVER a per-question text -> answer mapping (AGENTS.md §20.1, Req 11.5,
    Req 13.1). A phrase that does not match any rule class returns
    ``"unknown"`` — the scorer treats this as an abstention, not a wrong
    guess (Req 11.1).

    The same normalizer is applied symmetrically to both the gold answer
    and the produced answer in the scorer (Req 14.3) so equivalence is
    symmetric: gold "Doubled" → ``doubled`` and produced "increases" →
    ``increases`` then compare via :func:`labels_equivalent`.

    Args:
        text: The free-text input object (usually str) representing the qualitative answer.

    Returns:
        The matched CanonicalLabel or "unknown".
    """

    if not isinstance(text, str):
        return "unknown"
    t = text.strip().lower()
    if not t:
        return "unknown"

    # Order matters. ``unchanged`` is checked first so "remains the same"
    # (which contains "same") wins over any direction tokens. Extrema are
    # checked next so "maximum" stays distinct from any directional word.
    # Direction words win over magnitude words so "Increase by 2 times"
    # resolves to ``increases`` rather than ``doubled``.
    if _matches_any(t, _UNCHANGED_RES):
        return "unchanged"
    if _matches_any(t, _MAXIMUM_RES):
        return "maximum"
    if _matches_any(t, _MINIMUM_RES):
        return "minimum"
    if _matches_any(t, _INCREASES_RES):
        return "increases"
    if _matches_any(t, _DECREASES_RES):
        return "decreases"
    if _matches_any(t, _DOUBLED_RES):
        return "doubled"
    if _matches_any(t, _HALVED_RES):
        return "halved"
    return "unknown"


def labels_equivalent(produced: object, gold: object) -> bool:
    """True when produced and gold qualitative answers are equivalent.

    Both arguments are normalized via :func:`normalize_qualitative_label`,
    then compared after collapsing magnitude labels onto their directional
    counterpart via :data:`DIRECTIONAL_EQUIVALENCE`. The comparison is
    symmetric (Req 14.3):

    * ``"Doubled"`` (gold) ~ ``"increases"`` (produced) → True.
    * ``"Halved"`` (gold) ~ ``"decreases"`` (produced) → True.
    * ``"Increase by 2 times"`` (gold) ~ ``"increases"`` (produced) → True
      (the direction word wins in normalization).
    * ``"unknown"`` never matches a non-``"unknown"`` gold (Req 11.1) so
      abstaining never counts as correct against a directional gold.

    Args:
        produced: The produced answer object.
        gold: The gold answer object.

    Returns:
        True if the normalized versions represent equivalent directions/magnitudes.
    """

    p = normalize_qualitative_label(produced)
    g = normalize_qualitative_label(gold)
    if p == "unknown" or g == "unknown":
        return False
    p_eq = DIRECTIONAL_EQUIVALENCE.get(p, p)
    g_eq = DIRECTIONAL_EQUIVALENCE.get(g, g)
    return p_eq == g_eq


def is_qualitative_gold(gold_answer: object, gold_unit: object) -> bool:
    """Return True when a gold answer is a qualitative direction-of-change.

    A row counts as qualitative when (a) the gold unit is empty / absent /
    a placeholder dash and (b) the normalized gold answer maps to a
    non-``unknown`` canonical label. The check is a *structural* property
    of the row's columns (gold_unit, gold_answer); it never inspects the
    question text. The scorer uses this to opt the qualitative-equivalence
    pathway in for the right rows (Req 14.3).

    Args:
        gold_answer: The gold answer object from the dataset.
        gold_unit: The gold unit object from the dataset.

    Returns:
        True if the row represents a qualitative physics answer; False otherwise.
    """

    unit_str = "" if gold_unit is None else str(gold_unit).strip()
    # Dataset uses both "-" and the unicode em-dash "\u2014" as placeholders
    # for "no unit"; treat any single-character dash as empty.
    placeholders = {"", "-", "\u2014", "\u2013", "--", "n/a", "N/A", "none"}
    if unit_str not in placeholders:
        return False
    if normalize_qualitative_label(gold_answer) == "unknown":
        return False
    return True

