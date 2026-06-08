"""Logic-side pattern store: capture syntactic shapes the FOL+Z3 translator
struggles with, and let a planner-side Method short-circuit them with a
proven rewrite template instead of running the LLM-translation loop again.

Motivation (AGENTS.md §24 Phase F4)
-----------------------------------
The FOL+Z3 self-refinement loop is bounded (default 2 rounds). On novel
syntactic shapes (e.g. "X unless Y", "exactly N of {A, B, C}", "X iff Y
unless Z"), the loop often spends both rounds re-trying the same wrong
translation, then abstains. Even if the LLM eventually emits a correct
clause shape on round k, that effort is paid AGAIN on every future
question of the same shape.

Pattern store
-------------
A persistent JSON file (``models/logic_patterns.json``) records:

  * ``signature``: structural fingerprint of the failing premise text
    (regex-anchored, no per-question text — see AGENTS.md §20.1).
  * ``rewrite_template``: a SAFE, deterministic rewrite that turns the
    raw premise text into a form the FOL+Z3 translator already handles.
    For instance, "X unless Y" → "If not Y, then X" (Logic-LM rewrite).
  * ``successes`` / ``uses`` lifetime stats so a flaky pattern can be
    demoted.

Discovery loop
--------------
When FOL+Z3 abstains AND atom-coverage faithfulness reports a dropped
premise, the planner can record the failing premise's shape and ask the
LLM (one call) for a rewrite template. The template is then VALIDATED
backend-side by re-running the FOL+Z3 pipeline on the rewritten premise:
if it produces a definite verdict, the pattern is registered.

This file currently ships with the SEED of three shapes whose rewrite is
purely structural (no LLM needed): "unless", "except when", "provided
that". Future shapes will be appended via the discovery loop.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "logic_patterns.json"
)


@dataclass
class LogicPattern:
    """A failing-shape signature + its validated rewrite template.

    ``signature`` is a Python regex string applied with ``re.IGNORECASE``
    against the lowercased premise. ``rewrite_template`` is a Python
    format-style string with named groups from the signature; capture
    groups in the regex become named substitutions in the template.

    Example:

        signature        = r"^(?P<main>.+?)\\s+unless\\s+(?P<exc>.+?)\\.?$"
        rewrite_template = "if not {exc}, then {main}."
    """

    pattern_id: str
    signature: str
    rewrite_template: str
    description: str = ""
    uses: int = 0
    successes: int = 0

    def matches(self, premise_text: str) -> re.Match | None:
        return re.search(self.signature, premise_text, re.IGNORECASE)

    def apply(self, match: re.Match) -> str:
        groups = match.groupdict()
        return self.rewrite_template.format(**groups)

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses


# Seed patterns: structural rewrites that turn an "unless"-style premise
# into the canonical "if-then" the FOL+Z3 translator already handles.
# These are STRUCTURAL — same rewrite generalizes to every "X unless Y"
# instance, never per-question text (AGENTS.md §20.1).
_SEED_PATTERNS: list[LogicPattern] = [
    LogicPattern(
        pattern_id="seed.unless",
        signature=r"^(?P<main>.+?)\s+unless\s+(?P<exc>.+?)\.?$",
        rewrite_template="if not {exc}, then {main}.",
        description="\"X unless Y\" -> \"If not Y, then X\".",
    ),
    LogicPattern(
        pattern_id="seed.except_when",
        signature=r"^(?P<main>.+?)\s+except\s+when\s+(?P<exc>.+?)\.?$",
        rewrite_template="if not {exc}, then {main}.",
        description="\"X except when Y\" -> \"If not Y, then X\".",
    ),
    LogicPattern(
        pattern_id="seed.provided_that",
        signature=r"^(?:provided\s+that\s+)(?P<cond>.+?)[,]\s*(?P<conc>.+?)\.?$",
        rewrite_template="if {cond}, then {conc}.",
        description="\"Provided that X, Y\" -> \"If X, then Y\".",
    ),
    LogicPattern(
        pattern_id="seed.only_if",
        # "X only if Y" expresses Y as a NECESSARY condition for X (X -> Y).
        # The contrapositive (not Y -> not X) is what makes "X only if Y" + "not Y"
        # entail "not X". We rewrite to the explicit contrapositive form so the
        # FOL+Z3 translator picks up the modus-tollens chain.
        signature=r"^(?P<main>.+?)\s+only\s+if\s+(?P<cond>.+?)\.?$",
        rewrite_template="if not {cond}, then not {main}.",
        description="\"X only if Y\" -> \"If not Y, then not X\" (contrapositive of necessary cond).",
    ),
    LogicPattern(
        pattern_id="seed.however_does_not_apply",
        # "However, X does not apply to Y" / "X does not apply when Y" — common
        # policy-exception phrasing where a positive rule is overridden by a
        # specific exception. Rewrite as a separate exception clause the
        # translator can express as an exclusion.
        signature=r"^however,?\s+(?P<rule>.+?)\s+does\s+not\s+apply\s+to\s+(?P<exception>.+?)\.?$",
        rewrite_template="if {exception}, then not ({rule}).",
        description="\"However, X does not apply to Y\" -> exception clause.",
    ),
]


class LogicPatternStore:
    """Thread-safe registry of logic-rewrite patterns with optional persistence."""

    def __init__(self, *, persistence_path: Path | None = None) -> None:
        self._patterns: list[LogicPattern] = list(_SEED_PATTERNS)
        self._lock = threading.RLock()
        self._persistence_path = persistence_path

    def all(self) -> list[LogicPattern]:
        with self._lock:
            return list(self._patterns)

    def find_match(self, premise_text: str) -> tuple[LogicPattern, re.Match] | None:
        """First pattern whose signature matches the premise text."""
        with self._lock:
            for pattern in self._patterns:
                match = pattern.matches(premise_text)
                if match:
                    return pattern, match
        return None

    def register(self, pattern: LogicPattern) -> bool:
        with self._lock:
            for existing in self._patterns:
                if existing.signature == pattern.signature:
                    return False
            self._patterns.append(pattern)
            return True

    def record(self, pattern_id: str, success: bool) -> None:
        with self._lock:
            for pattern in self._patterns:
                if pattern.pattern_id == pattern_id:
                    pattern.uses += 1
                    if success:
                        pattern.successes += 1
                    return

    # ----- persistence (discovered patterns only) ---------------------------

    def load(self) -> int:
        if self._persistence_path is None or not self._persistence_path.exists():
            return 0
        try:
            data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        loaded = 0
        for entry in data.get("patterns", []):
            try:
                self.register(LogicPattern(
                    pattern_id=str(entry["pattern_id"]),
                    signature=str(entry["signature"]),
                    rewrite_template=str(entry["rewrite_template"]),
                    description=str(entry.get("description") or ""),
                    uses=int(entry.get("uses", 0) or 0),
                    successes=int(entry.get("successes", 0) or 0),
                ))
                loaded += 1
            except Exception:
                continue
        return loaded

    def persist(self) -> None:
        if self._persistence_path is None:
            return
        with self._lock:
            payload = {
                "version": 1,
                "patterns": [
                    {
                        "pattern_id": pattern.pattern_id,
                        "signature": pattern.signature,
                        "rewrite_template": pattern.rewrite_template,
                        "description": pattern.description,
                        "uses": pattern.uses,
                        "successes": pattern.successes,
                    }
                    for pattern in self._patterns
                    if not pattern.pattern_id.startswith("seed.")
                ],
            }
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


# ---------------------------------------------------------------------------

_DEFAULT_STORE: LogicPatternStore | None = None
_DEFAULT_STORE_LOCK = threading.RLock()


def get_default_pattern_store() -> LogicPatternStore:
    """Lazy singleton matching the MethodLibrary pattern."""
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            store = LogicPatternStore(persistence_path=_DEFAULT_STORE_PATH)
            store.load()
            _DEFAULT_STORE = store
        return _DEFAULT_STORE


# ---------------------------------------------------------------------------
# Helper: rewrite a list of premises through the store.
# ---------------------------------------------------------------------------


def rewrite_premises(
    premises: list[str], *, store: LogicPatternStore | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    """Apply every applicable pattern to each premise in order.

    Returns (rewritten_list, applied_log) where ``applied_log`` records
    each application so the audit trail can show which patterns fired.
    Premises with no matching pattern pass through unchanged.

    Handles premise-ID prefixes (e.g. "P1: ", "Premise 2: ") by stripping
    them before matching and re-prefixing after rewriting. This is
    structural (not per-question) because the prefix format is standard
    across the entire pipeline.
    """
    import re as _re
    _PREFIX_RE = _re.compile(r"^(P\d+:\s*|Premise\s+\d+:\s*)", _re.IGNORECASE)

    # Step 0: intra-premise anaphora resolution ("passes it" -> "passes Course B")
    # so conjunctive multi-hop chains whose conjunct uses a pronoun can match the
    # ground fact. Conservative (object-pronoun + nearest salient NP only).
    try:
        from app.logic._anaphora import resolve_anaphora
        premises, _anaphora_log = resolve_anaphora(premises)
    except Exception:
        _anaphora_log = []

    s = store or get_default_pattern_store()
    rewritten: list[str] = []
    log: list[dict[str, str]] = list(
        {"pattern_id": "anaphora", "before": e["before"], "after": e["after"]}
        for e in _anaphora_log
    )
    for premise in premises:
        # Strip premise-ID prefix before matching.
        prefix_match = _PREFIX_RE.match(premise)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = premise[len(prefix):] if prefix else premise

        match_result = s.find_match(body)
        if match_result is None:
            rewritten.append(premise)
            continue
        pattern, match = match_result
        try:
            new_text = pattern.apply(match)
        except Exception:
            rewritten.append(premise)
            continue
        # Re-prefix the rewritten text so downstream premise normalization
        # and ID assignment stay consistent.
        rewritten.append(prefix + new_text)
        log.append(
            {
                "pattern_id": pattern.pattern_id,
                "before": premise,
                "after": prefix + new_text,
            }
        )
    return rewritten, log
