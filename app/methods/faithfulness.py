"""Translation faithfulness gate (Logic-LM-style content preservation).

The translator may DROP or DISTORT a premise without saying so — this gate
catches that BEFORE the prover reasons over a degraded theory.

Strategy
--------
Two complementary checks:

  1. **Atom coverage** (cheap, runs on every translation):
     Every "salient content token" in the input must appear (after stemming)
     in at least one clause of the translation. A premise whose entire
     content set is missing from the clauses is a translation drop.

  2. **Round-trip semantic check** (LLM call, only when atom coverage looks
     OK): ask a second LLM call to render each clause back into NL, then
     ensure the round-tripped sentence preserves the original's polarity
     and entity set. Used sparingly (once per translation, NOT per round of
     refinement) so it doesn't blow the call budget.

Both checks return a structured ``FaithfulnessReport`` so the planner /
coverage gate / refinement loop can act on a single source of truth.

A faithfulness FAILURE is not a crash — it is an instruction to refine
(if the refinement loop has rounds left) or to abstain. Hard gate, soft
recovery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# --- Token utilities (re-use the deterministic stemmer used elsewhere) ----

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "of", "to", "in", "on",
    "for", "with", "and", "or", "but", "not", "no", "any", "some", "all",
    "every", "each", "this", "that", "these", "those", "it", "they",
    "them", "their", "there", "here", "what", "which", "who", "whom",
    "if", "then", "else", "than", "as", "by", "from", "into", "onto",
    "over", "under", "can", "may", "might", "would", "could", "should",
    "will", "shall", "must", "thus", "so", "yet",
}


def _content_tokens(text: str) -> set[str]:
    """Best-effort content-token set used for atom-coverage checks."""
    if not text:
        return set()
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9_]+", text)
    out: set[str] = set()
    for tok in tokens:
        if tok in _STOP or len(tok) < 3:
            continue
        # Mild stemming: drop trailing 's' / 'ed' / 'ing'.
        stem = tok
        if stem.endswith("ies") and len(stem) > 4:
            stem = stem[:-3] + "y"
        elif stem.endswith("ing") and len(stem) > 5:
            stem = stem[:-3]
        elif stem.endswith("ed") and len(stem) > 4:
            stem = stem[:-2]
        elif stem.endswith("s") and len(stem) > 3:
            stem = stem[:-1]
        out.add(stem)
    return out


def _polarity(text: str) -> bool:
    """Coarse polarity detector: True if the sentence reads as a positive claim."""
    low = text.lower()
    return not bool(re.search(r"\b(?:not|never|no|isn't|aren't|cannot)\b", low))


# ---------------------------------------------------------------------------


@dataclass
class FaithfulnessReport:
    """Outcome of a faithfulness check on a translation.

    ``passed`` is True only when EVERY checked premise (or every input
    quantity, in the physics case) is faithfully represented. ``drops`` and
    ``distortions`` are structured so the refinement loop can compose
    targeted feedback.
    """

    passed: bool
    drops: list[dict[str, str]] = field(default_factory=list)
    distortions: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.drops or self.distortions)

    def feedback_text(self) -> str:
        """Compose a structural feedback string for translation refinement."""
        parts: list[str] = []
        if self.drops:
            ids = ", ".join(d.get("id", "?") for d in self.drops)
            parts.append(
                f"The previous translation appears to have DROPPED these inputs: "
                f"{ids}. Re-translate so each is represented by at least one clause."
            )
        for dist in self.distortions:
            parts.append(
                f"Input {dist.get('id', '?')} may be DISTORTED: {dist.get('reason', 'mismatch')}. "
                f"Re-check polarity, entity, and atom names against the original text."
            )
        for note in self.notes:
            parts.append(note)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Atom coverage check (no LLM).
# ---------------------------------------------------------------------------


def check_atom_coverage(
    inputs: Iterable[Any],
    clause_blobs: Iterable[str],
    *,
    id_attr: str = "id",
    text_attr: str = "text",
    min_overlap: int = 1,
) -> FaithfulnessReport:
    """Verify every input has at least ``min_overlap`` content tokens covered.

    ``inputs`` is the iterable of premises / quantities the translator was
    asked to handle. ``clause_blobs`` is the iterable of strings extracted
    from the translation (concatenated atoms / formulae / variable names).
    """
    blob_tokens: set[str] = set()
    for blob in clause_blobs:
        blob_tokens |= _content_tokens(blob)
    drops: list[dict[str, str]] = []
    for item in inputs:
        item_id = str(getattr(item, id_attr, "") or "")
        item_text = str(getattr(item, text_attr, item) or "")
        item_tokens = _content_tokens(item_text)
        if not item_tokens:
            continue
        overlap = item_tokens & blob_tokens
        if len(overlap) < min_overlap:
            drops.append(
                {
                    "id": item_id or item_text[:40],
                    "reason": (
                        f"no content tokens of '{item_text[:60]}' appear in the "
                        f"translated clauses (missing: {sorted(item_tokens)[:5]})"
                    ),
                }
            )
    return FaithfulnessReport(passed=not drops, drops=drops)


# ---------------------------------------------------------------------------
# Round-trip polarity check (1 LLM call).
# ---------------------------------------------------------------------------


_ROUND_TRIP_PROMPT = """You are a translation auditor. Given an ORIGINAL premise and its FOL/DSL \
translation, re-render the translation in plain English and compare. Reply with JSON ONLY:

{{
  "round_trip": "the translation rendered back into one short English sentence",
  "polarity_matches": true | false,
  "entity_matches": true | false,
  "issue": "" or "<short reason>"
}}

ORIGINAL: {original}
TRANSLATION: {translation}
"""


def check_round_trip(
    inputs: list[Any],
    translations: list[str],
    *,
    llm_client: Any,
    max_checks: int = 3,
) -> FaithfulnessReport:
    """LLM-assisted round-trip check on up to ``max_checks`` premises.

    A drop in atom coverage is much cheaper to detect — call this only when
    atom coverage already passed and you want a higher-precision pass before
    accepting a non-decisive verdict. Bounded by ``max_checks`` so it never
    spends more than a fixed number of LLM calls per request.
    """
    if llm_client is None or not inputs or not translations:
        return FaithfulnessReport(passed=True, notes=["round_trip_skipped"])

    pairs = list(zip(inputs[:max_checks], translations[:max_checks]))
    distortions: list[dict[str, str]] = []
    for original, translation in pairs:
        original_text = str(getattr(original, "text", original))
        try:
            prompt = _ROUND_TRIP_PROMPT.format(
                original=original_text, translation=translation
            )
            # Use the chat shape if available; else generate.
            if hasattr(llm_client, "chat"):
                resp = llm_client.chat(
                    "default", prompt, max_tokens=200, response_format=False,
                )
                content = getattr(resp, "content", None) or str(resp or "")
            else:
                content = str(llm_client(prompt) or "")
            payload = _parse_json(content)
        except Exception as exc:
            distortions.append(
                {
                    "id": str(getattr(original, "id", "?")),
                    "reason": f"round_trip_error:{type(exc).__name__}",
                }
            )
            continue
        if not payload:
            continue
        if not bool(payload.get("polarity_matches", True)):
            distortions.append(
                {
                    "id": str(getattr(original, "id", "?")),
                    "reason": f"polarity_mismatch:{payload.get('issue', '')}".strip(":"),
                }
            )
        elif not bool(payload.get("entity_matches", True)):
            distortions.append(
                {
                    "id": str(getattr(original, "id", "?")),
                    "reason": f"entity_mismatch:{payload.get('issue', '')}".strip(":"),
                }
            )
    return FaithfulnessReport(passed=not distortions, distortions=distortions)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object from ``text`` (strips fences/prose)."""
    import json
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None
