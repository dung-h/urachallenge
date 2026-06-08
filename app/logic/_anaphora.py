"""Conservative intra-premise anaphora resolution.

Motivation (Session 11m). The deterministic solver abstained on chains like:

    P2: "If a student enrolls in Course B and passes it, they can enroll in Course C."
    P5: "David has enrolled in and passed Course B."

The conjunct "passes it" never matched the ground fact "passed Course B"
because "it" was not resolved to "Course B". Replacing "passes it" with
"passes Course B" makes the chain fire (verified). This module performs that
substitution.

Design principles (kept deliberately conservative to avoid wrong rewrites):
  * Resolve ONLY within a single premise (no cross-premise coreference).
  * Resolve a pronoun to the NEAREST PRECEDING salient noun phrase, where a
    salient NP is one of:
      - "Course X" / "Module X" / "Chapter X" / "Level X" (label + token), or
      - a capitalized proper noun (e.g. "Biochemistry", "David"), or
      - a "the <noun>" phrase ("the book", "the project").
  * Only resolve object pronouns in a verb-object position ("passes it",
    "completes it", "submits them") — the failure mode we observed.
  * NEVER touch "it" inside fixed phrases ("it is the case", "if it", weather
    "it"), and never resolve subject "it/they" at clause start.
  * If no confident antecedent is found, leave the pronoun untouched.

This is structural (no per-question text); it generalizes to every
"<verb> it/them" + preceding course/entity reference.
"""

from __future__ import annotations

import re

# Object pronouns we attempt to resolve (lowercased, whole-word).
_OBJ_PRONOUNS = ("it", "them")

# Salient noun-phrase patterns, matched against the text BEFORE the pronoun.
# Ordered by specificity; we take the LAST (nearest) match before the pronoun.
_LABELLED_NP = re.compile(
    r"\b((?:Course|Module|Chapter|Level|Unit|Section|Lesson|Exam|Test|Project|"
    r"Assignment|Task|Stage|Phase|Grade|Class)\s+[A-Z0-9][A-Za-z0-9]*)\b"
)
# A capitalized content noun (proper noun / named subject) — but NOT a sentence
# starter or common function word.
_CAP_NP = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Verb immediately preceding the pronoun (object position). We only rewrite
# "<verb> it/them" to avoid touching subject or idiomatic uses.
_VERB_PRON = re.compile(
    r"\b(passes|passed|pass|completes|completed|complete|submits|submitted|"
    r"submit|takes|took|take|finishes|finished|finish|attends|attended|attend|"
    r"enters|entered|enter|reviews|reviewed|review|repeats|repeated|repeat)\s+"
    r"(it|them)\b",
    re.IGNORECASE,
)

# Words that look capitalized but are sentence-initial function words; never
# use them as antecedents.
_CAP_STOP = {
    "If", "The", "A", "An", "All", "No", "Some", "Every", "Each", "When",
    "Then", "Anyone", "Someone", "Everyone", "Nobody", "They", "It", "He",
    "She", "We", "You", "Provided", "Unless", "Whenever", "While", "Since",
}


def _nearest_antecedent(text_before: str) -> str | None:
    """Find the nearest salient NP in ``text_before`` (text preceding pronoun)."""
    # Prefer a labelled NP ("Course B") — most reliable.
    labelled = list(_LABELLED_NP.finditer(text_before))
    if labelled:
        return labelled[-1].group(1)
    # Else the nearest capitalized content noun that is not a stopword.
    caps = [m.group(1) for m in _CAP_NP.finditer(text_before)
            if m.group(1) not in _CAP_STOP]
    if caps:
        return caps[-1]
    return None


def resolve_anaphora_in_premise(text: str) -> str:
    """Return ``text`` with confidently-resolvable object pronouns replaced.

    Only rewrites "<verb> it/them" when a nearest salient antecedent exists in
    the text preceding the verb. Leaves everything else untouched.
    """
    if not text:
        return text
    # Quick reject: nothing to do if no candidate pattern.
    if not _VERB_PRON.search(text):
        return text

    out = text
    # Resolve left-to-right; rebuild using offsets from a single pass so an
    # earlier resolution's antecedent search uses the ORIGINAL preceding text.
    result_parts: list[str] = []
    last_end = 0
    for m in _VERB_PRON.finditer(text):
        verb, pron = m.group(1), m.group(2)
        text_before = text[: m.start()]
        antecedent = _nearest_antecedent(text_before)
        # Singular pronoun should map to a singular antecedent; "them" is more
        # likely plural but our labelled NPs are singular — only rewrite "them"
        # when the antecedent is a labelled NP (e.g. "Modules 1-3" rare); to stay
        # safe, restrict "them" to labelled NPs.
        if antecedent is None:
            continue
        if pron.lower() == "them" and not _LABELLED_NP.search(text_before):
            continue
        # Append the unchanged gap + the rewritten "<verb> <antecedent>".
        result_parts.append(text[last_end : m.start()])
        result_parts.append(f"{verb} {antecedent}")
        last_end = m.end()
    if last_end == 0:
        return text  # nothing resolved
    result_parts.append(text[last_end:])
    return "".join(result_parts)


def resolve_anaphora(premises: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Resolve intra-premise anaphora across a premise list.

    Returns (rewritten_premises, log) where log records each substitution for
    the audit trail. Premise-ID prefixes ("P1: ") are preserved.
    """
    _PREFIX = re.compile(r"^(P\d+:\s*|Premise\s+\d+:\s*)", re.IGNORECASE)
    out: list[str] = []
    log: list[dict[str, str]] = []
    for premise in premises:
        pm = _PREFIX.match(premise)
        prefix = pm.group(0) if pm else ""
        body = premise[len(prefix):] if prefix else premise
        new_body = resolve_anaphora_in_premise(body)
        if new_body != body:
            log.append({"before": premise, "after": prefix + new_body})
        out.append(prefix + new_body)
    return out, log
