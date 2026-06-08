"""Parser for extracting structured queries and options from natural language questions.

Provides utility functions to parse multiple-choice options, identify question polarity,
extract subjects and predicates, and map Boolean answers to multiple-choice labels.
"""

from __future__ import annotations
import re
from typing import Any
from app.logic._text_primitives import (
    _norm, _singular, _stem, _strip_articles, _content_tokens,
    _clean_content_tokens, _is_negated, _conditional_parts,
    IGNORABLE_PREDICATE_WORDS, _NEGATION_PATTERN, _predicate_tokens,
)


_QUERY_NEGATION_RE = re.compile(
    r"\b(?:not|cannot|can't|no|never|definitely)\b",
    flags=re.I,
)


def _last_question_sentence(text: str) -> str:
    """Extract the final sentence containing the actual question from the text."""
    raw = str(text or "").strip()
    if re.search(r"\bstatement\s*:\s*if\b", raw, flags=re.I):
        return raw
    if len(_labeled_options(raw)) >= 2:
        return raw
    if re.search(r"(?m)^\s*[A-E][\.\):]\s+", raw) or re.search(r"(?m)^\s*\d+[\.)]\s+", raw):
        return raw
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", raw) if part.strip()]
    question_parts = [part for part in parts if part.endswith("?")]
    if question_parts:
        last = question_parts[-1]
        # Guard against false sentence splits on a title abbreviation
        # ("Dr.", "Mr.", "Prof.", ...): the period after the title is NOT a
        # sentence boundary, so "Can Dr. John teach ...?" must not become
        # "John teach ...?" (which loses the modal + subject and breaks the
        # question parser). If the chosen tail starts mid-question because the
        # PRECEDING fragment ends in a known abbreviation, glue them back.
        idx = parts.index(last) if last in parts else -1
        while idx > 0 and re.search(
            r"\b(?:dr|mr|mrs|ms|prof|st|sr|jr|mt|messrs|rev|hon|gen|sen|col|lt|"
            r"capt|cmdr|sgt|vs|etc|no|fig|eg|ie)\.$",
            parts[idx - 1].strip(), flags=re.I,
        ):
            last = parts[idx - 1] + " " + last
            idx -= 1
        return last
    return raw

def _labeled_options(question: str) -> dict[str, str]:
    """Parse multiple-choice options with their labels (A, B, C, D, E) from the text."""
    raw = str(question or "").strip()
    line_options: dict[str, str] = {}
    for line in raw.splitlines():
        match = re.match(r"^\s*([A-E])[\.\):]\s*(.+)$", line, flags=re.I)
        if match:
            line_options[match.group(1).upper()] = match.group(2).strip()
    if len(line_options) >= 2:
        return line_options
    segments = [raw]
    if "?" in raw:
        segments.insert(0, raw[raw.rfind("?") + 1 :].strip())
    label_pattern = re.compile(r"(?:^|\s)([A-E])[\.\):]\s+", flags=re.I)
    for segment in segments:
        matches = list(label_pattern.finditer(segment))
        if len(matches) < 2:
            continue
        options: dict[str, str] = {}
        for idx, match in enumerate(matches):
            label = match.group(1).upper()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
            text = segment[start:end].strip(" \t\r\n-–—.;")
            if text:
                options[label] = text
        if len(options) >= 2:
            return options
    bare_label_pattern = re.compile(r"(?:^|\s)([A-E])\s+")
    for segment in segments:
        matches = list(bare_label_pattern.finditer(segment))
        if len(matches) < 2:
            continue
        options = {}
        for idx, match in enumerate(matches):
            label = match.group(1).upper()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
            text = segment[start:end].strip(" \t\r\n-–—.;")
            if text:
                options[label] = text
        if len(options) >= 2:
            return options
    return {}

def _option_text_to_question(option_text: str) -> str:
    """Convert a flat option description into an interrogative question format."""
    text = option_text.strip().rstrip(".")
    if not text:
        return text
    if text.endswith("?"):
        return text
    if text.lower().startswith(("if ", "all ", "no ", "some ")):
        return f"Does it follow that {text}?"
    negated_be = re.match(r"^(.+?)\s+(isn't|is\s+not|aren't|are\s+not|wasn't|was\s+not|weren't|were\s+not)\s+(.+)$", text, flags=re.I)
    if negated_be:
        subject = negated_be.group(1).strip()
        predicate = negated_be.group(3).strip()
        aux = "Are" if negated_be.group(2).lower().startswith(("are", "aren")) else "Is"
        return f"{aux} {subject} not {predicate}?"
    negated_modal = re.match(r"^(.+?)\s+(can't|cannot|can\s+not)\s+(.+)$", text, flags=re.I)
    if negated_modal:
        return f"Can {negated_modal.group(1).strip()} not {negated_modal.group(3).strip()}?"
    subject_modal = re.match(r"^(.+?)\s+(can|could|must|will|would|should|may)\s+(.+)$", text, flags=re.I)
    if subject_modal:
        return f"{subject_modal.group(2).capitalize()} {subject_modal.group(1).strip()} {subject_modal.group(3).strip()}?"
    subject_be = re.match(r"^(.+?)\s+(is|are|was|were|has|have|needs|need|requires|require|eligible|qualified|authorized)\s+(.+)$", text, flags=re.I)
    if subject_be:
        aux = subject_be.group(2).capitalize()
        if aux.lower() in {"eligible", "qualified", "authorized"}:
            return f"Is {subject_be.group(1).strip()} {subject_be.group(2).lower()} {subject_be.group(3).strip()}?"
        return f"{aux} {subject_be.group(1).strip()} {subject_be.group(3).strip()}?"
    if re.match(r"^(?:does|is|are|did|may|can|must|will|would)\b", text, re.I):
        return text + "?" if not text.endswith("?") else text
    return f"Does {text}?"

def _is_abstain_option(text: str) -> bool:
    """Check if the option text indicates a neutral or uncertain decision."""
    norm_text = _norm(text)
    if re.search(r"\bcannot\s+(?:be\s+)?determined\b", norm_text):
        return True
    if re.search(r"\binsufficient\s+(?:evidence|information|data)\b", norm_text):
        return True
    if norm_text in {"unknown", "undetermined", "uncertain", "it is unknown", "it is undetermined", "not sure", "cannot tell", "cannot decide"}:
        return True
    return False

def _question_polarity(question: str) -> str | None:
    """Return the polarity the question is asking about, if explicit.

    We only use this for narrow polarity questions like "known/unknown",
    because those cases need yes/no mapping instead of a generic unknown.
    """

    _subject, predicate, negative = _question_subject_predicate(question)
    if not predicate:
        return None
    pred = _norm(predicate)
    if "unknown" in pred or "undetermined" in pred:
        return "unknown"
    if "known" in pred or "know" == pred:
        return "known" if not negative else "unknown"
    return None

def _question_existential(question: str) -> tuple[str, str | None] | None:
    """Detect if the question asks for the existence of an entity satisfying a condition."""
    q = _norm(question).rstrip("?")
    verb_query = re.match(
        r"^(?:does|do|did)\s+(?:there\s+)?(?:exist\s+)?at\s+least\s+one\s+(.+?)\s+"
        r"(submit|submits|participate|participates|attend|attends|receive|receives|have|has|complete|completes|"
        r"support|supports|maintain|maintains|create|creates|agree|agrees|take|takes|pass|passes)\s+(.+)$",
        q,
    )
    if verb_query:
        entity = _strip_articles(verb_query.group(1).strip())
        predicate = f"{verb_query.group(2)} {verb_query.group(3)}".strip()
        return entity, predicate
    match = re.match(r"^(?:are|is|were|was)\s+there\s+any\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        match = re.match(r"^(?:are|is|were|was)\s+any\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        match = re.match(r"^(?:does|do|did)\s+(?:there\s+)?(?:exist\s+)?at\s+least\s+one\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        match = re.match(r"^(?:does|do|did)\s+any\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        match = re.match(r"^some\s+(.+?)\s+(?:are|is|have|has|can|do|does|did|will)\s+(.+)$", q)
    if not match:
        return None
    entity = _strip_articles(match.group(1).strip())
    predicate = match.group(2).strip() if match.group(2) else None
    return entity, predicate

def _question_conditional_statement(question: str) -> tuple[str, str] | None:
    """Extract an explicit conditional claim from a meta question.

    Handles prompts like:
    - "Does it follow that if X, then Y?"
    - "Is it true that if X then Y?"
    """

    q = _norm(question).rstrip("?")
    statement_match = re.search(r"\bstatement\s*:\s*(if\b.+)$", q)
    if statement_match:
        q = statement_match.group(1).strip().rstrip("?")
    q = re.sub(r"^(?:does|is|are|would|will)\s+it\s+(?:follow|true|the case)\s+that\s+", "", q)
    q = re.sub(r"^(?:according to the premises,\s*)?", "", q)
    # Strip prompt noise suffixes and prefixes anywhere
    q = re.sub(r",?\s*(?:according|based)\s+to\s+(?:the\s+)?(?:premises|rules|coursework|facts)\b.*$", "", q)
    match = re.match(r"^if\s+(.+?),?\s+then\s+(.+)$", q)
    if not match:
        parts = _conditional_parts(q)
        if parts:
            return parts
    if not match:
        return None
    antecedent = match.group(1).strip()
    consequent = match.group(2).strip()
    if antecedent and consequent:
        return antecedent, consequent
    return None

def _question_status_subject(question: str) -> str | None:
    """Extract the unit entity mentioned in a status-related question."""
    low = _norm(question)
    if "status" not in low and "failed" not in low and "operational" not in low:
        return None
    match = re.search(r"\bunit\s+([a-z0-9]+)\b", low)
    return f"unit {match.group(1)}" if match else None

def _question_asks_antecedent(question: str, antecedent: str) -> bool:
    """Check if the question specifically queries the given antecedent clause."""
    subject, predicate, _negative = _question_subject_predicate(question)
    if not subject or not predicate:
        return False
    antecedent_tokens = _predicate_tokens(antecedent)
    question_tokens = _predicate_tokens(" ".join([subject, predicate]))
    return bool(antecedent_tokens) and antecedent_tokens <= question_tokens

def _question_subject_predicate(question: str) -> tuple[str | None, str | None, bool]:
    """Parse the query into subject, predicate, and polarity components."""
    q = _norm(question).rstrip("?")
    q = re.sub(
        r"^(does|is|are|did|can|could|should|would)\s+(?:the\s+)?(?:logical\s+)?(?:chain|progression|sequence|premises|rules|facts)\s+(?:demonstrate|show|prove|indicate|entail|support|supports)\s+(?:that\s+)?",
        r"\1 ",
        q,
        flags=re.I
    )
    # Strip prompt noise suffixes
    q = re.sub(r",?\s+(?:according\s+to|based\s+on)\s+(?:the\s+)?(?:premises|rules|coursework|facts)\b.*$", "", q)
    
    # Strip prefixes like 'based on dr. john\'s qualifications, '
    prefix_subject = None
    match_prefix = re.match(
        r"^(?:based\s+on|according\s+to)\s+([a-zA-Z0-9.\s]+)'s\s+(?:qualifications|status|progress|academic\s+progress|requirements|credentials|situation),?\s*",
        q,
        flags=re.I
    )
    if match_prefix:
        prefix_subject = match_prefix.group(1).strip()
        q = q[match_prefix.end():].strip()

    # Safe prefix cleaning
    q = re.sub(r"^does\s+it\s+(?:follow|true|the\s+case)\s+that\s+", "", q, flags=re.I)
    q = re.sub(r"^is\s+it\s+(?:follow|true|the\s+case)\s+that\s+", "", q, flags=re.I)
    q = re.sub(r"^(?:according\s+to|based\s+on)\s+(?:the\s+)?(?:premises|rules|coursework|facts|text|requirements|guidelines|principles|parameters|strategies|policy|policies),?\s*", "", q, flags=re.I)
    
    if not q.startswith("if "):
        starts = [idx for token in ["does ", "is ", "are ", "did ", "which "] if (idx := q.find(token)) >= 0]
        if starts:
            min_start = min(starts)
            if min_start < 15:
                q = q[min_start:]
    negative = bool(_QUERY_NEGATION_RE.search(q))

    # Future-tense subject-verb pattern: "Will <subject> <verb>?" / "Will the
    # <subject> <verb>?". The downstream parser only handles is/are/does/did.
    # Map this to subject + verb form so universal-syllogism chains can fire.
    # Generic and structural — no per-question text matching (AGENTS.md §20.1).
    will_match = re.match(
        r"^(will|won't|won\s+not|shall)\s+(?:the\s+)?([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*)?)\s+(.+)$",
        q,
        flags=re.I,
    )
    if will_match:
        subj = will_match.group(2).strip()
        verb_phrase = will_match.group(3).strip()
        if subj and verb_phrase and len(subj) <= 40:
            # Strip trailing period/qualifiers like " this semester".
            verb_phrase = re.sub(
                r"\s+(?:this|next|last)\s+(?:semester|year|month|week|day)\b.*$",
                "",
                verb_phrase,
            )
            return subj, verb_phrase, negative

    # "Is/Was <subject> a member of <category>?" / "...part of <X>?". The
    # generic pattern "Is X Y?" parses Y as the predicate, which gets
    # corrupted by "a member of category". Extract the bare category name.
    member_match = re.match(
        r"^(?:is|was|are|were)\s+([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*)?)\s+(?:a|an)\s+(?:member\s+of|part\s+of|element\s+of|instance\s+of)\s+(?:(?:the\s+|a\s+|an\s+|category\s+|class\s+|set\s+|group\s+)+)?(.+)$",
        q,
        flags=re.I,
    )
    if member_match:
        subj = member_match.group(1).strip()
        cat = member_match.group(2).strip().rstrip(".?!")
        if subj and cat:
            return subj, cat, negative

    universal_prefixes = ("are all ", "is all ", "do all ", "does all ", "did all ")
    universal_heads = [
        "thoroughly tested",
        "well-tested",
        "up to date with",
        "eligible",
        "qualified",
        "skilled",
        "suitable",
        "trained",
        "proficient",
        "ready",
        "allowed",
        "approved",
        "registered",
        "enrolled",
        "productive",
        "successful",
        "stable",
        "complete",
        "completed",
        "submitted",
        "submit",
        "earn",
        "receive",
        "have",
        "has",
        "is",
        "are",
        "be",
        "write"
    ]
    if q.startswith(universal_prefixes):
        rest = q.split(" ", 2)[2].strip() if len(q.split(" ", 2)) >= 3 else ""
        for head in universal_heads:
            match = re.search(rf"\b{re.escape(head)}\b", rest)
            if match and match.start() > 0:
                subject = rest[: match.start()].strip()
                predicate = rest[match.start() :].strip()
                if subject and predicate:
                    if q.startswith(("are all ", "do all ", "did all ")):
                        subject = "all " + subject
                    return subject, predicate, negative
    patterns = [
        r"^which\s+(?:statement|scenario|conclusion)\s+about\s+(.+?)\s+(?:is|are|was|were|does|has|have|requires|require)\s+(.+)$",
        r"^which\s+(?:[a-zA-Z0-9\s]+?)\s+(?:does|is|are|can|has|have|did|must)\s+(.+?)\s+(have|need|require|receive|pass|take|pursue|qualify|be\s+eligible|be\s+qualified|do)\??$",
        r"^which\s+of\s+the\s+following\s+((?:can|is|are)\s+(?:be\s+)?inferred\s+about)\s+(.+?)\??$",
        r"can (.+?) (.+)$",
        r"does (.+?) have (.+)$",
        r"does (.+?) need (.+)$",
        r"does (.+?) require (.+)$",
        r"does (.+?) receive (.+)$",
        r"does (.+?) (pass)$",
        r"does (.+?) (.+)$",
        r"must (.+?) be (.+)$",
        r"must (.+?) (.+)$",
        r"is the (.+?) (.+)$",
        r"is (.+?) (?:a |an )?(.+)$",
        r"are (.+?) (.+)$",
        r"did (.+?) (.+)$",
        r"have (.+?) (.+)$",
        r"has (.+?) (.+)$",
        r"had (.+?) (.+)$",
        r"do (.+?) (.+)$",
        r"was (.+?) (.+)$",
        r"were (.+?) (.+)$",
    ]
    subject, predicate = None, None
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                subject = groups[0].strip()
                predicate = groups[1].strip()
            else:
                subject = groups[0].strip()
                predicate = ""
            break
            
    if not subject and prefix_subject:
        subject = prefix_subject
        predicate = q

    if subject and prefix_subject and subject.lower() in {"he", "she", "it", "they"}:
        subject = prefix_subject

    if subject and predicate:
        for word in ["can", "cannot", "can't", "could", "couldn't", "unable", "fail", "fails", "may", "must", "will", "should", "does", "did", "do", "not", "exam", "scholarship", "admission", "award", "grant", "loan", "program"]:
            if subject.lower().endswith(" " + word):
                subject = subject[:-len(word)-1].strip()
                predicate = word + " " + predicate
                break
                
        # Handle titles
        titles = {"dr.", "dr", "professor", "prof.", "prof", "mr.", "mr", "mrs.", "mrs", "ms.", "ms", "unit"}
        sub_low = subject.lower()
        if sub_low in titles:
            parts = predicate.split(None, 1)
            if parts:
                subject = subject + " " + parts[0]
                predicate = parts[1] if len(parts) > 1 else ""

        # Shift non-verb words from predicate to subject (resolves multi-word subjects)
        predicate_verbs = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "cannot", "can't", "could", "must", "will", "would", "should",
            "attends", "attend", "completes", "complete", "studies", "study",
            "passes", "pass", "receives", "receive", "qualifies", "qualify",
            "graduates", "graduate", "publishes", "publish", "earns", "earn",
            "requires", "require", "meets", "meet", "teaches", "teach",
            "writes", "write", "holds", "hold", "enrolled", "enroll",
            "qualified", "eligible", "allowed", "approved", "registered",
            "safe", "suitable", "ready", "complete", "completed", "submitted", "submit",
            "provide", "provides", "support", "supports", "lead", "leads", "enhance", "enhances",
            "improve", "improves", "contribute", "contributes", "cancel", "cancels",
            "need", "needs", "needed", "needing",
            "propose", "proposes", "proposed", "proposing", "propos",
            "prescribe", "prescribes", "prescribed", "prescribing", "prescrib",
            "take", "takes", "took", "taken", "taking",
            "adhere", "adheres", "adhered", "adhering",
            "supervise", "supervises", "supervised", "supervising", "supervis",
            "apply", "applies", "applied", "applying",
            "benefit", "benefits", "benefited", "benefiting",
            "seek", "seeks", "sought", "seeking",
            "lack", "lacks", "lacked", "lacking",
            "access", "accesses", "accessed", "accessing",
            "enrolls", "enrolling",
            "demonstrate", "demonstrates", "demonstrated", "demonstrating", "demonstrat",
            "understand", "understands", "understood", "understanding",
            "solve", "solves", "solved", "solving",
            "pursue", "pursues", "pursued", "pursuing",
            "admit", "admits", "admitted", "admitting",
            "fail", "fails", "failed", "failing",
            "lose", "loses", "lost", "losing",
            "maintain", "maintains", "maintained", "maintaining",
            "achieve", "achieves", "achieved", "achieving",
            "perform", "performs", "performed", "performing",
            "agree", "agrees", "agreed", "agreeing",
            "rely", "relies", "relied", "relying",
        }
        parts = predicate.split()
        while len(parts) > 1 and parts[0].lower().rstrip(",.:;?") not in predicate_verbs:
            subject = subject + " " + parts.pop(0)
        predicate = " ".join(parts)
                
    return subject, predicate, negative

def _failure_status_prop(text: str) -> tuple[str, bool] | None:
    """Extract simple operational failure proposition: (entity, failed?)."""

    low = _norm(text)
    match = re.search(r"\bunit\s+([a-z0-9]+)\b", low)
    if not match:
        return None
    entity = f"unit {match.group(1)}"
    if re.search(r"\b(?:never\s+fails?|not\s+failed|not\s+fail|does\s+not\s+fail|operational)\b", low):
        return entity, False
    if re.search(r"\b(?:fails?|failed|has\s+failed)\b", low):
        return entity, True
    return None

def _choice_for_failure_status(question: str, failed: bool | None) -> str:
    """Map failure truth value to MCQ label when options are embedded."""

    options = dict(re.findall(r"\b([A-E])\)\s*(.*?)(?=\s+\b[A-E]\)|$)", question, flags=re.I | re.S))
    normalized_options = {label.upper(): _norm(text) for label, text in options.items()}
    if failed is True:
        for label, text in normalized_options.items():
            if "failed" in text and "not failed" not in text and "operational" not in text:
                return label
        return "yes"
    if failed is False:
        for label, text in normalized_options.items():
            if "operational" in text or "not failed" in text or "not fail" in text:
                return label
        return "no"
    for label, text in normalized_options.items():
        if "undetermined" in text or "unknown" in text or "cannot" in text:
            return label
    return "unknown"

def _choice_for_unknown(question: str) -> str:
    """Find the MCQ option label representing unknown or undetermined state."""
    options = _labeled_options(question)
    for label, text in options.items():
        if any(token in _norm(text) for token in ["undetermined", "unknown", "cannot be determined", "insufficient"]):
            return label.upper()
    return "unknown"

def _choice_for_boolean_answer(question: str, answer: str) -> str:
    """Map a 'yes', 'no', or 'unknown' resolution back to an MCQ option letter."""
    options = _labeled_options(question)
    if not options:
        return answer
    normalized = {label.upper(): _norm(text) for label, text in options.items()}
    if answer == "unknown":
        return _choice_for_unknown(question)

    affirmative_markers = [
        "yes",
        "true",
        "eligible",
        "operational",
        "complete",
        "certified",
        "safe",
        "known",
        "allowed",
        "pass",
        "passed",
        "present",
        "active",
    ]
    negative_markers = [
        "not ",
        "no ",
        "never",
        "cannot",
        "can't",
        "incomplete",
        "blocked",
        "ineligible",
        "inactive",
        "offline",
        "invalid",
        "unpaid",
        "not failed",
        "not eligible",
        "not safe",
        "not certified",
        "not complete",
    ]

    if answer == "yes":
        for label, text in normalized.items():
            if any(marker in text for marker in affirmative_markers) and not any(marker in text for marker in negative_markers):
                return label
        for label, text in normalized.items():
            if not any(marker in text for marker in negative_markers):
                return label
    elif answer == "no":
        for label, text in normalized.items():
            if any(marker in text for marker in negative_markers):
                return label
        for label, text in normalized.items():
            if "undetermined" in text or "unknown" in text:
                continue
            if "not" in text or "no" in text:
                return label
    return answer
