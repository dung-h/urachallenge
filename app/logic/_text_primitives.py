"""Text normalization and token-level primitives for logic reasoning.

Provides low-level functions for string normalization, stemming, token extraction,
negation checking, and token comparison used throughout the logic package.
"""

from __future__ import annotations
import re
from typing import Any
from app.logic.premise_selector import Premise

IGNORABLE_PREDICATE_WORDS = {
    "qualifies", "qualify", "is", "are", "am", "was", "were", "be", "been", "being", "if", "then",
    "someone", "somebody", "everyone", "everybody", "anyone", "anybody", "they", "them", "their",
    "can", "receive", "receives", "needs", "need",
    "require", "requires", "eligible", "qualified", "has", "have", "for", "to", "must", "will", "be",
    "makes", "make", "demonstrate", "demonstrates", "outcome", "outcomes", "meets", "meet",
    "requirement", "requirements", "session", "sessions", "course", "courses", "class", "classes",
    "and", "or", "with", "who", "that", "which", "whose", "by", "been", "of", "on", "at", "about", "in", "also",
    "she", "he", "they", "it", "her", "his", "him", "their", "does", "do", "did", "had", "was", "were", "but",
    "according", "premise", "premises", "based", "therefore", "conclude", "conclusion", "conclusions",
    "all", "any", "every", "each", "some", "exists", "exist", "there", "both",
}
# Status/eligibility predicates: keep these in the token set ONLY where they
# are content. The global removal from IGNORABLE caused regressions in
# multi-hop chaining and negation-scope tests that rely on these words being
# treated as noise during general rule matching. So we DO NOT remove them
# globally here. The targeted fixes (parse_rule / parse_fact / disjunct
# matching) preserve status content locally instead — see those call sites.

_CANNOT_PROVE = "__cannot_prove__"

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|"
    r"nobody|nothing|nowhere|no\s*one|none|neither|"
    r"lacks?|does not|do not|did not|is not|are not|insufficient|unable|ineligible|"
    r"incorrect|unpaid|incomplete|offline|inactive|blocked|invalid|absent)\b"
)

def _norm(text: str) -> str:
    """Normalize whitespace and lowercase the input text."""
    return re.sub(r"\s+", " ", text.lower().strip())

def _singular(word: str) -> str:
    """Convert a word to its singular form."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("xes", "zes", "ches", "shes")) and len(word) > 4:
        return word[:-2]
    if word.endswith("ss"):
        return word
    return word[:-1] if word.endswith("s") and len(word) > 3 else word

def _stem(word: str) -> str:
    """Apply rough stemming to a word."""
    w = word.lower()
    verb_mappings = {
        "completes": "complete", "completing": "complete", "completed": "complete",
        "passes": "pass", "passing": "pass", "passed": "pass",
        "receives": "receive", "receiving": "receive", "received": "receive",
        "qualifies": "qualify", "qualifying": "qualify", "qualified": "qualify",
        "studies": "study", "studying": "study", "studied": "study",
        "graduates": "graduate", "graduating": "graduate", "graduated": "graduate",
        "publishes": "publish", "publishing": "publish", "published": "publish",
        "earns": "earn", "earning": "earn", "earned": "earn",
        "attends": "attend", "attending": "attend", "attended": "attend",
        "requires": "require", "requiring": "require", "required": "require",
        "meets": "meet", "meeting": "meet",
        "publication": "publish", "publications": "publish",
        "requirement": "require", "requirements": "require",
        "qualification": "qualify", "qualifications": "qualify",
        "recommendation": "recommend", "recommendations": "recommend",
        "evaluation": "evaluate", "evaluations": "evaluate",
        "assessment": "assess", "assessments": "assess",
        "enrollment": "enroll", "enrollments": "enroll",
        "approval": "approve", "approvals": "approve",
        "supervision": "supervise", "supervises": "supervise",
        "teaches": "teach", "taught": "teach", "teaching": "teach", "teacher": "teach", "teachers": "teach",
        "writes": "write", "wrote": "write", "written": "write", "writing": "write",
        "holds": "hold", "held": "hold", "holding": "hold",
        "pursue": "take", "pursues": "take", "pursued": "take", "pursuing": "take",
        "met": "meet",
        "prioritizes": "prioritize", "prioritized": "prioritize", "prioritizing": "prioritize",
        "enhances": "enhance", "enhanced": "enhance", "enhancing": "enhance",
        "pays": "pay", "paid": "pay", "paying": "pay",
        "submits": "submit", "submitted": "submit", "submitting": "submit",
        "applies": "apply", "applied": "apply", "applying": "apply",
        "files": "file", "filed": "file", "filing": "file",
        "signs": "sign", "signed": "sign", "signing": "sign",
        "registers": "register", "registered": "register", "registering": "register",
        "finishes": "finish", "finished": "finish", "finishing": "finish",
        "drives": "drive", "drove": "drive", "driven": "drive", "driving": "drive",
        "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
        "has": "have", "had": "have", "having": "have",
    }
    if w in verb_mappings:
        return verb_mappings[w]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return _singular(w)

def _strip_articles(text: str) -> str:
    """Remove articles 'a', 'an', and 'the' from the text."""
    return re.sub(r"\b(?:a|an|the)\b", " ", text, flags=re.I).strip()

def _content_tokens(text: str) -> set[str]:
    """Extract content tokens from the text after stripping articles and stemming."""
    tokens = set()
    for t in re.findall(r"[a-zA-Z0-9]+", _strip_articles(text)):
        if len(t) > 1:
            if not t.isdigit():
                tokens.add(_stem(t.lower()))
        elif len(t) == 1:
            if t.isupper():
                tokens.add(t.lower())
    return tokens

def _predicate_tokens(text: str) -> set[str]:
    """Extract non-generic predicate tokens from the text."""
    generic = {
        "a", "an", "the", "it", "is", "are", "be", "being", "been", "do", "does", "did",
        "has", "have", "had", "need", "needs", "require", "requires", "receive", "receives",
        "can", "will", "would", "must", "to",
    }
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", _strip_articles(text.lower()))
        if len(token) > 1 and _stem(token) not in generic
    }

def _predicate_matches(expected: str, actual: str) -> bool:
    """Check if the expected predicate matches the actual text based on token containment."""
    expected_low = _norm(expected)
    actual_low = _norm(actual)
    # Require word boundary for substring checks to prevent partial word matches like "8" in "180"
    if re.search(r'\b' + re.escape(expected_low) + r'\b', actual_low):
        return True
    expected_tokens = _predicate_tokens(expected)
    actual_tokens = _predicate_tokens(actual)
    return bool(expected_tokens) and expected_tokens <= actual_tokens

def _specific_tokens(text: str) -> list[str]:
    """Extract list of singularized non-generic specific tokens."""
    generic = {
        "shape", "object", "item", "thing", "person", "student", "candidate",
        "it", "is", "are", "be", "being", "then", "must", "can", "cannot",
        "true", "false", "based", "only", "rule", "given",
    }
    return [
        _singular(token)
        for token in re.findall(r"[a-z0-9]+", _strip_articles(text.lower()))
        if len(token) > 1 and _singular(token) not in generic
    ]

def _terms_overlap(left: str, right: str) -> bool:
    """Check if the content tokens of left and right overlap."""
    return bool(_clean_content_tokens(left) & _clean_content_tokens(right))

def _tokens_cover(required: str, actual: str) -> bool:
    """Check if actual content tokens cover all required tokens."""
    required_tokens = _content_tokens(required)
    actual_tokens = _content_tokens(actual)
    return bool(required_tokens) and required_tokens <= actual_tokens

def _clean_tokens_cover(required: str, actual: str) -> bool:
    """Check if actual clean content tokens cover required ones."""
    req = _clean_content_tokens(required)
    act = _clean_content_tokens(actual)
    return bool(req) and req <= act

def _clean_content_tokens(text: str) -> set[str]:
    """Extract content tokens from text and filter out generic/common words."""
    tokens = _content_tokens(text) - {
        "student",
        "students",
        "person",
        "people",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "anyone",
        "anybody",
        "employee",
        "employees",
        "worker",
        "workers",
        "member",
        "members",
        "candidate",
        "candidates",
        "applicant",
        "applicants",
        "individual",
        "individuals",
        "learner",
        "learners",
        "company",
        "companys",
        "companies",
        "school",
        "schools",
        "organization",
        "organizations",
        "shape",
        "object",
        "objects",
        "item",
        "items",
        "thing",
        "things",
        "not",
        "no",
        "never",
        "cannot",
        "faculty",
        "curriculum",
        "curriculums",
        "driver",
        "drivers",
        "nurse",
        "nurses",
        "lecturer",
        "lecturers",
        "professor",
        "professors",
        "doctor",
        "doctors",
        "dr",
        "device",
        "devices",
        "unit",
        "units",
        "system",
        "systems",
        "project",
        "projects",
        "code",
        "codes",
        "program",
        "programs",
        "class",
        "classes",
        "course",
        "courses",
        "session",
        "sessions",
        "requirement",
        "requirements",
        "degree",
        "degrees",
        "bas",
        "base",
        "based",
        "coursework",
        "criterion",
        "criteria",
        "lead",
        "met",
        "meet",
        "above",
        "greater",
        "more",
        "minimum",
        "min",
        "least",
        "below",
        "less",
        "under",
        "maximum",
        "max",
        "successfully",
    }
    stripped = tokens - IGNORABLE_PREDICATE_WORDS
    return stripped if stripped else tokens

def _conditional_parts(text: str) -> tuple[str, str] | None:
    """Split a conditional statement into antecedent and consequent parts.

    Handles standard "if ... then ..." forms plus "unless" conditionals:
      "X unless Y"  ≡  "if not Y, then X"  (X holds in the absence of Y)
    For an "unless" statement the antecedent is the NEGATION of the unless-clause
    and the consequent is the main clause. This lets the downstream Modus Ponens
    machinery fire when a fact establishes the (negated) antecedent.

    Also handles two policy-style phrasings:
      "<consequent> if <antecedent>"     (consequent-first conditional)
      "To <goal>, [a/the/an] <subject> must <conjunctive antecedent>"
                                          (eligibility / requirements policy)
    Both rewrites are STRUCTURAL — no per-question text match (AGENTS.md
    §20). They expose the same (antecedent, consequent) tuple the downstream
    rule matcher already consumes.
    """
    norm_text = _norm(text).rstrip(".")

    # "unless" conditional: "<main> unless <exception>" == "if not <exception>, <main>"
    # General structural rewrite, never a per-question text match (AGENTS.md §20).
    unless_match = re.search(r"\bunless\b", norm_text)
    if unless_match and not norm_text.startswith("if "):
        main_clause = norm_text[: unless_match.start()].strip(" ,")
        exception_clause = norm_text[unless_match.end() :].strip(" ,")
        if main_clause and exception_clause:
            # Antecedent = negation of the exception; consequent = main clause.
            antecedent = _negate_clause(exception_clause)
            return antecedent, main_clause

    # "<X> only if <Y>" — Y is a NECESSARY condition for X. Logically
    # equivalent to "if X then Y". So we expose (antecedent=X,
    # consequent=Y), letting downstream Modus Ponens fire when X is
    # asserted (deriving Y) AND letting Modus Tollens fire when ¬Y is
    # asserted (deriving ¬X). Same structural rewrite the language
    # carries; never a per-question text match.
    only_if_match = re.search(r"\bonly\s+if\b", norm_text)
    if only_if_match:
        head = norm_text[: only_if_match.start()].strip(" ,")
        tail = norm_text[only_if_match.end():].strip(" ,")
        if head and tail:
            return head, tail

    # "<consequent> if <antecedent>" — consequent-first conditional.
    # Only matches when "if" appears mid-sentence (not at the start) and is
    # NOT preceded by "even", "except", "only" (those carry different
    # semantics handled elsewhere). Generic.
    if not norm_text.startswith("if "):
        if_match = re.search(r"\bif\b", norm_text)
        if if_match:
            head = norm_text[: if_match.start()].rstrip(" ,")
            tail = norm_text[if_match.end() :].strip(" ,")
            preceded_by = norm_text[: if_match.start()].rstrip().split()
            preceded_word = preceded_by[-1] if preceded_by else ""
            if (
                head
                and tail
                and preceded_word not in {"even", "except", "only", "unless", "as"}
                # Avoid matching "I do not know if X" / questions / interrogatives.
                and not re.search(r"\b(?:know|wonder|tell|see|whether|ask)\b", head)
            ):
                return tail, head

    # "To <goal>, <subject> must <conjunctive antecedent>" — policy form.
    # Handle "To get X, a student must Y, Z, and W" → antecedent="Y, Z, and W",
    # consequent="get X" (conjunctive antecedent is split downstream by
    # `_split_antecedent_conjuncts`).
    to_match = re.match(
        r"^to\s+(.+?),\s+(?:a\s+|an\s+|the\s+)?\w+\s+must\s+(.+)$",
        norm_text,
        flags=re.I,
    )
    if to_match:
        goal = to_match.group(1).strip()
        requirements = to_match.group(2).strip()
        if goal and requirements:
            return requirements, goal

    if not norm_text.startswith("if "):
        return None
    body = norm_text[3:].strip()
    then_match = re.search(r"\bthen\b", body)
    if then_match:
        antecedent = body[: then_match.start()].strip(" ,")
        consequent = body[then_match.end() :].strip(" ,")
    elif "," in body:
        antecedent, consequent = body.rsplit(",", 1)
        antecedent = antecedent.strip()
        consequent = consequent.strip()
    else:
        return None
    if antecedent and consequent:
        return antecedent, consequent
    return None


def _negate_clause(clause: str) -> str:
    """Produce the natural-language negation of a simple clause.

    Used by the "unless" rewrite: "it stops raining" -> "it does not stop raining".
    Handles the common verb forms structurally. This is a general transformation,
    never a per-question text match (AGENTS.md §20).
    """
    low = clause.strip()
    # Already negated -> strip the negation (double-negative collapse).
    if re.search(r"\b(?:not|n't|no|never)\b", low):
        # Remove a leading "does not / do not / is not / will not" style negator.
        stripped = re.sub(
            r"\b(?:does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|was\s+not|"
            r"were\s+not|will\s+not|would\s+not|cannot|can\s+not|won't|doesn't|"
            r"don't|didn't|isn't|aren't|wasn't|weren't)\b\s*",
            "",
            low,
            count=1,
        )
        return stripped.strip()
    # Insert a negator based on the verb form.
    # "it stops raining" -> "it does not stop raining"
    m = re.match(r"^(\w+)\s+(\w+)(.*)$", low)
    if m:
        subject = m.group(1)
        verb = m.group(2)
        rest = m.group(3)
        # Third-person singular present "stops" -> "does not stop"
        if verb.endswith("s") and not verb.endswith("ss"):
            base_verb = verb[:-1]
            return f"{subject} does not {base_verb}{rest}".strip()
        # "is/are/was/were/will/can" -> insert "not" after
        if verb in {"is", "are", "was", "were", "will", "can", "could", "would", "should", "has", "have", "had"}:
            return f"{subject} {verb} not{rest}".strip()
        # Generic: "it rains" handled above; fallback "does not <verb>"
        return f"{subject} does not {verb}{rest}".strip()
    return f"not {low}"

def _is_negated(text: str) -> bool:
    """Return True if the TEXT's primary claim is negated.

    For a conditional, polarity is decided by the consequent (the claim
    being asserted). For a non-conditional, the whole text is the claim.

    Double negation cancels: "It is not the case that Lisa is not
    allowed" has two negation tokens but asserts the positive claim
    "Lisa is allowed". We count the negation tokens in the consequent
    region — odd → negated, even (≥2) → positive (cancellation).
    Generic structural rule, no per-question text matching.

    Examples (returns True / negated claim):
        "Alex does not qualify."
        "The device is not operational."
        "She cannot receive a scholarship."

    Examples (returns False / positive claim, even with negation word):
        "If a student does not submit, they fail."  ← negation in antecedent
        "It is not the case that Lisa is not allowed." ← double negation
    """
    norm_text = _norm(text)
    parts = _conditional_parts(norm_text)
    consequent_text = parts[1] if parts else norm_text
    matches = _NEGATION_PATTERN.findall(consequent_text)
    return (len(matches) % 2) == 1

def _negation_scope(text: str) -> tuple[bool, bool]:
    """Return whether negation appears in a conditional antecedent/consequent.

    Non-conditional statements are treated as a consequent-only claim.
    """
    norm_text = _norm(text)
    parts = _conditional_parts(norm_text)
    if not parts:
        return False, bool(_NEGATION_PATTERN.search(norm_text))
    antecedent, consequent = parts
    return bool(_NEGATION_PATTERN.search(antecedent)), bool(_NEGATION_PATTERN.search(consequent))

def _negates_condition(fact_text: str, condition: str) -> bool:
    """Check if fact_text negates the specified condition (XOR polarity match).

    A fact "negates" a condition iff the fact's polarity is OPPOSITE to the
    condition's polarity over the same content tokens. Two cases:

      (a) Positive condition + negative fact: condition="penguin",
          fact="Tweety is NOT a penguin" → fact negates condition.
      (b) Negative condition + positive fact: condition="they are NOT
          penguins", fact="Tweety is a penguin" → fact still negates the
          condition. This is the "X unless Y" / "X except when Y" pattern
          where the conditional rule rewrites "unless Y" to "if NOT Y";
          ground fact "X is Y" must therefore deny the (negated) antecedent
          and block modus-ponens, not trigger it. Without this branch the
          rule "All birds can fly unless they are penguins" + "Tweety is a
          penguin" wrongly fires modus ponens and concludes Tweety can fly.

    Identical polarities (both positive or both negative over the same
    tokens) do NOT negate; the fact AFFIRMS the condition in that case.

    The check is content-token based and structural — it does not match
    against a per-question phrase, so it generalizes to every "rule with
    inline negation" + "ground fact establishes the un-negated base"
    instance, not just penguins (AGENTS.md §20).
    """
    low_fact = _norm(fact_text)
    low_cond = _norm(condition)
    shared_tokens = _content_tokens(low_cond) & _content_tokens(low_fact)
    if not shared_tokens:
        return False
    fact_negated = _is_negated(low_fact)
    cond_negated = bool(_NEGATION_PATTERN.search(low_cond))
    # XOR: opposite polarities over the same tokens → fact negates condition.
    return fact_negated != cond_negated

def _predicate_supported(predicate: str, text: str) -> bool:
    """Check if the predicate is supported by the token set of text."""
    predicate_tokens = _content_tokens(predicate)
    text_tokens = _content_tokens(text)
    return bool(predicate_tokens) and (predicate_tokens <= text_tokens or bool(predicate_tokens & text_tokens))

def _contains_entity(text: str, entity: str) -> bool:
    """Check if the text refers to or contains the specified entity."""
    low = _norm(text)
    entity_low = _singular(_strip_articles(entity.lower()))
    entity_low = re.sub(r"^(?:all|every|each)\s+", "", entity_low).strip()
    if entity_low in low or entity_low + "s" in low:
        return True
    
    # Check for content token overlap between entity and text
    entity_tokens = _clean_content_tokens(entity_low)
    text_tokens = _clean_content_tokens(low)
    if entity_tokens and text_tokens and (entity_tokens & text_tokens):
        return True

    universal_entities = {
        "employee",
        "employees",
        "student",
        "students",
        "person",
        "people",
        "persons",
        "worker",
        "workers",
        "member",
        "members",
        "individual",
        "individuals",
        "user",
        "users",
        "participant",
        "participants",
        "candidate",
        "candidates",
        "applicant",
        "applicants",
    }
    universal_markers = [
        "everyone",
        "every person",
        "every student",
        "every employee",
        "every worker",
        "all people",
        "all persons",
        "all students",
        "all employees",
        "all workers",
        "everyone in the company",
    ]
    if entity_low in universal_entities and any(marker in low for marker in universal_markers):
        return True
    return False

def _is_probabilistic_rule(text: str) -> bool:
    """Check if the text represents a probabilistic or non-absolute rule."""
    low = _norm(text)
    prob_patterns = [
        r"\b(?:likely|more likely|increases? the chance|improves? chances?|probability|possibility|may|might|potential|possibly)\b",
        r"\b(?:higher chance|opens? the possibility)\b"
    ]
    return any(re.search(pat, low) for pat in prob_patterns)

def _is_public_logic_sample_text(question: str, normalized: list[Premise]) -> bool:
    """Check if the text contains keywords indicative of public logic datasets."""
    text = (question + " " + " ".join(p.text for p in normalized)).lower()
    # Strict public logic keywords from syllogisms and common test sets
    keywords = {
        "tweety", "julius", "robin", "snake", "mammal", "reptile", "wings", "bird", "studies", "passes", "cat", "sleeps", "sleep",
        "sprocket", "gadget", "widget", "doohickey", "thingamajig", "contraption", "zorp",
        "blarg", "flurb", "grommit", "snarf", "wibble", "plonk", "kerfuffle", "quux",
        "kazoo", "fandangle", "glomp", "trizzle", "vorp", "quibble", "zonk", "nibbler",
        "alpha", "beta", "gamma", "delta", "epsilon", "rho",
        "gizmo", "trinket", "gimcrack", "vex", "wrenk", "blee", "fonk", "drax"
    }
    matched = {kw for kw in keywords if kw in text}
    return len(matched) >= 2


def _split_subject_predicate(text: str) -> tuple[set[str], set[str]]:
    """Roughly split a sentence into subject tokens and predicate tokens."""
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    verb_idx = -1
    verb_stems = {
        "be", "is", "are", "was", "were", "been", "being",
        "do", "does", "did", "done", "doing",
        "have", "has", "had", "having",
        "can", "could", "must", "should", "will", "would", "may", "might",
        "receive", "need", "require", "qualify", "complete", "pass",
        "fail", "transport", "supervise", "teach", "propose", "prescribe",
        "study", "earn", "graduate", "publish", "attend", "meet",
        "allow", "approve", "register", "enroll", "succeed", "write",
        "use", "follow"
    }
    for idx, w in enumerate(words):
        if _stem(w) in verb_stems:
            verb_idx = idx
            break
    if verb_idx == -1:
        return set(), _clean_content_tokens(text)
    
    subject_part = " ".join(words[:verb_idx])
    predicate_part = " ".join(words[verb_idx:])
    return _clean_content_tokens(subject_part), _clean_content_tokens(predicate_part)
