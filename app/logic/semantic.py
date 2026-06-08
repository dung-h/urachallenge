"""Semantic rule-based reasoning engine for multiple-choice questions.

Parses premises into logical atoms and rules, performs simple forward/backward
chaining, and scores MCQ options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Atom:
    """Represents a simplified logic atom containing quantifier, subject, and predicate."""
    quantifier: str
    subject: str
    predicate: str
    polarity: bool = True


@dataclass(frozen=True)
class Rule:
    """Represents a logical implication rule from antecedent to consequent."""
    antecedent: Atom
    consequent: Atom
    premise: Any


@dataclass
class SemanticResult:
    """Stores the verification results for the best matching MCQ option."""
    answer: str
    support: list[Any]
    reason: str


_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "can",
    "will",
    "would",
    "should",
    "must",
    "do",
    "does",
    "did",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "their",
    "his",
    "her",
    "its",
    "they",
    "them",
    "he",
    "she",
    "it",
    "student",
    "students",
    "person",
    "people",
    "someone",
    "somebody",
    "everyone",
    "everybody",
    "all",
    "every",
    "some",
    "there",
    "exists",
    "exist",
    "least",
    "one",
    "both",
    "and",
}

_SUBJECT_STOP = {
    "a",
    "an",
    "the",
    "all",
    "every",
    "each",
    "some",
    "there",
    "exists",
    "exist",
    "least",
    "one",
}

_PRONOUN_SUBJECTS = {"they", "them", "he", "she", "it"}


def _norm(text: str) -> str:
    """Normalize text spacing and clean apostrophe characters."""
    text = str(text or "").replace("’", "'")
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def _stem(word: str) -> str:
    """Stem common words to a normalized root form."""
    mapping = {
        "studious": "study",
        "studying": "study",
        "studies": "study",
        "studied": "study",
        "revising": "revise",
        "revises": "revise",
        "understanding": "understand",
        "understands": "understand",
        "submitting": "submit",
        "submits": "submit",
        "submitted": "submit",
        "assignments": "assignment",
        "thesis": "thesis",
        "receiving": "receive",
        "receives": "receive",
        "received": "receive",
        "meeting": "meet",
        "meets": "meet",
        "completing": "complete",
        "completes": "complete",
        "completed": "complete",
        "passing": "pass",
        "passes": "pass",
        "passed": "pass",
        "attending": "attend",
        "attends": "attend",
        "performance": "performance",
    }
    if word in mapping:
        return mapping[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _tokens(text: str) -> frozenset[str]:
    """Tokenize and stem content words in the text, skipping stopwords."""
    return frozenset(
        _stem(tok)
        for tok in re.findall(r"[a-z0-9]+", _norm(text))
        if len(tok) > 1 and _stem(tok) not in _STOP
    )


def _subject(text: str) -> str:
    """Extract and stem the subject noun of the text."""
    low = _norm(text)
    low = re.sub(r"^(?:all|every|each|some|at least one|there exists at least one)\s+", "", low)
    low = re.sub(r"\bwho\b.*$", "", low).strip()
    words = [w for w in re.findall(r"[a-z0-9]+", low) if w not in _SUBJECT_STOP]
    return _stem(words[0]) if words else ""


def _predicate(text: str) -> str:
    """Build a sorted token string representing the predicate properties of the text."""
    return " ".join(sorted(_tokens(text)))


def _polarity(text: str) -> bool:
    """Determine the logical polarity of the text (False if negated, True otherwise)."""
    return not bool(re.search(r"\b(?:not|no|never|cannot|can't|does not|do not|did not)\b", _norm(text)))


def _strip_negation(text: str) -> str:
    """Remove negative words from the normalized text string."""
    return re.sub(r"\b(?:not|no|never|cannot|can't|does not|do not|did not)\b", " ", _norm(text)).strip()


def _split_subject_predicate_text(text: str, inherited_subject: str | None = None) -> tuple[str, str]:
    """Split clause text into raw subject noun and predicate phrase."""
    low = _norm(text)
    low = re.sub(r"^(?:all|every|each|some|at least one|there exists at least one|there exists|there is|there are)\s+", "", low)
    low = re.sub(r"^(?:a|an|the)\s+", "", low)
    words = re.findall(r"[a-z0-9]+", low)
    if not words:
        return "", ""

    subject = _stem(words[0])
    if subject in _PRONOUN_SUBJECTS and inherited_subject:
        subject = inherited_subject

    pred = low
    if words[0] in _PRONOUN_SUBJECTS or subject == _stem(words[0]):
        pred = re.sub(rf"^{re.escape(words[0])}\b", "", pred, count=1).strip()

    pred = re.sub(r"^(?:who|that)\s+", "", pred)
    pred = re.sub(r"^(?:is|are|am|was|were|be|been|being|has|have|had|can|will|would|should|must|do|does|did)\s+", "", pred)
    return subject, pred


def _atom_from_clause(
    text: str,
    default_quantifier: str = "all",
    inherited_subject: str | None = None,
) -> Atom | None:
    """Build an Atom representation from a text clause."""
    low = _norm(text)
    quant = default_quantifier
    if re.match(r"^(?:there exists|there is|there are|at least one|some)\b", low):
        quant = "some"
    elif re.match(r"^(?:all|every|each)\b", low):
        quant = "all"

    subj, pred_text = _split_subject_predicate_text(low, inherited_subject=inherited_subject)
    pred = _predicate(_strip_negation(pred_text))
    if not subj or not pred:
        return None
    return Atom(quantifier=quant, subject=subj, predicate=pred, polarity=_polarity(low))


def _conditional_parts(text: str) -> tuple[str, str] | None:
    """Split a conditional string into antecedent and consequent clauses."""
    low = _norm(text)
    if not low.startswith("if "):
        return None
    body = low[3:].strip()
    match = re.search(r"\bthen\b", body)
    if match:
        return body[: match.start()].strip(" ,"), body[match.end() :].strip(" ,")
    if "," in body:
        left, right = body.rsplit(",", 1)
        return left.strip(), right.strip()
    return None


def _parse_premises(premises: list[Any]) -> tuple[list[tuple[Atom, Any]], list[Rule]]:
    """Parse list of premise objects into facts and rules."""
    facts: list[tuple[Atom, Any]] = []
    rules: list[Rule] = []
    for premise in premises:
        text = getattr(premise, "text", str(premise))
        parts = _conditional_parts(text)
        if parts:
            ant = _atom_from_clause(parts[0], default_quantifier="all")
            cons = _atom_from_clause(
                parts[1],
                default_quantifier="all",
                inherited_subject=ant.subject if ant else None,
            )
            if ant and cons:
                rules.append(Rule(ant, cons, premise))
            continue
        atom = _atom_from_clause(text)
        if atom:
            facts.append((atom, premise))
    return facts, rules


def _covers(actual: Atom, wanted: Atom) -> bool:
    """Check if actual Atom logically covers the wanted Atom properties."""
    if actual.polarity != wanted.polarity:
        return False
    if actual.subject != wanted.subject:
        return False
    actual_tokens = set(actual.predicate.split())
    wanted_tokens = set(wanted.predicate.split())
    return bool(wanted_tokens) and wanted_tokens <= actual_tokens


def _verify_atom(wanted: Atom, facts: list[tuple[Atom, Any]], rules: list[Rule]) -> tuple[bool, list[Any]]:
    """Determine if wanted Atom is entailed by facts and rules."""
    known: list[tuple[Atom, list[Any]]] = [(atom, [premise]) for atom, premise in facts]
    changed = True
    while changed:
        changed = False
        for atom, support in list(known):
            if wanted.quantifier == atom.quantifier and _covers(atom, wanted):
                return True, support
        for rule in rules:
            for atom, support in list(known):
                if atom.quantifier not in {"all", "some"}:
                    continue
                if not _covers(atom, rule.antecedent):
                    continue
                derived = Atom(atom.quantifier, atom.subject, rule.consequent.predicate, rule.consequent.polarity)
                if not any(existing == derived for existing, _support in known):
                    known.append((derived, list(dict.fromkeys(support + [rule.premise]))))
                    changed = True

        # Safe universal contraposition: all B, rule not A -> not B entails all A.
        for rule in rules:
            if rule.antecedent.polarity or rule.consequent.polarity:
                continue
            positive_ant = Atom("all", rule.antecedent.subject, rule.antecedent.predicate, True)
            positive_cons = Atom("all", rule.consequent.subject, rule.consequent.predicate, True)
            for atom, support in list(known):
                if atom.quantifier == "all" and _covers(atom, positive_cons):
                    derived = Atom("all", atom.subject, positive_ant.predicate, True)
                    if not any(existing == derived for existing, _support in known):
                        known.append((derived, list(dict.fromkeys(support + [rule.premise]))))
                        changed = True
    return False, []


def _split_option_claims(question: str, option: str) -> list[Atom] | None:
    """Extract list of Atoms claimed by the MCQ option text."""
    q = _norm(question)
    opt = _norm(option)

    # Contextual list option: "Which traits/outcomes do all students possess?"
    subject_match = re.search(r"\b(?:traits|outcomes|properties|capabilities)\s+(?:do|are)\s+(all|every)\s+([a-z0-9 -]+?)(?:\s+possess|\s+have|\s+guaranteed|\?|$)", q)
    if subject_match and "," in opt:
        subj = _stem(subject_match.group(2).split()[0])
        atoms = []
        for part in [p.strip() for p in opt.split(",") if p.strip()]:
            atoms.append(Atom("all", subj, _predicate(part), True))
        return atoms or None

    both = re.match(r"^(?:all|every)\s+(.+?)\s+(?:are|have|can|do|does)\s+both\s+(.+?)\s+and\s+(.+)$", opt)
    if both:
        subj = _subject(both.group(1))
        return [
            Atom("all", subj, _predicate(both.group(2)), True),
            Atom("all", subj, _predicate(both.group(3)), True),
        ]
    return None


def solve_mcq_semantic(question: str, premises: list[Any], options: dict[str, str]) -> SemanticResult | None:
    """Solve educational MCQ questions using semantic atom containment and forward rules.

    Args:
        question: The MCQ question text.
        premises: List of premise objects.
        options: Dict of option labels to option text.

    Returns:
        The matched SemanticResult or None.
    """
    facts, rules = _parse_premises(premises)
    if not facts and not rules:
        return None

    scored: list[tuple[int, str, list[Any], str]] = []
    for label, text in options.items():
        atoms = _split_option_claims(question, text)
        if not atoms:
            continue
        support: list[Any] = []
        ok = True
        for atom in atoms:
            proven, atom_support = _verify_atom(atom, facts, rules)
            if not proven:
                ok = False
                break
            support.extend(atom_support)
        if ok:
            scored.append((len(atoms), label, list(dict.fromkeys(support)), "semantic option verification"))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = [item for item in scored if item[0] == scored[0][0]]
    if len(best) != 1:
        return None
    _count, label, support, reason = best[0]
    return SemanticResult(label, support, reason)
