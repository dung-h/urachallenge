from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Literal


@dataclass(frozen=True)
class HybridResult:
    answer: str
    explanation: str
    method: str
    confidence: float
    z3_status: str = "unavailable"
    conclusion_fol: str = ""
    used_premises: list[str] = field(default_factory=list)
    unsupported_premises: list[str] = field(default_factory=list)
    translation_confidence: float = 0.0
    model_calls: int = 0


@dataclass(frozen=True)
class _Rule:
    premise_id: str
    antecedent: str
    consequent: str
    kind: Literal["implies", "implies_not"]
    direction: str = "sufficient"


@dataclass(frozen=True)
class _Fact:
    premise_id: str
    subject: str
    predicate: str


@dataclass(frozen=True)
class _Translation:
    rules: list[_Rule]
    facts: list[_Fact]
    unsupported: list[str]


_STOPWORDS = {"a", "an", "the", "is", "are", "be", "being", "been", "for", "to", "of"}


def _clean(text: str) -> str:
    text = text.strip().rstrip(".?!")
    text = re.sub(r"\s+", " ", text)
    return text


def _stem_token(token: str) -> str:
    if token == "eligibility":
        return "eligible"
    if token in {"submitted", "submitting", "submits"}:
        return "submit"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _predicate(text: str) -> str:
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(_stem_token(token))
    return "_".join(tokens)


def _subject(text: str) -> str:
    tokens = [_stem_token(t) for t in re.findall(r"[a-zA-Z0-9]+", text.lower())]
    return "_".join(tokens)


def _parse_question(question: str) -> tuple[str, str] | None:
    text = _clean(question)
    match = re.match(r"^(?:is|are|does|do|did)\s+(.+?)\s+(.+)$", text, re.I)
    if not match:
        return None
    subject = _subject(match.group(1))
    predicate = _predicate(match.group(2))
    if not subject or not predicate:
        return None
    return subject, predicate


def _parse_rule(premise_id: str, text: str) -> _Rule | None:
    cleaned = _clean(text)
    low = cleaned.lower()

    required = re.match(r"^(.+?)\s+requires?\s+(.+)$", low)
    if required:
        # "A requires B" means A -> B. B alone must not prove A.
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(required.group(1)),
            consequent=_predicate(required.group(2)),
            kind="implies",
            direction="necessary_condition",
        )

    only_if = re.match(r"^(.+?)\s+only if\s+(.+)$", low)
    if only_if:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(only_if.group(1)),
            consequent=_predicate(only_if.group(2)),
            kind="implies",
            direction="necessary_condition",
        )

    if_then = re.match(r"^if\s+(.+?),?\s+then\s+(.+)$", low)
    if if_then:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(if_then.group(1)),
            consequent=_predicate(if_then.group(2)),
            kind="implies",
        )

    all_are = re.match(r"^all\s+(.+?)\s+are\s+(.+)$", low)
    if all_are:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(all_are.group(1)),
            consequent=_predicate(all_are.group(2)),
            kind="implies",
        )

    no_are = re.match(r"^no\s+(.+?)\s+are\s+(.+)$", low)
    if no_are:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(no_are.group(1)),
            consequent=_predicate(no_are.group(2)),
            kind="implies_not",
        )
    return None


def _parse_fact(premise_id: str, text: str) -> _Fact | None:
    cleaned = _clean(text)
    low = cleaned.lower()
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9 _-]*?)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+)$", low)
    if match:
        return _Fact(premise_id=premise_id, subject=_subject(match.group(1)), predicate=_predicate(match.group(2)))

    match = re.match(r"^([a-zA-Z][a-zA-Z0-9 _-]*?)\s+(has|have|submitted|submits|meets|completed|paid|needs?|requires?)\s+(.+)$", low)
    if match:
        return _Fact(
            premise_id=premise_id,
            subject=_subject(match.group(1)),
            predicate=_predicate(f"{match.group(2)} {match.group(3)}"),
        )
    return None


def _translate_premises(premises: list[str]) -> _Translation:
    rules: list[_Rule] = []
    facts: list[_Fact] = []
    unsupported: list[str] = []
    for idx, premise in enumerate(premises, start=1):
        match = re.match(r"^(P\d+)\s*:\s*(.+)$", premise.strip(), re.I)
        premise_id = match.group(1).upper() if match else f"P{idx}"
        text = match.group(2).strip() if match else premise.strip()
        rule = _parse_rule(premise_id, text)
        if rule:
            rules.append(rule)
            continue
        fact = _parse_fact(premise_id, text)
        if fact:
            facts.append(fact)
            continue
        unsupported.append(premise_id)
    return _Translation(rules=rules, facts=facts, unsupported=unsupported)


def _bool_name(predicate: str, subject: str) -> str:
    return f"{predicate}__{subject}"


def _solve_with_z3(question: str, premises: list[str]) -> HybridResult:
    try:
        import z3
    except Exception:
        return HybridResult(
            answer="unknown",
            explanation="Hybrid symbolic solver is unavailable because z3 is not installed.",
            method="rules_to_z3_unavailable",
            confidence=0.0,
            z3_status="unavailable",
        )

    parsed_query = _parse_question(question)
    if not parsed_query:
        return HybridResult(
            answer="unknown",
            explanation="Hybrid symbolic solver abstained because the question could not be translated safely.",
            method="rules_to_z3_abstain",
            confidence=0.0,
            z3_status="abstained",
        )

    query_subject, query_predicate = parsed_query
    translated = _translate_premises(premises)
    if translated.unsupported or not (translated.rules or translated.facts):
        return HybridResult(
            answer="unknown",
            explanation="Hybrid symbolic solver abstained because not all premises could be translated safely.",
            method="rules_to_z3_abstain",
            confidence=0.0,
            z3_status="abstained",
            unsupported_premises=translated.unsupported,
        )

    start = time.perf_counter()
    subjects = {query_subject} | {fact.subject for fact in translated.facts}
    symbols: dict[str, object] = {}

    def atom(predicate: str, subject: str) -> object:
        key = _bool_name(predicate, subject)
        if key not in symbols:
            symbols[key] = z3.Bool(key)
        return symbols[key]

    assertions: list[object] = []
    for fact in translated.facts:
        assertions.append(atom(fact.predicate, fact.subject))
    for rule in translated.rules:
        for subject in subjects:
            left = atom(rule.antecedent, subject)
            right = atom(rule.consequent, subject)
            assertions.append(z3.Implies(left, z3.Not(right) if rule.kind == "implies_not" else right))

    query = atom(query_predicate, query_subject)
    solver = z3.Solver()
    solver.add(*assertions)
    solver.push()
    solver.add(z3.Not(query))
    entails_yes = solver.check() == z3.unsat
    solver.pop()
    solver.push()
    solver.add(query)
    entails_no = solver.check() == z3.unsat
    solver.pop()

    if entails_yes and not entails_no:
        answer = "yes"
        status = "entailed"
        confidence = 0.86
        explanation = "Hybrid solver translated safe premise patterns to Z3 and proved the queried claim."
    elif entails_no and not entails_yes:
        answer = "no"
        status = "contradicted"
        confidence = 0.86
        explanation = "Hybrid solver translated safe premise patterns to Z3 and proved the queried claim is false."
    else:
        answer = "unknown"
        status = "not_entailed"
        confidence = 0.70
        if any(rule.direction == "necessary_condition" for rule in translated.rules):
            explanation = (
                "Hybrid solver translated the required-condition premise in the safe direction and Z3 did not prove the target claim. "
                "A required condition being satisfied is not enough to establish the conclusion."
            )
        else:
            explanation = "Hybrid solver translated safe premise patterns to Z3, but the target claim was not entailed."

    used_ids = sorted({rule.premise_id for rule in translated.rules} | {fact.premise_id for fact in translated.facts})
    conclusion_fol = f"{query_predicate}({query_subject})"
    latency_ms = (time.perf_counter() - start) * 1000
    return HybridResult(
        answer=answer,
        explanation=explanation,
        method="rules_to_z3",
        confidence=confidence,
        z3_status=status,
        conclusion_fol=conclusion_fol,
        used_premises=used_ids,
        unsupported_premises=[],
        translation_confidence=0.90 if used_ids else 0.0,
    )


def solve_hybrid(
    question: str,
    premises: list[str],
    premises_fol: list[str] | None = None,
    api_url: str | None = None,
    model: str | None = None,
) -> HybridResult:
    """Safely translate clear NL logic patterns to Z3.

    `premises_fol`, `api_url`, and `model` are accepted for router compatibility.
    LLM translation is intentionally not trusted here yet; this first hybrid path
    only accepts deterministic translations with known-safe direction handling.
    """
    _ = (premises_fol, api_url, model)
    return _solve_with_z3(question, premises)
