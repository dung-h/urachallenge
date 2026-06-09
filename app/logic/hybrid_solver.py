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
    text: str = ""


@dataclass(frozen=True)
class _Fact:
    premise_id: str
    subject: str
    predicate: str
    text: str = ""


@dataclass(frozen=True)
class _Translation:
    rules: list[_Rule]
    facts: list[_Fact]
    unsupported: list[str]


@dataclass(frozen=True)
class _ReasoningPath:
    fact: _Fact
    rules: list[_Rule]


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


def _human_subject(subject: str) -> str:
    parts = [part for part in subject.split("_") if part]
    if not parts:
        return "the subject"
    return " ".join(part.upper() if len(part) == 1 else part.capitalize() for part in parts)


def _human_predicate(predicate: str) -> str:
    if predicate.startswith("eligible_"):
        rest = predicate.removeprefix("eligible_").replace("_", " ").strip()
        return f"eligible for the {rest}" if rest else "eligible"
    if predicate.startswith("submit_"):
        rest = predicate.removeprefix("submit_").replace("_", " ").strip()
        return f"submitting a {rest}" if rest else "submitting the required item"
    text = predicate.replace("_", " ").strip()
    return text or "the queried property"


def _article(noun_phrase: str) -> str:
    first = noun_phrase[:1].lower()
    return "an" if first in {"a", "e", "i", "o", "u"} else "a"


def _claim_phrase(subject: str, predicate: str) -> str:
    subject_text = _human_subject(subject)
    predicate_text = _human_predicate(predicate)
    if predicate.startswith("eligible_"):
        return f"{subject_text} is {predicate_text}"
    if predicate in {"bird", "mammal", "animal", "machine", "robot", "student", "teacher", "applicant"}:
        return f"{subject_text} is {_article(predicate_text)} {predicate_text}"
    return f"{subject_text} satisfies {predicate_text}"


def _gerund_claim_phrase(subject: str, predicate: str) -> str:
    subject_text = _human_subject(subject)
    predicate_text = _human_predicate(predicate)
    if predicate.startswith("eligible_"):
        return f"{subject_text} being {predicate_text}"
    return f"{subject_text} satisfying {predicate_text}"


def _infinitive_claim_phrase(subject: str, predicate: str) -> str:
    subject_text = _human_subject(subject)
    predicate_text = _human_predicate(predicate)
    if predicate.startswith("eligible_"):
        return f"{subject_text} to be {predicate_text}"
    return f"{subject_text} to satisfy {predicate_text}"


def _premise_step(premise_id: str, text: str) -> str:
    return f"{premise_id} says: {text}."


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
            text=cleaned,
        )

    only_if = re.match(r"^(.+?)\s+only if\s+(.+)$", low)
    if only_if:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(only_if.group(1)),
            consequent=_predicate(only_if.group(2)),
            kind="implies",
            direction="necessary_condition",
            text=cleaned,
        )

    if_then = re.match(r"^if\s+(.+?),?\s+then\s+(.+)$", low)
    if if_then:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(if_then.group(1)),
            consequent=_predicate(if_then.group(2)),
            kind="implies",
            text=cleaned,
        )

    all_are = re.match(r"^all\s+(.+?)\s+are\s+(.+)$", low)
    if all_are:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(all_are.group(1)),
            consequent=_predicate(all_are.group(2)),
            kind="implies",
            text=cleaned,
        )

    no_are = re.match(r"^no\s+(.+?)\s+are\s+(.+)$", low)
    if no_are:
        return _Rule(
            premise_id=premise_id,
            antecedent=_predicate(no_are.group(1)),
            consequent=_predicate(no_are.group(2)),
            kind="implies_not",
            text=cleaned,
        )
    return None


def _parse_fact(premise_id: str, text: str) -> _Fact | None:
    cleaned = _clean(text)
    low = cleaned.lower()
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9 _-]*?)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+)$", low)
    if match:
        return _Fact(
            premise_id=premise_id,
            subject=_subject(match.group(1)),
            predicate=_predicate(match.group(2)),
            text=cleaned,
        )

    match = re.match(r"^([a-zA-Z][a-zA-Z0-9 _-]*?)\s+(has|have|submitted|submits|meets|completed|paid|needs?|requires?)\s+(.+)$", low)
    if match:
        return _Fact(
            premise_id=premise_id,
            subject=_subject(match.group(1)),
            predicate=_predicate(f"{match.group(2)} {match.group(3)}"),
            text=cleaned,
        )
    return None


def _find_direct_chain(
    translated: _Translation,
    query_subject: str,
    query_predicate: str,
    *,
    negative: bool = False,
) -> tuple[_Rule, _Fact] | None:
    expected_kind = "implies_not" if negative else "implies"
    for rule in translated.rules:
        if rule.kind != expected_kind or rule.consequent != query_predicate:
            continue
        for fact in translated.facts:
            if fact.subject == query_subject and fact.predicate == rule.antecedent:
                return rule, fact
    return None


def _find_reasoning_path(
    translated: _Translation,
    query_subject: str,
    query_predicate: str,
    *,
    negative: bool = False,
) -> _ReasoningPath | None:
    queue: list[tuple[str, _Fact, list[_Rule]]] = [
        (fact.predicate, fact, []) for fact in translated.facts if fact.subject == query_subject
    ]
    visited = {predicate for predicate, _, _ in queue}

    while queue:
        current_predicate, fact, rules = queue.pop(0)
        if not negative and current_predicate == query_predicate:
            return _ReasoningPath(fact=fact, rules=rules)

        for rule in translated.rules:
            if rule.antecedent != current_predicate:
                continue
            next_rules = [*rules, rule]
            if negative and rule.kind == "implies_not" and rule.consequent == query_predicate:
                return _ReasoningPath(fact=fact, rules=next_rules)
            if rule.kind != "implies":
                continue
            if not negative and rule.consequent == query_predicate:
                return _ReasoningPath(fact=fact, rules=next_rules)
            if rule.consequent not in visited:
                visited.add(rule.consequent)
                queue.append((rule.consequent, fact, next_rules))
    return None


def _path_explanation(
    path: _ReasoningPath,
    query_subject: str,
    query_predicate: str,
    *,
    negative: bool = False,
) -> str:
    subject = _human_subject(query_subject)
    sentences = [_premise_step(path.fact.premise_id, path.fact.text)]
    current_predicate = path.fact.predicate

    for rule in path.rules:
        sentences.append(_premise_step(rule.premise_id, rule.text))
        if rule.kind == "implies_not":
            sentences.append(
                f"Applying {rule.premise_id} to {subject} rules out that "
                f"{_claim_phrase(query_subject, rule.consequent)}."
            )
            continue
        current_predicate = rule.consequent
        sentences.append(
            f"Applying {rule.premise_id} to {subject} gives that "
            f"{_claim_phrase(query_subject, current_predicate)}."
        )

    if not path.rules:
        sentences.append(f"The premise directly states that {_claim_phrase(query_subject, query_predicate)}.")
    sentences.append(f"Therefore the answer is {'no' if negative else 'yes'}.")
    return " ".join(sentences)


def _find_required_condition_gap(
    translated: _Translation,
    query_subject: str,
    query_predicate: str,
) -> tuple[_Rule, _Fact | None] | None:
    for rule in translated.rules:
        if rule.direction != "necessary_condition" or rule.antecedent != query_predicate:
            continue
        matching_fact = next(
            (
                fact
                for fact in translated.facts
                if fact.subject == query_subject and fact.predicate == rule.consequent
            ),
            None,
        )
        return rule, matching_fact
    return None


def _build_reasoning_explanation(
    answer: str,
    translated: _Translation,
    query_subject: str,
    query_predicate: str,
) -> str:
    subject = _human_subject(query_subject)
    query_claim = _claim_phrase(query_subject, query_predicate)

    if answer == "yes":
        path = _find_reasoning_path(translated, query_subject, query_predicate)
        if path:
            return _path_explanation(path, query_subject, query_predicate)
        chain = _find_direct_chain(translated, query_subject, query_predicate)
        if chain:
            rule, fact = chain
            return (
                f"{_premise_step(rule.premise_id, rule.text)} "
                f"{_premise_step(fact.premise_id, fact.text)} "
                f"Applying {rule.premise_id} to {subject} gives that {query_claim}. "
                "Therefore the answer is yes."
            )

    if answer == "no":
        path = _find_reasoning_path(translated, query_subject, query_predicate, negative=True)
        if path:
            return _path_explanation(path, query_subject, query_predicate, negative=True)
        chain = _find_direct_chain(translated, query_subject, query_predicate, negative=True)
        if chain:
            rule, fact = chain
            return (
                f"{_premise_step(rule.premise_id, rule.text)} "
                f"{_premise_step(fact.premise_id, fact.text)} "
                f"Applying {rule.premise_id} to {subject} rules out that {query_claim}. "
                "Therefore the answer is no."
            )

    if answer == "unknown":
        gap = _find_required_condition_gap(translated, query_subject, query_predicate)
        if gap:
            rule, fact = gap
            fact_text = f" {_premise_step(fact.premise_id, fact.text)}" if fact else ""
            necessary_condition = _human_predicate(rule.consequent)
            return (
                f"{_premise_step(rule.premise_id, rule.text)}{fact_text} "
                f"This means {_gerund_claim_phrase(query_subject, query_predicate)} would require {necessary_condition}, "
                f"but it does not say that {necessary_condition} is enough for {_infinitive_claim_phrase(query_subject, query_predicate)}. "
                "To answer yes, the premises would need a rule in that opposite direction. "
                "No such rule is given, so the answer is unknown."
            )

    premise_steps = " ".join(_premise_step(rule.premise_id, rule.text) for rule in translated.rules)
    premise_steps = " ".join(
        step
        for step in [
            premise_steps,
            " ".join(_premise_step(fact.premise_id, fact.text) for fact in translated.facts),
        ]
        if step
    )
    if answer == "unknown":
        return (
            f"{premise_steps} The premises do not provide a complete chain to prove or disprove "
            f"that {query_claim}. Therefore the answer is unknown."
        )
    if answer == "no":
        return f"{premise_steps} The premises rule out that {query_claim}. Therefore the answer is no."
    return (
        f"{premise_steps} The supported premise chain determines that {query_claim}. "
        f"Therefore the answer is {answer}."
    )


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
        explanation = _build_reasoning_explanation(answer, translated, query_subject, query_predicate)
    elif entails_no and not entails_yes:
        answer = "no"
        status = "contradicted"
        confidence = 0.86
        explanation = _build_reasoning_explanation(answer, translated, query_subject, query_predicate)
    else:
        answer = "unknown"
        status = "not_entailed"
        confidence = 0.70
        explanation = _build_reasoning_explanation(answer, translated, query_subject, query_predicate)

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
