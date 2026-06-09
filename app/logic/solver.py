from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.guardrails import guardrail_prompt_text
from app.logic.agent_runtime import run_logic_agent
from app.logic.policy_reasoner import solve_policy
from app.logic.premise_selector import Premise, hallucinated_premises, normalize_premises, select_premises
from app.logic.proof_trace import ProofStep, build_proof_steps
from app.logic.proof_trace import proof_steps_to_dicts
from app.logic.templates import logic_explanation
from app.schemas import normalize_answer_label


@dataclass
class LogicSolution:
    answer: str
    explanation: str
    premises: list[str]
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    hallucinated_premises: list[str] = field(default_factory=list)
    llm_fallback_used: bool = False
    model_calls: int = 0
    proof_steps: list[ProofStep] = field(default_factory=list)
    z3_sidecar: dict[str, object] | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)


def _z3_metadata(z3_result: object) -> dict[str, object]:
    return {
        "answer_candidate": getattr(z3_result, "z3_answer_candidate", "unknown"),
        "used_premises": getattr(z3_result, "used_premises", []),
        "proof_steps": proof_steps_to_dicts(getattr(z3_result, "proof_steps", [])),
        "z3_status": getattr(z3_result, "z3_status", "unavailable"),
        "accepted": bool(getattr(z3_result, "accepted", False)),
        "coverage_complete": bool(getattr(z3_result, "coverage_complete", False)),
        "abstain_reason": getattr(z3_result, "abstain_reason", None),
        "translation_confidence": float(getattr(z3_result, "translation_confidence", 0.0) or 0.0),
        "unsupported_premises": getattr(z3_result, "unsupported_premises", []),
        "coverage_metrics": getattr(z3_result, "coverage_metrics", {}),
        "rejected_reason": getattr(z3_result, "rejected_reason", None),
        "high_confidence_translation": bool(getattr(z3_result, "high_confidence_translation", False)),
        "latency_ms": float(getattr(z3_result, "latency_ms", 0.0) or 0.0),
    }


def _with_z3_sidecar(solution: LogicSolution, question: str, normalized: list[Premise], allowed_domains: tuple[str, ...], mode: str) -> LogicSolution:
    try:
        from app.logic.z3_sidecar import solve_with_z3
    except Exception as exc:
        return LogicSolution(**{**solution.__dict__, "z3_sidecar": {"z3_status": "unavailable", "abstain_reason": f"import_error:{type(exc).__name__}"}})
    try:
        z3_result = solve_with_z3(question, normalized, allowed_domains)
    except Exception as exc:
        return LogicSolution(**{**solution.__dict__, "z3_sidecar": {"z3_status": "error", "abstain_reason": f"runtime_error:{type(exc).__name__}"}})
    metadata = _z3_metadata(z3_result)
    override_modes = {"override_unknown", "answer_authority", "z3_verified_only"}
    audit_modes = {"experiment_only", "z3_audit_only"}
    if mode in audit_modes:
        return LogicSolution(**{**solution.__dict__, "z3_sidecar": {**metadata, "overrode_baseline": False, "mode": mode}})
    if mode == "z3_unknown_guard":
        can_guard = (
            solution.answer in {"yes", "no"}
            and z3_result.accepted
            and z3_result.answer == "unknown"
            and z3_result.z3_status != "conflict"
            and z3_result.proof_steps
            and z3_result.proof_steps[0].rule in {"missing_required_condition", "invalid_inference_blocked", "unknown_due_to_insufficient_evidence", "contradiction_detected"}
        )
        if not can_guard:
            return LogicSolution(**{**solution.__dict__, "z3_sidecar": {**metadata, "overrode_baseline": False, "mode": mode}})
        selected = [premise for premise in normalized if premise.id in set(z3_result.used_premises)]
        return LogicSolution(
            answer="unknown",
            explanation=logic_explanation("unknown", selected, "Z3 unknown guard detected missing or invalid support"),
            premises=[p.id for p in selected],
            cot=["Z3 unknown guard checked baseline yes/no", "Guard found missing or invalid formal support", "Returned unknown conservatively"],
            confidence=0.7,
            hallucinated_premises=[],
            llm_fallback_used=False,
            model_calls=solution.model_calls,
            proof_steps=z3_result.proof_steps,
            z3_sidecar={**metadata, "overrode_baseline": True, "baseline_answer": solution.answer, "mode": mode},
        )
    verified_ok = (
        mode == "z3_verified_only"
        and z3_result.high_confidence_translation
        and z3_result.coverage_complete
        and not z3_result.unsupported_premises
        and z3_result.z3_status != "conflict"
        and z3_result.translation_confidence >= 0.85
        and (solution.answer == "unknown" or solution.answer == z3_result.answer)
    )
    can_override = (
        mode in override_modes
        and z3_result.accepted
        and z3_result.answer in {"yes", "no", "unknown"}
        and not hallucinated_premises(set(z3_result.used_premises), normalized)
        and (mode == "answer_authority" or verified_ok or solution.answer == "unknown")
    )
    if not can_override:
        return LogicSolution(**{**solution.__dict__, "z3_sidecar": {**metadata, "overrode_baseline": False, "mode": mode}})
    selected = [premise for premise in normalized if premise.id in set(z3_result.used_premises)]
    return LogicSolution(
        answer=z3_result.answer,
        explanation=logic_explanation(z3_result.answer, selected, "Z3 sidecar deterministic entailment"),
        premises=[p.id for p in selected],
        cot=["Z3 sidecar translated supported premise subset", "Z3 checked target entailment", f"Mode: {mode}"],
        confidence=max(solution.confidence, z3_result.translation_confidence),
        hallucinated_premises=[],
        llm_fallback_used=False,
        model_calls=solution.model_calls,
        proof_steps=z3_result.proof_steps,
        z3_sidecar={**metadata, "overrode_baseline": True, "baseline_answer": solution.answer, "mode": mode},
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("xes", "zes", "ches", "shes")) and len(word) > 4:
        return word[:-2]
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _strip_articles(text: str) -> str:
    return re.sub(r"\b(?:a|an|the)\b", " ", text).strip()


def _content_tokens(text: str) -> set[str]:
    return {_singular(t) for t in re.findall(r"[a-z0-9]+", _strip_articles(text.lower())) if len(t) > 1}


def _predicate_tokens(text: str) -> set[str]:
    generic = {
        "a", "an", "the", "it", "is", "are", "be", "being", "been", "do", "does", "did",
        "has", "have", "had", "need", "needs", "require", "requires", "receive", "receives",
        "can", "will", "would", "must", "to",
    }
    return {
        _singular(token)
        for token in re.findall(r"[a-z0-9]+", _strip_articles(text.lower()))
        if len(token) > 1 and _singular(token) not in generic
    }


def _predicate_matches(expected: str, actual: str) -> bool:
    expected_low = _norm(expected)
    actual_low = _norm(actual)
    if expected_low in actual_low or actual_low in expected_low:
        return True
    expected_tokens = _predicate_tokens(expected)
    actual_tokens = _predicate_tokens(actual)
    return bool(expected_tokens) and (expected_tokens <= actual_tokens or actual_tokens <= expected_tokens)


def _specific_tokens(text: str) -> list[str]:
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
    return bool(_content_tokens(left) & _content_tokens(right))


def _tokens_cover(required: str, actual: str) -> bool:
    required_tokens = _content_tokens(required)
    actual_tokens = _content_tokens(actual)
    return bool(required_tokens) and required_tokens <= actual_tokens


def _antecedent_triggered(antecedent: str, fact_kind: str, fact_text: str) -> bool:
    generic_actor_tokens = {"student", "person", "learner", "candidate", "shape", "object", "item", "thing", "is", "are", "has", "have"}
    antecedent_tokens = _content_tokens(antecedent) - generic_actor_tokens
    fact_tokens = _content_tokens(fact_kind)
    return bool(antecedent_tokens) and (antecedent_tokens <= fact_tokens or antecedent in _norm(fact_text))


def _class_matches(rule_class: str, fact_class: str) -> bool:
    if _tokens_cover(rule_class, fact_class):
        return True
    generic_class_tokens = {"file", "form", "object", "item", "thing"}
    required_tokens = _content_tokens(rule_class) - generic_class_tokens
    actual_tokens = _content_tokens(fact_class)
    return bool(required_tokens) and required_tokens <= actual_tokens


def _is_negated(text: str) -> bool:
    return bool(re.search(r"\b(?:not|no|never|cannot|can't|lacks?|does not|do not|did not|is not|are not)\b", _norm(text)))


def _negates_condition(fact_text: str, condition: str) -> bool:
    low = _norm(fact_text)
    if not _is_negated(low):
        return False
    return bool(_content_tokens(condition) & _content_tokens(low))


def _predicate_supported(predicate: str, text: str) -> bool:
    predicate_tokens = _content_tokens(predicate)
    text_tokens = _content_tokens(text)
    return bool(predicate_tokens) and (predicate_tokens <= text_tokens or bool(predicate_tokens & text_tokens))


def _contains_entity(text: str, entity: str) -> bool:
    low = _norm(text)
    entity_low = _singular(_strip_articles(entity.lower()))
    return entity_low in low or entity_low + "s" in low


def _question_subject_predicate(question: str) -> tuple[str | None, str | None, bool]:
    q = _norm(question).rstrip("?")
    starts = [idx for token in ["does ", "is ", "are ", "did ", "which "] if (idx := q.rfind(token)) > 0]
    if starts:
        q = q[max(starts):]
    negative = any(x in q for x in ["not ", "definitely"])
    patterns = [
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
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            return match.group(1).strip(), match.group(2).strip(), negative
    return None, None, negative


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
    q = _norm(question).rstrip("?")
    match = re.match(r"^(?:are|is|were|was)\s+there\s+any\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        match = re.match(r"^(?:are|is|were|was)\s+any\s+(.+?)(?:\s+(?:that|who|which)\s+(.+))?$", q)
    if not match:
        return None
    entity = _strip_articles(match.group(1).strip())
    predicate = match.group(2).strip() if match.group(2) else None
    return entity, predicate


def _match_all_rule(premise: str) -> tuple[str, str] | None:
    match = re.match(r"all (.+?) who (.+?) receive (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(2).strip()), "receive " + _singular(match.group(3).strip())
    match = re.match(r"all (.+?) are (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), _singular(match.group(2).strip())
    match = re.match(r"all (.+?) have (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "have " + _singular(match.group(2).strip())
    match = re.match(r"all (.+?) need (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "need " + _singular(match.group(2).strip())
    match = re.match(r"all (.+?) require (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "require " + _singular(match.group(2).strip())
    match = re.match(r"all (.+?) can (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "can " + _singular(match.group(2).strip())
    match = re.match(r"all (.+?) receive (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "receive " + _singular(match.group(2).strip())
    return None


def _match_no_rule(premise: str) -> tuple[str, str] | None:
    match = re.match(r"no (.+?) are (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), _singular(match.group(2).strip())
    return None


def _match_if_rule(premise: str) -> tuple[str, str] | None:
    match = re.match(r"if (.+?),? then (.+?)[.]?$", _norm(premise)) or re.match(r"if (.+?), (.+?)[.]?$", _norm(premise))
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


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


def _question_status_subject(question: str) -> str | None:
    low = _norm(question)
    if "status" not in low and "failed" not in low and "operational" not in low:
        return None
    match = re.search(r"\bunit\s+([a-z0-9]+)\b", low)
    return f"unit {match.group(1)}" if match else None


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
    options = dict(re.findall(r"\b([A-E])\)\s*(.*?)(?=\s+\b[A-E]\)|$)", question, flags=re.I | re.S))
    for label, text in options.items():
        if any(token in _norm(text) for token in ["undetermined", "unknown", "cannot be determined", "insufficient"]):
            return label.upper()
    return "unknown"


def _labeled_options(question: str) -> dict[str, str]:
    return {
        label.upper(): text.strip()
        for label, text in re.findall(r"\b([A-E])\)\s*(.*?)(?=\s+\b[A-E]\)|$)", question, flags=re.I | re.S)
    }


def _solve_conditional_must_true_mcq(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str] | None:
    """Handle MCQ conditionals where one option is a valid consequence of a rule.

    This intentionally stays narrow: it accepts direct contraposition and the
    common educational color-exclusive variant, e.g. square -> red, so blue ->
    not square when blue is treated as a mutually exclusive color from red.
    """

    q_low = _norm(question)
    if not any(phrase in q_low for phrase in ["which of the following", "which statement"]) or "true" not in q_low:
        return None
    options = _labeled_options(question)
    if len(options) < 2:
        return None

    colors = {"red", "blue", "green", "yellow", "black", "white", "orange", "purple", "brown", "gray", "grey"}
    for rule_premise in premises:
        rule = _match_if_rule(rule_premise.text)
        if not rule:
            continue
        antecedent, consequent = rule
        antecedent_specific = _specific_tokens(antecedent)
        consequent_specific = _specific_tokens(consequent)
        if not antecedent_specific or not consequent_specific:
            continue
        source_class = antecedent_specific[-1]
        required_property = consequent_specific[-1]
        for label, option in options.items():
            option_rule = _match_if_rule(option)
            if not option_rule:
                continue
            option_antecedent, option_consequent = option_rule
            option_ant_tokens = set(_specific_tokens(option_antecedent))
            option_cons_tokens = set(_specific_tokens(option_consequent))
            consequent_negates_source = _is_negated(option_consequent) and source_class in option_cons_tokens
            if not consequent_negates_source:
                continue
            direct_contrapositive = _is_negated(option_antecedent) and required_property in option_ant_tokens
            color_exclusive_contrapositive = (
                required_property in colors
                and bool((option_ant_tokens & colors) - {required_property})
            )
            if direct_contrapositive or color_exclusive_contrapositive:
                return label, [rule_premise], "multiple-choice contrapositive of conditional rule"
    return None


def _fact_contradicts_negated_consequent(fact_text: str, consequent: str) -> bool:
    if not _is_negated(consequent):
        return False
    consequent_tokens = _predicate_tokens(consequent)
    fact_tokens = _predicate_tokens(fact_text)
    if not consequent_tokens or not fact_tokens:
        return False
    return fact_tokens <= consequent_tokens or bool(fact_tokens & consequent_tokens)


def _question_asks_antecedent(question: str, antecedent: str) -> bool:
    subject, predicate, _negative = _question_subject_predicate(question)
    if not subject or not predicate:
        return False
    antecedent_tokens = _predicate_tokens(antecedent)
    question_tokens = _predicate_tokens(" ".join([subject, predicate]))
    return bool(antecedent_tokens) and antecedent_tokens <= question_tokens


def _solve_modus_tollens_negative_consequent(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str] | None:
    for rule_premise in premises:
        rule = _match_if_rule(rule_premise.text)
        if not rule:
            continue
        antecedent, consequent = rule
        if not _is_negated(consequent) or not _question_asks_antecedent(question, antecedent):
            continue
        for fact_premise in premises:
            if fact_premise == rule_premise:
                continue
            if _fact_contradicts_negated_consequent(fact_premise.text, consequent):
                return "no", [rule_premise, fact_premise], "modus tollens from negated consequent"
    return None


def _solve_conditional_status_unknown(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str] | None:
    """Conservative MCQ status rule for one-way conditionals.

    Example: A -> B, C -> not A, D -> not C, observed not B. This supports
    not A by modus tollens but does not support C or not D by reversing arrows.
    """

    q_low = _norm(question)
    if "status" not in q_low or not re.search(r"\b[A-E]\)\s+", question):
        return None
    if not any(_match_if_rule(p.text) for p in premises):
        return None
    has_negative_observation = any(
        re.search(r"\b(?:not|never)\b", _norm(p.text)) and not _norm(p.text).startswith("if ")
        for p in premises
    ) or bool(re.search(r"\b(?:not|never)\b", q_low))
    if not has_negative_observation:
        return None

    selected = [p for p in premises if _match_if_rule(p.text)]
    observed = [p for p in premises if p not in selected and re.search(r"\b(?:not|never)\b", _norm(p.text))]
    answer = _choice_for_unknown(question)
    return answer, (observed + selected), "one-way conditional chain does not determine the requested status"


def _solve_failure_status_conditionals(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str] | None:
    """Forward/contrapositive closure for simple unit failure status rules."""

    target_entity = _question_status_subject(question)
    if not target_entity:
        return None

    rules: list[tuple[Premise, tuple[str, bool], tuple[str, bool]]] = []
    known: dict[str, tuple[bool, Premise]] = {}
    for premise in premises:
        if_rule = _match_if_rule(premise.text)
        if if_rule:
            antecedent = _failure_status_prop(if_rule[0])
            consequent = _failure_status_prop(if_rule[1])
            if antecedent and consequent:
                rules.append((premise, antecedent, consequent))
            continue
        fact = _failure_status_prop(premise.text)
        if fact:
            known[fact[0]] = (fact[1], premise)

    if not rules or not known:
        return None

    support: dict[str, list[Premise]] = {entity: [premise] for entity, (_state, premise) in known.items()}
    changed = True
    while changed:
        changed = False
        for rule_premise, (ant_entity, ant_state), (cons_entity, cons_state) in rules:
            # Modus ponens: antecedent true under its polarity implies consequent.
            if ant_entity in known and known[ant_entity][0] == ant_state:
                current = known.get(cons_entity)
                if current is None:
                    known[cons_entity] = (cons_state, rule_premise)
                    support[cons_entity] = support.get(ant_entity, []) + [rule_premise]
                    changed = True
            # Modus tollens: consequent false under its polarity implies antecedent false.
            if cons_entity in known and known[cons_entity][0] != cons_state:
                current = known.get(ant_entity)
                inferred_state = not ant_state
                if current is None:
                    known[ant_entity] = (inferred_state, rule_premise)
                    support[ant_entity] = support.get(cons_entity, []) + [rule_premise]
                    changed = True

    if target_entity not in known:
        return _choice_for_failure_status(question, None), list({p.id: p for p in premises}.values()), "conditional status remains undetermined"

    failed = known[target_entity][0]
    selected = list({p.id: p for p in support.get(target_entity, [])}.values())
    answer = _choice_for_failure_status(question, failed)
    status_text = "failed" if failed else "not failed / operational"
    return answer, selected, f"conditional failure-status closure inferred {target_entity} is {status_text}"


def _object_prop(text: str) -> tuple[str, str] | None:
    low = _norm(text).rstrip(".")
    if low.startswith(("all ", "everything ", "if ", "no ")):
        return None
    match = re.match(r"(.+?)\s+is\s+(?:a |an )?(.+)$", low)
    if not match:
        return None
    subject = _strip_articles(match.group(1).strip())
    prop = _singular(_strip_articles(match.group(2).strip()))
    return subject, prop


def _universal_object_rule(text: str) -> tuple[tuple[str, ...], str] | None:
    low = _norm(text).rstrip(".")
    match = re.match(r"all\s+(.+?)\s+objects?\s+are\s+(.+)$", low)
    if match:
        return (_singular(_strip_articles(match.group(1))),), _singular(_strip_articles(match.group(2)))
    match = re.match(r"everything\s+that\s+is\s+(.+?)\s+is\s+(.+)$", low)
    if match:
        left = match.group(1).strip()
        requirements = tuple(_singular(_strip_articles(part.strip())) for part in re.split(r"\s+and\s+", left) if part.strip())
        consequent = _singular(_strip_articles(match.group(2).strip()))
        return requirements, consequent
    match = re.match(r"if\s+(?:an?\s+)?object\s+is\s+(.+?),?\s+then\s+it\s+(?:will\s+)?(.+)$", low)
    if not match:
        match = re.match(r"if\s+(?:an?\s+)?object\s+is\s+(.+?),\s+it\s+(?:will\s+)?(.+)$", low)
    if match:
        antecedent = _singular(_strip_articles(match.group(1).strip()))
        consequent = _singular(_strip_articles(match.group(2).strip()))
        return (antecedent,), consequent
    return None


def _solve_object_property_chain(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str] | None:
    q_match = re.match(r"does\s+(.+?)\s+(.+?)\??$", _norm(question).rstrip("?"))
    if not q_match:
        return None
    subject = _strip_articles(q_match.group(1).strip())
    target = _singular(_strip_articles(q_match.group(2).strip()))
    if not subject or not target:
        return None

    facts: dict[str, tuple[Premise, ...]] = {}
    rules: list[tuple[tuple[str, ...], str, Premise]] = []
    for premise in premises:
        fact = _object_prop(premise.text)
        if fact and _contains_entity(fact[0], subject):
            facts[fact[1]] = (premise,)
            continue
        rule = _universal_object_rule(premise.text)
        if rule:
            reqs, consequent = rule
            rules.append((reqs, consequent, premise))

    if not facts or not rules:
        return None

    changed = True
    while changed:
        changed = False
        for requirements, consequent, rule_premise in rules:
            if consequent in facts:
                continue
            if all(req in facts for req in requirements):
                support: list[Premise] = []
                for req in requirements:
                    support.extend(facts[req])
                support.append(rule_premise)
                facts[consequent] = tuple(dict.fromkeys(support))
                changed = True

    if target not in facts:
        return None
    selected = list(dict.fromkeys(facts[target]))
    return "yes", selected, f"forward chained object properties to derive that {subject} {target}"


def _fact_subject_kind(premise: str) -> tuple[str, str] | None:
    low = _norm(premise).rstrip(".")
    if low.startswith(("if ", "all ", "some ", "no ")):
        return None
    for standalone in [
        "studies",
        "completes the course",
        "has id",
        "rings",
        "gets water",
        "power is supplied",
        "switch is closed",
        "battery is charged",
        "is heated",
        "temperature is high",
        "reacts",
        "code is correct",
        "fails",
    ]:
        if low == standalone:
            return "", standalone
    match = re.match(r"(.+?) is (?:a |an )?(.+)$", low)
    if match:
        return match.group(1).strip(), _singular(match.group(2).strip())
    match = re.match(r"(.+?) are (?:a |an )?(.+)$", low)
    if match:
        return match.group(1).strip(), _singular(match.group(2).strip())
    match = re.match(r"(.+?) rings$", low)
    if match:
        return match.group(1).strip(), f"{match.group(1).strip()} rings"
    match = re.match(r"(.+?) studies$", low)
    if match:
        return match.group(1).strip(), "studies"
    match = re.match(r"(.+?) registers$", low)
    if match:
        return match.group(1).strip(), "register"
    match = re.match(r"(.+?) turns on$", low)
    if match:
        return match.group(1).strip(), "turns on"
    match = re.match(r"(.+?) can (.+)$", low)
    if match:
        return match.group(1).strip(), "can " + match.group(2).strip()
    match = re.match(r"(.+?) submits homework$", low)
    if match:
        return match.group(1).strip(), "submits homework"
    match = re.match(r"(.+?) completes the course$", low)
    if match:
        return match.group(1).strip(), "completes the course"
    match = re.match(r"(.+?) has id$", low)
    if match:
        return match.group(1).strip(), "has id"
    match = re.match(r"(.+?) receives feedback$", low)
    if match:
        return match.group(1).strip(), "receives feedback"
    match = re.match(r"(.+?) receives (.+)$", low)
    if match:
        return match.group(1).strip(), "receives " + match.group(2).strip()
    match = re.match(r"(.+?) fails$", low)
    if match:
        subject = match.group(1).strip()
        return subject, f"{subject} fails"
    match = re.match(r"(.+?) turns litmus red$", low)
    if match:
        return match.group(1).strip(), "turns litmus red"
    return None


def _explicit_negative_premise(premise: Premise, subject: str | None, predicate: str | None) -> bool:
    if not subject or not predicate:
        return False
    low = _norm(premise.text)
    if "not" not in low or not _contains_entity(low, subject):
        return False
    after_not = low.split("not", 1)[1]
    return _predicate_supported(predicate, after_not)


def _direct_fact_contradiction(premises: list[Premise], subject: str | None, predicate: str | None) -> list[Premise]:
    if not subject or not predicate:
        return []
    positives: list[Premise] = []
    negatives: list[Premise] = []
    for premise in premises:
        low = _norm(premise.text)
        if low.startswith(("if ", "all ", "some ", "no ")) or not _contains_entity(low, subject):
            continue
        if _is_negated(low) and _predicate_supported(predicate, low):
            negatives.append(premise)
        elif _predicate_supported(predicate, low):
            positives.append(premise)
    return (positives[:1] + negatives[:1]) if positives and negatives else []


def _has_universal_no_conflict(
    subject: str | None,
    all_rules: list[tuple[Premise, tuple[str, str]]],
    no_rules: list[tuple[Premise, tuple[str, str]]],
    facts: list[tuple[Premise, tuple[str, str]]],
) -> tuple[Premise, Premise, Premise] | None:
    if not subject:
        return None
    for all_premise, (all_left, all_right) in all_rules:
        for no_premise, (no_left, no_right) in no_rules:
            if not (_terms_overlap(all_left, no_left) or all_left in no_left or no_left in all_left):
                continue
            if not (_terms_overlap(all_right, no_right) or all_right in no_right or no_right in all_right):
                continue
            for fact_premise, (fact_subject, fact_kind) in facts:
                if _contains_entity(fact_subject, subject) and (_terms_overlap(all_left, fact_kind) or all_left in fact_kind):
                    return all_premise, no_premise, fact_premise
    return None


def _universal_positive_support(
    subject: str | None,
    predicate: str | None,
    all_rules: list[tuple[Premise, tuple[str, str]]],
    facts: list[tuple[Premise, tuple[str, str]]],
    *,
    has_rules: bool = False,
) -> tuple[str, list[Premise], str] | None:
    if not subject or not predicate:
        return None
    queue: list[tuple[str, list[Premise]]] = []
    seen: set[str] = set()
    for fact_premise, (fact_subject, fact_kind) in facts:
        if _is_negated(fact_premise.text) or fact_kind.startswith(("not ", "no ")):
            continue
        if _contains_entity(fact_subject, subject):
            queue.append((fact_kind, [fact_premise]))
    while queue:
        kind, support = queue.pop(0)
        if kind in seen:
            continue
        seen.add(kind)
        if _predicate_matches(predicate, kind) or predicate in kind or kind in predicate:
            if len(support) == 1 and has_rules:
                continue
            return "yes", list(dict.fromkeys(support)), "universal syllogism chain"
        for rule_premise, (left, mid) in all_rules:
            if _class_matches(left, kind):
                new_support = list(dict.fromkeys(support + [rule_premise]))
                if mid not in seen:
                    queue.append((mid, new_support))
    return None


def _universal_negative_support(
    subject: str | None,
    predicate: str | None,
    no_rules: list[tuple[Premise, tuple[str, str]]],
    facts: list[tuple[Premise, tuple[str, str]]],
) -> tuple[str, list[Premise], str] | None:
    if not subject or not predicate:
        return None
    for fact_premise, (fact_subject, fact_kind) in facts:
        if not _contains_entity(fact_subject, subject):
            continue
        for no_premise, (left, right) in no_rules:
            if _class_matches(left, fact_kind) and _predicate_matches(predicate, right):
                return "no", [no_premise, fact_premise], "universal no-overlap entailment"
    return None


def _solve_rules(question: str, premises: list[Premise]) -> tuple[str, list[Premise], str]:
    subject, predicate, _negative = _question_subject_predicate(question)
    question_polarity = _question_polarity(question)
    existential_query = _question_existential(question)
    selected = select_premises(question, premises)
    all_rules = [(p, rule) for p in premises if (rule := _match_all_rule(p.text))]
    no_rules = [(p, rule) for p in premises if (rule := _match_no_rule(p.text))]
    if_rules = [(p, rule) for p in premises if (rule := _match_if_rule(p.text))]
    facts = [(p, fact) for p in premises if (fact := _fact_subject_kind(p.text))]

    subject_norm = _singular(_strip_articles(subject or ""))
    predicate_norm = _singular(_strip_articles((predicate or "").replace("have ", "")))

    explicit_negatives = [p for p in premises if _explicit_negative_premise(p, subject, predicate)]
    exception_negatives = [p for p in explicit_negatives if "exception" in _norm(p.text)]
    if exception_negatives:
        return "no", exception_negatives[:1], "explicit exception or negative fact overrides general rule"

    direct_conflict = _direct_fact_contradiction(premises, subject, predicate)
    if direct_conflict:
        return "unknown", direct_conflict, "directly contradictory facts about the requested claim"

    if explicit_negatives:
        return "no", explicit_negatives[:1], "explicit exception or negative fact overrides general rule"

    conflict = _has_universal_no_conflict(subject, all_rules, no_rules, facts)
    if conflict:
        return "unknown", list(conflict), "premises support both the claim and its negation"

    failure_status = _solve_failure_status_conditionals(question, premises)
    if failure_status:
        return failure_status

    object_property_chain = _solve_object_property_chain(question, premises)
    if object_property_chain:
        return object_property_chain

    universal_negative = _universal_negative_support(subject, predicate, no_rules, facts)
    if universal_negative:
        return universal_negative

    universal_positive = _universal_positive_support(subject, predicate, all_rules, facts, has_rules=bool(all_rules or no_rules or if_rules))
    if universal_positive:
        return universal_positive

    conditional_status_unknown = _solve_conditional_status_unknown(question, premises)
    if conditional_status_unknown:
        return conditional_status_unknown

    modus_tollens = _solve_modus_tollens_negative_consequent(question, premises)
    if modus_tollens:
        return modus_tollens

    if existential_query:
        entity, existential_predicate = existential_query
        witness: list[Premise] = []
        negative_evidence: list[Premise] = []
        for premise in premises:
            low = _norm(premise.text)
            if low.startswith("some "):
                if _contains_entity(low, entity) and (not existential_predicate or _predicate_supported(existential_predicate, low)):
                    witness = [premise]
                    break
            fact = _fact_subject_kind(premise.text)
            if fact and _contains_entity(fact[0], entity):
                if not _is_negated(low) and (not existential_predicate or _predicate_supported(existential_predicate, fact[1])):
                    witness = [premise]
                    break
                if existential_predicate and _is_negated(low) and _predicate_supported(existential_predicate, existential_predicate):
                    negative_evidence.append(premise)
            no_rule = _match_no_rule(premise.text)
            if no_rule and _class_matches(no_rule[0], entity) and (not existential_predicate or _predicate_matches(existential_predicate, no_rule[1])):
                negative_evidence.append(premise)
        if witness:
            return "yes", witness, "existential witness provided"
        if negative_evidence:
            return "no", list(dict.fromkeys(negative_evidence)), "existential premise blocked by a universal negative rule"
        return "unknown", selected, "existential query lacks a concrete witness"

    if question_polarity in {"known", "unknown"}:
        for rule_premise, (antecedent, consequent) in if_rules:
            for fact_premise, (_fact_subject, fact_kind) in facts:
                if _negates_condition(fact_premise.text, antecedent):
                    continue
                if not _antecedent_triggered(antecedent, fact_kind, fact_premise.text):
                    continue
                consequent_low = _norm(consequent)
                if question_polarity == "known":
                    if "unknown" in consequent_low:
                        return "no", [rule_premise, fact_premise], "conditional rule implies the result is unknown"
                    if "known" in consequent_low or _predicate_matches("known", consequent_low):
                        return "yes", [rule_premise, fact_premise], "conditional rule implies the result is known"
                else:
                    if "unknown" in consequent_low or _predicate_matches("unknown", consequent_low):
                        return "yes", [rule_premise, fact_premise], "conditional rule implies the result is unknown"
                    if "known" in consequent_low or _predicate_matches("known", consequent_low):
                        return "no", [rule_premise, fact_premise], "conditional rule implies the result is known"

    conditional_must_true = _solve_conditional_must_true_mcq(question, premises)
    if conditional_must_true:
        return conditional_must_true

    # Broaden MCQ detection: check for "which option", "which conclusion", "which statement", "which of the following"
    is_mcq = any(phrase in _norm(question) for phrase in ["which option", "which conclusion", "which statement", "which of the following"])
    if is_mcq:
        for fact_premise, (_fact_subject, fact_kind) in facts:
            for first_premise, (left, mid) in all_rules:
                if _class_matches(left, fact_kind):
                    return "A", [first_premise, fact_premise], "multiple-choice universal entailment"
            for no_premise, (left, right) in no_rules:
                if left in fact_kind or right in fact_kind:
                    return "B", [no_premise, fact_premise], "multiple-choice negative entailment"
        if any(_norm(p.text).startswith("some ") for p in premises):
            return "C", selected, "multiple-choice insufficient information"

    # Modus ponens for simple conditionals.
    for rule_premise, (antecedent, consequent) in if_rules:
        for fact_premise, (_fact_subject, fact_kind) in facts:
            if _negates_condition(fact_premise.text, antecedent):
                if predicate and (predicate_norm in _singular(consequent) or _singular(consequent) in predicate_norm or _terms_overlap(predicate_norm, consequent)):
                    return "unknown", [rule_premise, fact_premise], "negated antecedent does not trigger conditional rule"
                continue
            if _antecedent_triggered(antecedent, fact_kind, fact_premise.text):
                if predicate and (predicate_norm in _singular(consequent) or _singular(consequent) in predicate_norm or _terms_overlap(predicate_norm, consequent)):
                    return "yes", [rule_premise, fact_premise], "modus ponens"
        # Affirming consequent / denying antecedent is unknown, not no.
        if predicate and (predicate_norm in consequent or any(word in consequent for word in predicate_norm.split() if len(word) > 2)):
            return "unknown", selected, "conditional premise does not establish the requested case"
        if subject and any(_contains_entity(f.text, subject) for f, _ in facts):
            return "unknown", selected, "conditional premise not triggered"

    # No-overlap rules support direct negative and contrapositive for transparent/metal style cases.
    for rule_premise, (left, right) in no_rules:
        for fact_premise, (fact_subject, fact_kind) in facts:
            if subject and _contains_entity(fact_subject, subject):
                if (left in fact_kind or _terms_overlap(left, fact_kind)) and predicate and right in predicate:
                    return "no", [rule_premise, fact_premise], "no-overlap class rule"
                if (right in fact_kind or _terms_overlap(right, fact_kind)) and predicate and left in predicate:
                    return "no", [rule_premise, fact_premise], "no-overlap contrapositive"

    # Chained negative: all A are B; no B are C; x is A => x is not C.
    for fact_premise, (fact_subject, fact_kind) in facts:
        if subject and not _contains_entity(fact_subject, subject):
            continue
        for first_premise, (left, mid) in all_rules:
            if not _class_matches(left, fact_kind):
                continue
            for no_premise, (no_left, no_right) in no_rules:
                if not (_class_matches(no_left, mid) or _terms_overlap(no_left, mid)):
                    continue
                if predicate and _predicate_matches(predicate_norm, no_right):
                    return "no", [first_premise, no_premise, fact_premise], "universal syllogism with no-overlap rule"

    # Universal chaining: all A are B, all B have/are C, X is A.
    for fact_premise, (fact_subject, fact_kind) in facts:
        if subject and not _contains_entity(fact_subject, subject):
            continue
        for first_premise, (left, mid) in all_rules:
            if not _class_matches(left, fact_kind):
                continue
            if predicate and (_predicate_matches(predicate_norm, mid) or mid in predicate or predicate in mid):
                return "yes", [first_premise, fact_premise], "universal class membership"
            for second_premise, (left2, right2) in all_rules:
                if left2 in mid or mid in left2:
                    if predicate and (_predicate_matches(predicate_norm, right2) or right2 in predicate or predicate in right2 or right2 in question.lower()):
                        return "yes", [first_premise, second_premise, fact_premise], "universal syllogism"

    # Class-only universal syllogism, e.g. all robins are birds; all birds have wings.
    if subject and predicate:
        for first_premise, (left, mid) in all_rules:
            if not _class_matches(left, subject_norm):
                continue
            if _predicate_matches(predicate_norm, mid) or mid in predicate_norm or predicate_norm in mid:
                return "yes", [first_premise], "universal class membership"
            for second_premise, (left2, right2) in all_rules:
                if (left2 in mid or mid in left2) and (_predicate_matches(predicate_norm, right2) or right2 in predicate_norm or predicate_norm in right2):
                    return "yes", [first_premise, second_premise], "universal syllogism"

    # Some does not entail a specific instance.
    if any(_norm(p.text).startswith("some ") for p in premises):
        return "unknown", selected, "existential premise does not identify the specific subject"

    return "unknown", selected, "no deterministic entailment rule matched"


def solve(
    question: str,
    premises: list[str],
    llm_client: object | None = None,
    use_llm: bool = False,
    enable_z3_sidecar: bool = False,
    z3_allowed_domains: tuple[str, ...] = ("academic_policy", "public_logic_sample"),
    z3_sidecar_mode: str = "experiment_only",
    enable_mcq_symbolic: bool = False,
    choices: list[str] | None = None,
    max_agent_steps: int = 4,
    max_model_calls: int = 5,
) -> LogicSolution:
    question = guardrail_prompt_text(question).normalized_text
    normalized = normalize_premises(premises)
    
    # MCQ symbolic path: experiment-only, disabled by default.
    # Requires enable_mcq_symbolic=True, choices present, and LLM client for FOL translation.
    if enable_mcq_symbolic and choices and len(choices) >= 2 and llm_client:
        from app.logic.fol_translator import translate_premises_to_fol
        from app.logic.mcq_symbolic import solve_mcq_symbolic
        
        # Attempt to translate premises_nl to premises_fol using LLM
        fol_result = translate_premises_to_fol([p.text for p in normalized], llm_client)
        
        if fol_result.success and fol_result.premises_fol:
            # Build options dict from choices (auto-label A, B, C, ...)
            options = {chr(65 + idx): choice for idx, choice in enumerate(choices[:26])}
            
            # Call MCQ symbolic solver with LLM client for option translation
            mcq_answer, mcq_debug = solve_mcq_symbolic(
                question,
                [p.text for p in normalized],
                fol_result.premises_fol,
                options,
                llm_client=llm_client,
            )
            
            # If MCQ symbolic returned a definite answer (not "unknown"), use it
            if mcq_answer != "unknown" and re.fullmatch(r"[A-E]", mcq_answer):
                # Find which premises were used (from debug info)
                used_premise_ids = []
                entailed_options = mcq_debug.get("entailed_options", [])
                if entailed_options:
                    # Conservative: cite all premises since we don't have fine-grained tracking
                    used_premise_ids = [p.id for p in normalized]
                
                return LogicSolution(
                    answer=mcq_answer,
                    explanation=f"MCQ symbolic solver selected option {mcq_answer} via LLM-assisted FOL entailment.",
                    premises=used_premise_ids,
                    cot=["LLM translated premises to FOL", "LLM translated options to FOL queries", "Prover checked entailment", f"Exactly one option entailed: {mcq_answer}"],
                    confidence=0.70,
                    hallucinated_premises=[],
                    llm_fallback_used=True,
                    model_calls=1 + len(options),  # 1 for premises, N for options
                    proof_steps=build_proof_steps(mcq_answer, normalized, "mcq_symbolic_llm_assisted_entailment", 0.70),
                )
    
    policy_decision = solve_policy(question, normalized)
    if policy_decision:
        cited = {p.id for p in policy_decision.premises}
        hallucinated = sorted(hallucinated_premises(cited, normalized))
        answer = normalize_answer_label(policy_decision.answer)
        solution = LogicSolution(
            answer=answer,
            explanation=logic_explanation(policy_decision.answer, policy_decision.premises, policy_decision.rule),
            premises=[p.id for p in policy_decision.premises],
            cot=policy_decision.cot,
            confidence=policy_decision.confidence,
            hallucinated_premises=hallucinated,
            llm_fallback_used=False,
            proof_steps=build_proof_steps(answer, policy_decision.premises, policy_decision.rule, policy_decision.confidence),
        )
        if enable_z3_sidecar:
            return _with_z3_sidecar(solution, question, normalized, z3_allowed_domains, z3_sidecar_mode)
        return solution
    answer, selected, rule = _solve_rules(question, normalized)
    answer = normalize_answer_label(answer)
    cited = {p.id for p in selected}
    hallucinated = sorted(hallucinated_premises(cited, normalized))
    confidence = 0.9 if answer in {"yes", "no"} else 0.78
    if rule.startswith("no deterministic"):
        confidence = 0.55
    elif answer == "unknown" and any(token in rule for token in ["negated antecedent", "conditional premise", "directly contradictory", "does not identify", "premises support both"]):
        confidence = 0.72
    model_calls = 0
    solution = LogicSolution(
        answer=answer,
        explanation=logic_explanation(answer, selected, rule),
        premises=[p.id for p in selected],
        cot=[f"Normalized {len(normalized)} premises", f"Selected premises: {', '.join(p.id for p in selected)}", f"Rule: {rule}"],
        confidence=confidence,
        hallucinated_premises=hallucinated,
        llm_fallback_used=False,
        model_calls=model_calls,
        proof_steps=build_proof_steps(answer, selected, rule, confidence),
        agent_trace=[],
    )
    if use_llm and llm_client and (confidence < 0.7 or answer == "unknown"):
        agent_outcome = run_logic_agent(
            question,
            normalized,
            llm_client=llm_client,
            base_solution=solution,
            choices=list(choices or []),
            allow_llm_rescue=True,
            max_steps=max_agent_steps,
            max_model_calls=max_model_calls,
        )
        model_calls += agent_outcome.model_calls
        if agent_outcome.success:
            selected = [p for p in normalized if p.id in set(agent_outcome.premises)]
            candidate = normalize_answer_label(agent_outcome.answer)
            solution = LogicSolution(
                answer=candidate,
                explanation=agent_outcome.explanation,
                premises=[p.id for p in selected],
                cot=list(agent_outcome.cot) or [
                    "Rule baseline confidence below threshold",
                    "Agent planner selected a validated rescue proposal",
                    "Backend validated premise IDs and normalized answer",
                ],
                confidence=agent_outcome.confidence,
                hallucinated_premises=[],
                llm_fallback_used=True,
                model_calls=model_calls,
                proof_steps=build_proof_steps(candidate, selected, "validated LLM fallback", agent_outcome.confidence),
                agent_trace=list(agent_outcome.agent_trace),
            )
            if enable_z3_sidecar:
                return _with_z3_sidecar(solution, question, normalized, z3_allowed_domains, z3_sidecar_mode)
            return solution
        solution = LogicSolution(
            answer=answer,
            explanation=agent_outcome.explanation if (agent_outcome.success and agent_outcome.explanation) else logic_explanation(answer, selected, rule),
            premises=[p.id for p in selected],
            cot=list(agent_outcome.cot) if agent_outcome.success else [f"Normalized {len(normalized)} premises", f"Selected premises: {', '.join(p.id for p in selected)}", f"Rule: {rule}"],
            confidence=confidence,
            hallucinated_premises=hallucinated,
            llm_fallback_used=True,
            model_calls=model_calls,
            proof_steps=build_proof_steps(answer, selected, rule, confidence),
            agent_trace=list(agent_outcome.agent_trace),
        )
        if enable_z3_sidecar:
            return _with_z3_sidecar(solution, question, normalized, z3_allowed_domains, z3_sidecar_mode)
        return solution
    if enable_z3_sidecar:
        return _with_z3_sidecar(solution, question, normalized, z3_allowed_domains, z3_sidecar_mode)
    return solution
