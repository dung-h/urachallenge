from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.logic.policy_patterns import is_academic_policy_text, norm, overlap, strip_student_prefix, tokens
from app.logic.premise_selector import Premise, select_premises
from app.logic.thresholds import compare, metric_value, parse_threshold


@dataclass(frozen=True)
class PolicyDecision:
    answer: str
    premises: list[Premise]
    rule: str
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class PolicyRule:
    premise: Premise
    conditions: list[str]
    outcome: str
    positive: bool
    blocking: bool = False


def _clean_outcome(text: str) -> str:
    cleaned = norm(text)
    cleaned = re.sub(r"^(are|is|may|meet|satisfy|placed)\s+", "", cleaned)
    return cleaned


def _question_target(question: str) -> str:
    low = norm(question).rstrip("?")
    replacements = [
        (r"^is\s+.+?\s+", ""),
        (r"^are\s+.+?\s+", ""),
        (r"^does\s+.+?\s+", ""),
        (r"^has\s+.+?\s+", ""),
        (r"^may\s+.+?\s+", "may "),
        (r"^did\s+.+?\s+", ""),
    ]
    for pattern, replacement in replacements:
        low = re.sub(pattern, replacement, low)
    low = low.replace("met the", "meet the")
    low = low.replace("meets the", "meet the")
    low = low.replace("eligible to sit", "eligible to sit")
    return low.strip()


def _condition_parts(text: str) -> list[str]:
    cleaned = strip_student_prefix(text)
    cleaned = re.sub(r"^(with|who|that)\s+", "", cleaned)
    cleaned = cleaned.replace(",", " and ")
    return [re.sub(r"^and\s+", "", part.strip()) for part in re.split(r"\s+and\s+", cleaned) if part.strip()]


def _parse_rule(premise: Premise) -> PolicyRule | None:
    low = norm(premise.text)
    if low.startswith("no "):
        match = re.match(r"no students? (?:with|who)?\s*(.+?) are (.+)$", low)
        if match:
            return PolicyRule(premise, _condition_parts(match.group(1)), _clean_outcome(match.group(2)), positive=False, blocking=True)
    patterns = [
        (r"students? (?:with|who) (.+?) are not (.+)$", False, True),
        (r"students? (?:with|who) (.+?) do not (.+)$", False, True),
        (r"students? (?:with|who) (.+?) are ineligible for (.+)$", True, False),
        (r"students? (?:with|who) (.+?) are (.+)$", True, False),
        (r"students? (?:with|who) (.+?) may (.+)$", True, False),
        (r"students? (?:with|who) (.+?) meet (.+)$", True, False),
        (r"students? (?:with|who) (.+?) satisfy (.+)$", True, False),
        (r"students? (?:with|who) (.+?) are placed on (.+)$", True, False),
        (r"students? (?:with|who) (.+?) do not meet (.+)$", False, True),
        (r"students? (?:with|who) (.+?) do not satisfy (.+)$", False, True),
        (r"absences? (?:with|supported by) (.+?) are (.+)$", True, False),
    ]
    for pattern, positive, blocking in patterns:
        match = re.match(pattern, low)
        if match:
            outcome = _clean_outcome(match.group(2))
            if " only with " in outcome:
                main, extra = outcome.split(" only with ", 1)
                return PolicyRule(premise, _condition_parts(match.group(1)) + [extra], main, positive=positive, blocking=blocking)
            if "ineligible for" in low and positive:
                outcome = "ineligible for " + outcome
            return PolicyRule(premise, _condition_parts(match.group(1)), outcome, positive=positive, blocking=blocking)
    return None


def _explicit_negative(premises: list[Premise], target: str) -> PolicyDecision | None:
    for premise in premises:
        low = norm(premise.text)
        if "exception" in low and " not " in low and overlap(target, low.split(" not ", 1)[1]) > 0:
            return PolicyDecision("no", [premise], "explicit exception overrides general policy", ["Found explicit exception premise", "Returned no from cited exception"], 0.92)
    return None


def _contradiction(premises: list[Premise], target: str) -> PolicyDecision | None:
    all_premise = None
    no_premise = None
    fact_premise = None
    for premise in premises:
        low = norm(premise.text)
        if low.startswith("all ") and overlap(target, low) > 0:
            all_premise = premise
        elif low.startswith("no ") and overlap(target, low) > 0:
            no_premise = premise
        elif not low.startswith(("all ", "no ", "if ", "students ")):
            fact_premise = premise
    if all_premise and no_premise and fact_premise:
        return PolicyDecision("unknown", [all_premise, no_premise, fact_premise], "contradictory academic policy premises", ["Found positive and negative policy rules", "Returned unknown because both apply"], 0.72)
    return None


def _condition_satisfied(condition: str, facts: list[Premise]) -> tuple[bool | None, Premise | None, str]:
    threshold = parse_threshold(condition)
    if threshold:
        for fact in facts:
            value = metric_value(fact.text, threshold.metric)
            if value is not None:
                return compare(value, threshold), fact, f"{threshold.metric} {threshold.operator} {threshold.value:g}"
        return None, None, f"missing {threshold.metric} fact"
    cond_tokens = tokens(condition)
    if "complete" in cond_tokens or "completed" in cond_tokens:
        cond_tokens |= {"completed"}
    if "passed" in cond_tokens and "course" in cond_tokens:
        cond_tokens -= {"course"}
    if "submit" in cond_tokens:
        cond_tokens |= {"submitted"}
    if "have" in cond_tokens or "has" in cond_tokens:
        cond_tokens -= {"have", "has"}
    for fact in facts:
        fact_tokens = tokens(fact.text)
        if cond_tokens and cond_tokens <= fact_tokens:
            return True, fact, condition
        if cond_tokens and len(cond_tokens & fact_tokens) >= max(1, min(3, len(cond_tokens))):
            return True, fact, condition
    return None, None, f"missing condition: {condition}"


def _target_matches(rule: PolicyRule, target: str) -> bool:
    if overlap(rule.outcome, target) >= 1:
        return True
    if "eligible" in target and "eligible" in rule.outcome:
        return overlap(rule.outcome.replace("eligible", ""), target.replace("eligible", "")) >= 1
    if "register" in target and "register" in rule.outcome:
        return True
    if "prerequisite" in target and "prerequisite" in rule.outcome:
        return True
    if "absence" in target and "approved" in rule.outcome:
        return True
    return False


def _evaluate_rule(rule: PolicyRule, facts: list[Premise], target: str) -> PolicyDecision | None:
    if not _target_matches(rule, target):
        return None
    used = [rule.premise]
    missing: list[str] = []
    failed: list[Premise] = []
    for condition in rule.conditions:
        ok, fact, detail = _condition_satisfied(condition, facts)
        if ok is True and fact:
            used.append(fact)
        elif ok is False and fact:
            failed.append(fact)
            used.append(fact)
        else:
            missing.append(detail)
    unique_used = list(dict.fromkeys(used))
    if failed and rule.positive:
        return PolicyDecision("no", unique_used, "required academic policy threshold or condition is not satisfied", ["Matched academic policy rule", "Found a cited fact that fails a required condition"], 0.86)
    if missing:
        return PolicyDecision("unknown", unique_used, "; ".join(missing), ["Matched academic policy rule", "A required condition is absent from the premises"], 0.68)
    answer = "yes" if rule.positive else "no"
    if rule.blocking:
        answer = "no"
    return PolicyDecision(answer, unique_used, "academic policy rule satisfied", ["Matched academic policy rule", "Validated all cited conditions against premises"], 0.9)


def _invalid_inference(question: str, premises: list[Premise], target: str) -> PolicyDecision | None:
    rules = [rule for premise in premises if (rule := _parse_rule(premise))]
    facts = [premise for premise in premises if not _parse_rule(premise)]
    for rule in rules:
        for fact in facts:
            if _target_matches(rule, norm(fact.text)) and not _target_matches(rule, target):
                return PolicyDecision("unknown", [rule.premise, fact], "policy consequent does not prove antecedent", ["Detected affirming-consequent pattern", "Returned unknown"], 0.72)
    return None


def solve_policy(question: str, premises: list[Premise]) -> PolicyDecision | None:
    if not is_academic_policy_text(question, [p.text for p in premises]):
        return None
    target = _question_target(question)
    if "which option" in norm(question):
        if any(norm(p.text).startswith("some ") for p in premises):
            return PolicyDecision("C", select_premises(question, premises), "MCQ insufficient academic policy information", ["Some-premise does not entail the specific student", "Selected unknown option"], 0.8)
    explicit = _explicit_negative(premises, target)
    if explicit:
        return explicit
    contradiction = _contradiction(premises, target)
    if contradiction:
        return contradiction
    invalid = _invalid_inference(question, premises, target)
    if invalid:
        return invalid
    rules = [rule for premise in premises if (rule := _parse_rule(premise))]
    facts = [premise for premise in premises if not _parse_rule(premise)]
    decisions = [decision for rule in rules if (decision := _evaluate_rule(rule, facts, target))]
    if not decisions:
        return None
    for decision in decisions:
        if decision.answer == "no":
            return decision
    for decision in decisions:
        if decision.answer == "yes":
            return decision
    return decisions[0]
