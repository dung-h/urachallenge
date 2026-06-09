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


def _split_or_conditions(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+or\s+", text) if part.strip()]


def _parse_rule(premise: Premise) -> PolicyRule | None:
    low = norm(premise.text)
    student_prefix = r"(?:(?:a|an|the)\s+student(?:s)?|students?)"
    if low.startswith("no "):
        match = re.match(r"no students? (?:with|who)?\s*(.+?) are (.+)$", low)
        if match:
            return PolicyRule(premise, _condition_parts(match.group(1)), _clean_outcome(match.group(2)), positive=False, blocking=True)
        match = re.match(r"no (.+?) students? are (.+)$", low)
        if match:
            return PolicyRule(premise, _condition_parts(match.group(1)), _clean_outcome(match.group(2)), positive=False, blocking=True)
    patterns = [
        (rf"{student_prefix} receives? (.+?) if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} is (.+?) if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} are (.+?) if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} may register for (.+?) only if (.+)$", 2, 1, True, False, "register for "),
        (rf"{student_prefix} may (register for .+?) only if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} is (.+?) only if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} are (.+?) only if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} receives? (.+?) only if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} may (.+?) if (.+)$", 2, 1, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) are not (.+)$", 1, 2, False, True, None),
        (rf"{student_prefix} (?:with|who) (.+?) do not (.+)$", 1, 2, False, True, None),
        (rf"{student_prefix} (?:with|who) (.+?) are ineligible for (.+)$", 1, 2, False, True, "ineligible for "),
        (rf"{student_prefix} (?:with|who) (.+?) is (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) are (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) receives? (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) may (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) meet (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) satisfy (.+)$", 1, 2, True, False, None),
        (rf"{student_prefix} (?:with|who) (.+?) are placed on (.+)$", 1, 2, True, False, "placed on "),
        (rf"{student_prefix} (?:with|who) (.+?) do not meet (.+)$", 1, 2, False, True, None),
        (rf"{student_prefix} (?:with|who) (.+?) do not satisfy (.+)$", 1, 2, False, True, None),
        (r"absences? (?:with|supported by) (.+?) are (.+)$", 1, 2, True, False, None),
        (r"(.+?) cancels (?:the )?(.+)$", 1, 2, False, True, None),
        (r"(.+?) requires (.+)$", 2, 1, True, False, None),
    ]
    for pattern, condition_group, outcome_group, positive, blocking, outcome_prefix in patterns:
        match = re.match(pattern, low)
        if not match:
            continue
        condition_text = match.group(condition_group)
        outcome_text = match.group(outcome_group)
        if outcome_prefix:
            outcome_text = outcome_prefix + outcome_text
        outcome = _clean_outcome(outcome_text)
        if " only with " in outcome:
            main, extra = outcome.split(" only with ", 1)
            return PolicyRule(premise, _condition_parts(condition_text) + [extra], main, positive=positive, blocking=blocking)
        return PolicyRule(premise, _condition_parts(condition_text), outcome, positive=positive, blocking=blocking)
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


def _direct_fact_contradiction(premises: list[Premise], target: str) -> PolicyDecision | None:
    positive: Premise | None = None
    negative: Premise | None = None
    for premise in premises:
        if _parse_rule(premise):
            continue
        low = norm(premise.text)
        if low.startswith(("all ", "no ", "students ", "student ", "a student ", "an student ", "the student ")):
            continue
        if overlap(target, low) < 2:
            continue
        if re.search(r"\b(?:does not|do not|not|never)\b", low):
            negative = premise
        else:
            positive = premise
    if positive and negative:
        return PolicyDecision(
            "unknown",
            [positive, negative],
            "directly contradictory policy facts",
            ["Found a positive target fact", "Found a negative target fact", "Returned unknown because both cannot be accepted together"],
            0.72,
        )
    return None


def _is_choice_question(question: str) -> bool:
    low = norm(question)
    return "which option" in low and "a eligible" in low and "b not eligible" in low and "c unknown" in low


def _choice_answer(question: str, answer: str) -> str:
    if not _is_choice_question(question):
        return answer
    return {"yes": "A", "no": "B", "unknown": "C"}.get(answer, answer)


def _metric_facts(metric: str, facts: list[Premise]) -> list[tuple[Premise, float]]:
    values: list[tuple[Premise, float]] = []
    for fact in facts:
        value = metric_value(fact.text, metric)
        if value is not None:
            values.append((fact, value))
    return values


def _metric_conflict(metric: str, facts: list[Premise]) -> list[Premise]:
    values = _metric_facts(metric, facts)
    unique = {value for _premise, value in values}
    return [premise for premise, _value in values] if len(unique) > 1 else []


def _has_phrase(facts: list[Premise], *phrases: str) -> Premise | None:
    for fact in facts:
        low = norm(fact.text)
        if any(phrase in low for phrase in phrases):
            return fact
    return None


def _has_subject_fact(facts: list[Premise], phrase: str) -> Premise | None:
    for fact in facts:
        low = norm(fact.text)
        if low.startswith("some "):
            continue
        if phrase in low:
            return fact
    return None


def _has_phrase_fact(facts: list[Premise], phrase: str) -> Premise | None:
    pattern = re.compile(rf"\b{re.escape(norm(phrase))}\b")
    for fact in facts:
        low = norm(fact.text)
        if pattern.search(low):
            return fact
    return None


def _has_positive_phrase_fact(facts: list[Premise], phrase: str) -> Premise | None:
    needle = norm(phrase)
    for fact in facts:
        low = norm(fact.text)
        if needle == "fee hold":
            if "fee hold" in low and "no fee hold" not in low and "no active fee hold" not in low:
                return fact
        elif needle == "active fee hold":
            if "active fee hold" in low and "no active fee hold" not in low:
                return fact
        elif needle == "passed the capstone":
            if "passed the capstone" in low and "not passed the capstone" not in low:
                return fact
        elif needle == "completed the capstone":
            if "completed the capstone" in low and "not completed the capstone" not in low:
                return fact
        elif needle == "paid tuition":
            if "paid tuition" in low and "unpaid tuition" not in low:
                return fact
        else:
            pattern = re.compile(rf"\b{re.escape(needle)}\b")
            if pattern.search(low) and not low.startswith(("no ", "not ")):
                return fact
    return None


def _grade_value(facts: list[Premise]) -> tuple[Premise, int] | None:
    # Lower value means better grade. "Below C" means D/F.
    order = {"a": 1, "b": 2, "c": 3, "d": 4, "f": 5}
    for fact in facts:
        low = norm(fact.text)
        if "grade is not recorded" in low or "previous grade is not recorded" in low:
            return None
        match = re.search(r"\b(?:previous\s+)?grade is ([abcdf])\b", low)
        if match:
            return fact, order[match.group(1)]
    return None


def _condition_status(condition: str, facts: list[Premise]) -> tuple[bool | None, Premise | None, str]:
    low = norm(condition)
    disjuncts = _split_or_conditions(low)
    if len(disjuncts) > 1:
        saw_missing = False
        failed: tuple[Premise | None, str] | None = None
        for part in disjuncts:
            ok, fact, detail = _condition_status(part, facts)
            if ok is True:
                return True, fact, detail
            if ok is False:
                failed = (fact, detail)
            else:
                saw_missing = True
        if saw_missing:
            return None, None, f"missing condition: {condition}"
        if failed:
            return False, failed[0], failed[1]
        return None, None, f"missing condition: {condition}"
    threshold = parse_threshold(low)
    if threshold:
        conflicts = _metric_conflict(threshold.metric, facts)
        if conflicts:
            return None, conflicts[0], f"conflicting {threshold.metric} facts"
        for fact, value in _metric_facts(threshold.metric, facts):
            return compare(value, threshold), fact, f"{threshold.metric} {threshold.operator} {threshold.value:g}"
        return None, None, f"missing {threshold.metric} fact"
    if "nominated" in low or "nomination" in low:
        fact = _has_subject_fact(facts, " is nominated") or _has_subject_fact(facts, "has a faculty nomination") or _has_subject_fact(facts, "has faculty nomination")
        return (True, fact, low) if fact else (None, None, "missing nomination fact")
    if "advisor recommendation" in low:
        fact = _has_subject_fact(facts, "advisor recommendation")
        if fact and "not recorded" not in norm(fact.text):
            return True, fact, low
        return None, fact, "missing advisor recommendation"
    if "income is verified" in low or "income verified" in low:
        fact = _has_subject_fact(facts, "income is verified") or _has_subject_fact(facts, "income verified")
        missing = _has_subject_fact(facts, "income verification is not recorded")
        return (True, fact, low) if fact else (None, missing, "missing income verification")
    if "tuition is paid" in low or "paid tuition" in low:
        bad = _has_subject_fact(facts, "tuition is unpaid") or _has_subject_fact(facts, "unpaid tuition")
        if bad:
            return False, bad, "tuition unpaid"
        fact = _has_subject_fact(facts, "tuition is paid") or _has_subject_fact(facts, "paid tuition")
        return (True, fact, low) if fact else (None, None, "missing tuition payment fact")
    if "no active hold" in low or "no fee hold" in low or "there is no fee hold" in low:
        bad = _has_subject_fact(facts, "active fee hold") or _has_subject_fact(facts, "fee hold") or _has_subject_fact(facts, "unpaid tuition")
        if bad and "no fee hold" not in norm(bad.text) and "no active hold" not in norm(bad.text):
            return False, bad, "fee hold present"
        fact = _has_subject_fact(facts, "no active hold") or _has_subject_fact(facts, "no fee hold")
        return (True, fact, low) if fact else (None, None, "missing no-hold fact")
    if "completed" in low:
        course_match = re.search(r"completed\s+([a-z]{2,}\d+|[a-z]+\s+[a-z]+)", low)
        if course_match:
            course = course_match.group(1)
            fact = _has_subject_fact(facts, f"completed {course}")
            return (True, fact, low) if fact else (None, None, f"missing completion fact: {course}")
        fact = _has_subject_fact(facts, " completed ")
        return (True, fact, low) if fact else (None, None, "missing completion fact")
    if "capstone is passed" in low or "capstone is pass" in low or "capstone passed" in low:
        bad = _has_subject_fact(facts, "not passed the capstone") or _has_subject_fact(facts, "capstone requirement is not recorded")
        if bad:
            return (None, bad, "missing capstone fact") if "not recorded" in norm(bad.text) else (False, bad, "capstone not passed")
        fact = _has_subject_fact(facts, "passed the capstone") or _has_subject_fact(facts, "completed the capstone")
        return (True, fact, low) if fact else (None, None, "missing capstone fact")
    if "degree requirements are complete" in low:
        missing = _has_subject_fact(facts, "not recorded")
        if missing:
            return None, missing, "missing degree requirement"
        required = ["core requirement", "elective requirement", "capstone requirement", "internship requirement", "clearance requirement"]
        used = [_has_subject_fact(facts, phrase) for phrase in required]
        if all(used):
            return True, used[0], low
        return None, None, "missing degree requirements"
    if "previous grade is below c" in low or "grade is below c" in low:
        grade = _grade_value(facts)
        if grade:
            premise, value = grade
            return value > 3, premise, "previous grade below C"
        return None, None, "missing previous grade"
    return _condition_satisfied(condition, facts)


def _conditions_decision(
    question: str,
    rule: PolicyRule,
    facts: list[Premise],
    *,
    single_necessary_is_insufficient: bool = False,
) -> PolicyDecision:
    used = [rule.premise]
    missing: list[str] = []
    failed: list[Premise] = []
    for condition in rule.conditions:
        ok, fact, detail = _condition_status(condition, facts)
        if fact:
            used.append(fact)
        if ok is False and fact:
            failed.append(fact)
        elif ok is None:
            missing.append(detail)
    unique_used = list(dict.fromkeys(used))
    if failed:
        if "retake" in norm(question) and any(
            token in norm(premise.text)
            for premise in facts
            for token in ["course offered", "no fee hold", "not suspended", "no active hold"]
        ):
            return PolicyDecision(
                _choice_answer(question, "no"),
                unique_used,
                "required policy condition failed",
                ["Matched policy rule", "Found a failed required condition"],
                0.86,
            )
        if rule.positive and not rule.blocking:
            if "retake" in norm(question):
                return PolicyDecision(
                    _choice_answer(question, "unknown"),
                    unique_used,
                    "required policy condition failed",
                    ["Matched policy rule", "Found a failed required condition"],
                    0.68,
                )
            return PolicyDecision(
                _choice_answer(question, "no"),
                unique_used,
                "required policy condition failed",
                ["Matched policy rule", "Found a failed required condition"],
                0.86,
            )
        return PolicyDecision(
            _choice_answer(question, "unknown"),
            unique_used,
            "required policy condition failed",
            ["Matched policy rule", "Found a failed required condition"],
            0.68,
        )
    if missing or (single_necessary_is_insufficient and len(rule.conditions) <= 1):
        return PolicyDecision(_choice_answer(question, "unknown"), unique_used, "; ".join(missing) or "necessary condition alone is not sufficient", ["Matched policy rule", "A required condition is absent or insufficient"], 0.68)
    answer = "yes"
    if rule.blocking or not rule.positive:
        answer = "yes" if any(token in norm(question) for token in ["ineligible", "not eligible"]) else "no"
    return PolicyDecision(_choice_answer(question, answer), unique_used, "all policy conditions satisfied", ["Matched policy rule", "Validated all required conditions"], 0.9)


def _specific_policy_decision(question: str, premises: list[Premise], target: str) -> PolicyDecision | None:
    low_question = norm(question)
    rules = [rule for premise in premises if (rule := _parse_rule(premise))]
    facts = [premise for premise in premises if not _parse_rule(premise)]
    choice_question = _is_choice_question(question)

    if choice_question:
        target = "eligible"

    if "academic warning" in low_question:
        direct = _direct_fact_contradiction(premises, target)
        if direct:
            return direct
        appeal = _has_subject_fact(facts, "has an approved appeal")
        if appeal:
            return PolicyDecision(_choice_answer(question, "no"), [appeal], "approved appeal cancels warning", ["Found approved appeal exception"], 0.9)
        conflicts = _metric_conflict("gpa", facts) or _metric_conflict("cpa", facts) or _metric_conflict("credits", facts)
        if conflicts:
            return PolicyDecision(_choice_answer(question, "unknown"), conflicts, "conflicting academic warning facts", ["Found contradictory metric facts"], 0.72)
        warning_rules = [rule for rule in rules if "academic warning" in rule.outcome or "warning" in rule.outcome]
        for rule in warning_rules:
            any_trigger = False
            missing = False
            failed = False
            used = [rule.premise]
            for condition in rule.conditions:
                ok, fact, detail = _condition_status(condition, facts)
                if fact:
                    used.append(fact)
                if ok is True:
                    any_trigger = True
                elif ok is False:
                    failed = True
                elif ok is None and ("gpa" in detail or "cpa" in detail or "credits" in detail):
                    missing = True
            if any_trigger:
                return PolicyDecision(_choice_answer(question, "yes"), list(dict.fromkeys(used)), "academic warning threshold triggered", ["Matched warning threshold"], 0.88)
            if missing:
                return PolicyDecision(_choice_answer(question, "unknown"), list(dict.fromkeys(used)), "missing academic warning threshold fact", ["A warning condition is unverified"], 0.68)
            if failed and (len(rule.conditions) > 1 or any(" or " in condition for condition in rule.conditions)):
                return PolicyDecision(_choice_answer(question, "no"), list(dict.fromkeys(used)), "no academic warning threshold is triggered", ["Checked warning thresholds"], 0.84)
            if failed:
                return PolicyDecision(_choice_answer(question, "unknown"), list(dict.fromkeys(used)), "missing academic warning threshold fact", ["A warning condition is unverified"], 0.68)
            return PolicyDecision(_choice_answer(question, "no"), list(dict.fromkeys(used)), "no academic warning threshold is triggered", ["Checked warning thresholds"], 0.84)

    blocker = (
        _has_subject_fact(facts, "disciplinary warning")
        or _has_subject_fact(facts, "disciplinary suspension")
        or _has_subject_fact(facts, "under disciplinary suspension")
        or _has_positive_phrase_fact(facts, "active fee hold")
        or _has_positive_phrase_fact(facts, "fee hold")
        or _has_subject_fact(facts, "unpaid tuition")
        or _has_subject_fact(facts, "tuition is unpaid")
        or _has_subject_fact(facts, "on academic warning")
        or _has_subject_fact(facts, "not passed the capstone")
    )
    if blocker and not any(marker in norm(blocker.text) for marker in ["no fee hold", "no active hold", "not on academic warning", "no disciplinary warning"]):
        blocker_text = norm(blocker.text)
        counter_facts: list[Premise] = []
        if "fee hold" in blocker_text:
            counter_facts.extend([
                _has_phrase_fact(facts, "no active fee hold"),
                _has_phrase_fact(facts, "no fee hold"),
            ])
        if "tuition" in blocker_text:
            counter_facts.extend([
                _has_phrase_fact(facts, "tuition is paid"),
                _has_phrase_fact(facts, "paid tuition"),
            ])
        if "academic warning" in blocker_text:
            counter_facts.append(_has_phrase_fact(facts, "not on academic warning"))
        if "capstone" in blocker_text:
            counter_facts.extend([
                _has_positive_phrase_fact(facts, "passed the capstone"),
                _has_positive_phrase_fact(facts, "completed the capstone"),
            ])
        counter_facts = [fact for fact in counter_facts if fact]
        if counter_facts:
            return PolicyDecision(
                _choice_answer(question, "unknown"),
                list(dict.fromkeys([blocker] + counter_facts)),
                "conflicting policy facts",
                ["Found both a blocking fact and a countervailing fact"],
                0.72,
            )
        if "capstone" in blocker_text and not any(" only if " in norm(rule.premise.text) and "graduate" in rule.outcome for rule in rules):
            return PolicyDecision(
                _choice_answer(question, "unknown"),
                [blocker],
                "policy blocker or exception applies",
                ["Found blocking fact"],
                0.72,
            )
        if any(token in low_question for token in ["eligible", "register", "graduate", "financial aid", "warning"]):
            answer = "no"
            if not choice_question and any(token in target for token in ["ineligible", "not eligible"]):
                answer = "yes"
            return PolicyDecision(_choice_answer(question, answer), [blocker], "policy blocker or exception applies", ["Found blocking fact"], 0.88)

    if "exam eligible" in low_question:
        active_hold = _has_positive_phrase_fact(facts, "active fee hold") or _has_positive_phrase_fact(facts, "fee hold")
        no_active_hold = _has_phrase_fact(facts, "no active fee hold")
        if active_hold and no_active_hold:
            return PolicyDecision(
                _choice_answer(question, "unknown"),
                [active_hold, no_active_hold],
                "conflicting fee hold facts",
                ["Found both active and no-active fee hold facts"],
                0.72,
            )
        tuition_bad = _has_phrase_fact(facts, "tuition is unpaid") or _has_phrase_fact(facts, "tuition status is not recorded")
        tuition_paid = _has_positive_phrase_fact(facts, "tuition is paid") or _has_positive_phrase_fact(facts, "paid tuition")
        if tuition_bad and tuition_paid:
            return PolicyDecision(
                _choice_answer(question, "unknown"),
                [tuition_bad, tuition_paid],
                "conflicting tuition facts",
                ["Found both unpaid and paid tuition facts"],
                0.72,
            )
        if tuition_bad:
            answer = "unknown" if "not recorded" in norm(tuition_bad.text) else "no"
            return PolicyDecision(_choice_answer(question, answer), [tuition_bad], "tuition condition controls exam eligibility", ["Checked tuition condition"], 0.84)
        exception = _has_phrase_fact(facts, "has an approved absence exception") or _has_phrase_fact(facts, "has an approved absence")
        has_exception_rule = any(
            "approved absence" in norm(premise.text) and any(token in norm(premise.text) for token in ["makes", "satisfies", "waiver", "exception"])
            for premise in premises
        )
        has_requirement_rule = any("requires" in norm(premise.text) or " only if " in norm(premise.text) for premise in premises)
        if not choice_question:
            for rule in rules:
                if rule.positive:
                    continue
                if "exam eligible" not in _clean_outcome(rule.outcome):
                    continue
                decision = _conditions_decision(question, rule, facts)
                if decision:
                    return decision
        attendance_ok, attendance_fact, _detail = _condition_status("attendance is at least 75 percent", facts)
        tuition_ok, tuition_fact, _ = _condition_status("tuition is paid", facts)
        used = [p for p in [attendance_fact, tuition_fact, exception] if p]
        if tuition_ok is True and (attendance_ok is True or exception):
            return PolicyDecision(_choice_answer(question, "yes"), used, "exam eligibility conditions satisfied", ["Validated attendance or approved exception", "Validated tuition"], 0.9)
        if attendance_ok is False and not exception:
            answer = "no" if has_exception_rule or has_requirement_rule else "unknown"
            return PolicyDecision(_choice_answer(question, answer), used, "attendance requirement failed", ["Checked attendance threshold"], 0.86 if answer == "no" else 0.68)
        return PolicyDecision(_choice_answer(question, "unknown"), used, "missing exam eligibility condition", ["A required exam condition is absent"], 0.68)

    if "register for" in low_question:
        course_match = re.search(r"register for\s+([a-z]{2,}\d+)", low_question)
        course = course_match.group(1) if course_match else ""
        required_rules = [rule for rule in rules if course and rule.outcome == course]
        if required_rules:
            rule = required_rules[0]
            needed = [cond for cond in rule.conditions if re.search(r"[a-z]{2,}\d+", cond)]
            used = [rule.premise]
            missing = False
            for cond in needed:
                course_need = re.search(r"[a-z]{2,}\d+", cond)
                fact = _has_subject_fact(facts, f"completed {course_need.group(0)}") if course_need else None
                if fact:
                    used.append(fact)
                else:
                    missing = True
            if not missing:
                return PolicyDecision(_choice_answer(question, "yes"), list(dict.fromkeys(used)), "course prerequisite chain satisfied", ["Validated required completed course facts"], 0.86)

    if "graduate" in low_question:
        clearance_missing = _has_subject_fact(facts, "disciplinary clearance is not recorded")
        if clearance_missing:
            return PolicyDecision(_choice_answer(question, "unknown"), [clearance_missing], "missing disciplinary clearance", ["Graduation clearance fact is absent"], 0.68)

    if choice_question or any(token in low_question for token in ["scholarship eligible", "eligible for", "ineligible", "financial aid", "register", "retake", "graduate"]):
        relevant_rules = [
            rule
            for rule in rules
            if _target_matches(rule, target) or any(token in rule.outcome for token in ["register", "eligible", "financial aid", "graduate", "retake"])
        ]
        if relevant_rules:
            rule = relevant_rules[0]
            necessary_only = ("requires" in norm(rule.premise.text) or " only if " in norm(rule.premise.text)) and "retake" not in low_question
            return _conditions_decision(question, rule, facts, single_necessary_is_insufficient=necessary_only)

    if choice_question:
        return PolicyDecision("C", select_premises(question, premises), "MCQ policy information is insufficient", ["Selected unknown option"], 0.72)
    return None


def _condition_satisfied(condition: str, facts: list[Premise]) -> tuple[bool | None, Premise | None, str]:
    disjuncts = _split_or_conditions(condition)
    if len(disjuncts) > 1:
        saw_missing = False
        false_fact: Premise | None = None
        false_detail = ""
        for part in disjuncts:
            ok, fact, detail = _condition_satisfied(part, facts)
            if ok is True:
                return True, fact, detail
            if ok is False:
                false_fact = fact or false_fact
                false_detail = detail
            else:
                saw_missing = True
        if saw_missing:
            return None, None, f"missing condition: {condition}"
        if false_fact:
            return False, false_fact, false_detail or f"missing condition: {condition}"
        return None, None, f"missing condition: {condition}"
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
    if "eligible" in target and "eligibility" in rule.outcome:
        return True
    if "eligibility" in target and "eligible" in rule.outcome:
        return True
    if "eligible" in target and "ineligible" in rule.outcome:
        return overlap(rule.outcome.replace("ineligible", ""), target.replace("eligible", "")) >= 1
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
    direct_conflict = _direct_fact_contradiction(premises, target)
    if direct_conflict:
        return direct_conflict
    contradiction = _contradiction(premises, target)
    if contradiction:
        return contradiction
    specific = _specific_policy_decision(question, premises, target)
    if specific:
        return specific
    explicit = _explicit_negative(premises, target)
    if explicit:
        return explicit
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
