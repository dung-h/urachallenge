from __future__ import annotations

import re

from app.logic.premise_selector import Premise
from app.logic.policy_patterns import tokens
from app.logic.thresholds import compare, metric_value, parse_threshold


_NECESSARY_MARKERS = (
    " require ",
    " requires ",
    " required ",
    " needed ",
    " necessary ",
    " only if ",
    " prerequisite ",
)


def _norm(text: str) -> str:
    return str(text or "").lower().strip()


def _evidence(selected: list[Premise]) -> str:
    return "; ".join(f"{p.id}: {p.text.rstrip('.')}" for p in selected)


def _has_necessary_condition(selected: list[Premise]) -> bool:
    for premise in selected:
        text = f" {premise.text.lower()} "
        if any(marker in text for marker in _NECESSARY_MARKERS):
            return True
    return False


def _extract_required_clauses(rule_text: str) -> list[str]:
    low = rule_text.lower().strip().rstrip(".")
    for marker in (" only if ", " requires ", " with ", " who ", " that "):
        if marker in low:
            tail = low.split(marker, 1)[1]
            normalized_tail = re.sub(r"\s+and\s+", ", ", tail)
            clauses = [_clean_required_clause(part) for part in re.split(r"\s*,\s*", normalized_tail) if part.strip(" ,;.")]
            clauses = [clause for clause in clauses if clause]
            if clauses:
                return clauses
    return []


def _clean_required_clause(text: str) -> str:
    clause = text.strip(" ,;.")
    clause = re.sub(r"\s+", " ", clause)
    # Rules phrased as "Students with A and B are eligible..." put the
    # consequent after the final condition. Keep only the condition itself.
    clause = re.sub(
        r"\s+(?:are|is|be|become|qualify|qualifies|qualified|may|can)\s+"
        r"(?:eligible|allowed|permitted|able|qualified|receive|receives|register|registered|pass|passes|graduate|graduates)\b.*$",
        "",
        clause,
    ).strip(" ,;.")
    clause = re.sub(r"^(?:a|an|the)\s+", "", clause)
    return clause.strip()


def _missing_text(missing: set[str], fallback: str) -> str:
    if not missing:
        return fallback
    ordered = sorted(missing)
    if {"completed", "complete"} & set(ordered) and "prerequisite" in ordered:
        return "completed prerequisite"
    if {"submitted", "submit"} & set(ordered) and "portfolio" in ordered:
        return "submitted portfolio"
    return " ".join(ordered)


def _match_if_rule(text: str) -> tuple[str, str] | None:
    match = re.match(r"if (.+?),? then (.+?)[.]?$", _norm(text)) or re.match(r"if (.+?), (.+?)[.]?$", _norm(text))
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def _match_all_rule(text: str) -> tuple[str, str] | None:
    low = _norm(text)
    match = re.match(r"all (.+?) are (.+?)[.]?$", low)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.match(r"all (.+?) have (.+?)[.]?$", low)
    if match:
        return match.group(1).strip(), "have " + match.group(2).strip()
    match = re.match(r"all (.+?) need (.+?)[.]?$", low)
    if match:
        return match.group(1).strip(), "need " + match.group(2).strip()
    return None


def _token_subset_needed(clause: str, text: str) -> tuple[bool, set[str], set[str]]:
    clause_tokens = tokens(clause)
    text_tokens = tokens(text)
    missing = clause_tokens - text_tokens
    return not missing, clause_tokens, missing


def _clause_supported(clause: str, facts: list[Premise]) -> bool:
    return _clause_support_detail(clause, facts)[0]


def _clause_support_detail(clause: str, facts: list[Premise]) -> tuple[bool, Premise | None]:
    threshold = parse_threshold(clause)
    if threshold:
        for premise in facts:
            value = metric_value(premise.text, threshold.metric)
            if value is not None and compare(value, threshold):
                return True, premise
    clause_tokens = tokens(clause)
    if not clause_tokens:
        return False, None
    for premise in facts:
        fact_tokens = tokens(premise.text)
        overlap = len(clause_tokens & fact_tokens)
        if overlap >= max(1, min(3, len(clause_tokens))):
            return True, premise
    return False, None


def logic_explanation(answer: str, selected: list[Premise], rule: str) -> str:
    evidence = _evidence(selected)
    normalized_rule = rule.lower()
    if answer == "unknown" and selected:
        rule_text = selected[0].text if selected else ""
        required_clauses = _extract_required_clauses(rule_text)
        facts = selected[1:] if len(selected) > 1 else []
        supported_details: list[tuple[str, Premise | None]] = []
        for clause in required_clauses:
            supported, premise = _clause_support_detail(clause, facts)
            if supported:
                supported_details.append((clause, premise))
        supported = [clause for clause, _premise in supported_details]
        missing = [clause for clause in required_clauses if clause not in supported]
        if required_clauses and missing:
            if supported:
                support_clause, support_premise = supported_details[0]
                support_text = support_premise.text.rstrip(".") if support_premise else support_clause
                missing_text = missing[0]
                return (
                    "Answer is unknown. "
                    f"{support_text}, which satisfies the {support_clause} condition, but the premises do not establish the missing {missing_text} condition. "
                    "Since the rule requires all of its conditions, this is not a sufficient rule on its own, so the conclusion cannot be confirmed. "
                    f"Evidence: {evidence}."
                )
            return (
                "Answer is unknown. The rule states a required condition: "
                f"{missing[0]}, but the premises do not establish that condition. "
                "Since this is not a sufficient rule on its own, the conclusion cannot be confirmed. "
                f"Evidence: {evidence}."
            )
        if required_clauses and len(required_clauses) == 1 and supported:
            support_clause, support_premise = supported_details[0]
            support_text = support_premise.text.rstrip(".") if support_premise else support_clause
            return (
                "Answer is unknown. "
                f"The premises establish {support_text}, which satisfies the required condition {support_clause}, but the rule is not a sufficient rule for the conclusion. "
                f"Evidence: {evidence}."
            )
    if answer == "unknown" and _has_necessary_condition(selected):
        if evidence:
            return (
                "Answer is unknown. The cited premise states a required condition, but it is not a sufficient rule on its own to prove the conclusion. "
                "A missing condition or supporting rule is still needed. "
                f"Evidence: {evidence}."
            )
        return (
            "Answer is unknown. A required condition is mentioned, but the premises do not show that it is sufficient to prove the conclusion."
        )
    if answer == "unknown" and selected:
        if_rule = _match_if_rule(selected[0].text)
        if if_rule and len(selected) > 1:
            antecedent, consequent = if_rule
            fact_text = selected[1].text
            consequent_supported, _consequent_tokens, _consequent_missing = _token_subset_needed(consequent, fact_text)
            antecedent_supported, _antecedent_tokens, antecedent_missing = _token_subset_needed(antecedent, fact_text)
            if consequent_supported and not antecedent_supported:
                missing_text = _missing_text(antecedent_missing, antecedent)
                return (
                    "Answer is unknown. The premises state the consequence, but they do not establish the if-condition; "
                    "this is affirming the consequent, so the rule does not prove the conclusion. "
                    f"The missing condition is {missing_text}. Evidence: {evidence}."
                )
        all_rule = _match_all_rule(selected[0].text)
        if all_rule and len(selected) > 1:
            antecedent, consequent = all_rule
            fact_text = selected[1].text
            antecedent_supported, _antecedent_tokens, antecedent_missing = _token_subset_needed(antecedent, fact_text)
            if not antecedent_supported:
                missing_text = _missing_text(antecedent_missing, antecedent)
                return (
                    "Answer is unknown. The universal rule requires the antecedent class, but the premises only establish "
                    f"{fact_text.rstrip('.')}; they do not show the subject belongs to that class. "
                    f"The missing class property is {missing_text}. Evidence: {evidence}."
                )
    if answer == "unknown" and (
        "no deterministic" in normalized_rule
        or "insufficient" in normalized_rule
        or "does not identify" in normalized_rule
    ):
        if evidence:
            return (
                "Answer is unknown. The selected premises are relevant, but they do not form a complete chain to the conclusion. "
                f"Evidence: {evidence}."
            )
        return (
            "Answer is unknown. The premises are relevant, but they do not provide enough information to prove the conclusion."
        )
    if evidence:
        return f"Answer is {answer}. Rule used: {rule}. Evidence: {evidence}."
    return f"Answer is {answer}. Rule used: {rule}."
