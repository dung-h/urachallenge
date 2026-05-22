from __future__ import annotations

from app.logic.premise_selector import Premise


_NECESSARY_MARKERS = (
    " require ",
    " requires ",
    " required ",
    " needed ",
    " necessary ",
    " only if ",
    " prerequisite ",
)


def _evidence(selected: list[Premise]) -> str:
    return "; ".join(f"{p.id}: {p.text}" for p in selected)


def _has_necessary_condition(selected: list[Premise]) -> bool:
    for premise in selected:
        text = f" {premise.text.lower()} "
        if any(marker in text for marker in _NECESSARY_MARKERS):
            return True
    return False


def logic_explanation(answer: str, selected: list[Premise], rule: str) -> str:
    evidence = _evidence(selected)
    normalized_rule = rule.lower()
    if answer == "unknown" and _has_necessary_condition(selected):
        return (
            "Answer is unknown. The selected premise states a required condition, not a sufficient rule. "
            "Satisfying a required condition does not by itself prove the target claim. "
            "The premises would need an additional rule saying that this condition is enough, but no such rule is given. "
            f"Evidence: {evidence}."
        )
    if answer == "unknown" and (
        "no deterministic" in normalized_rule
        or "insufficient" in normalized_rule
        or "does not identify" in normalized_rule
    ):
        return (
            "Answer is unknown because the selected premises do not entail a definite yes or no. "
            "They provide relevant evidence, but no validated rule connects that evidence to the asked conclusion. "
            f"Evidence: {evidence}."
        )
    return f"Answer is {answer}. Rule used: {rule}. Evidence: {evidence}."
