from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdCondition:
    metric: str
    operator: str
    value: float


def parse_threshold(text: str) -> ThresholdCondition | None:
    low = text.lower()
    metric_match = re.search(r"\b(gpa|cpa|attendance|completed credits?|credits?)\b", low)
    if not metric_match:
        return None
    metric = metric_match.group(1)
    if metric.startswith("credit") or metric.startswith("completed"):
        metric = "credits"
    patterns = [
        (r"(?:at least|minimum|no less than)\s+([0-9]+(?:\.[0-9]+)?)", ">="),
        (r"(?:above|greater than|more than)\s+([0-9]+(?:\.[0-9]+)?)", ">"),
        (r"(?:at most|maximum|no more than)\s+([0-9]+(?:\.[0-9]+)?)", "<="),
        (r"(?:below|less than|fewer than)\s+([0-9]+(?:\.[0-9]+)?)", "<"),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, low)
        if match:
            return ThresholdCondition(metric=metric, operator=operator, value=float(match.group(1)))
    return None


def metric_value(text: str, metric: str) -> float | None:
    low = text.lower()
    if metric in {"gpa", "cpa"}:
        match = re.search(rf"\b{metric}\s*(?:is|=)?\s*([0-9]+(?:\.[0-9]+)?)\b", low)
        return float(match.group(1)) if match else None
    if metric == "credits":
        match = re.search(r"(?:completed\s+)?([0-9]+(?:\.[0-9]+)?)\s+credits?\b", low)
        return float(match.group(1)) if match else None
    if metric == "attendance":
        match = re.search(r"attendance\s*(?:is|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*percent\b", low)
        return float(match.group(1)) if match else None
    return None


def compare(value: float, condition: ThresholdCondition) -> bool:
    if condition.operator == ">=":
        return value >= condition.value
    if condition.operator == ">":
        return value > condition.value
    if condition.operator == "<=":
        return value <= condition.value
    if condition.operator == "<":
        return value < condition.value
    return False
