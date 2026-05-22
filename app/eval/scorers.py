from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Any

from app.schemas import QAResponse, normalize_answer_label


UNIT_TABLE: dict[str, tuple[str, float]] = {
    "v": ("V", 1.0),
    "mv": ("V", 1e-3),
    "kv": ("V", 1e3),
    "volt": ("V", 1.0),
    "volts": ("V", 1.0),
    "a": ("A", 1.0),
    "ma": ("A", 1e-3),
    "amp": ("A", 1.0),
    "amps": ("A", 1.0),
    "ohm": ("ohm", 1.0),
    "ohms": ("ohm", 1.0),
    "omega": ("ohm", 1.0),
    "ω": ("ohm", 1.0),
    "Ω": ("ohm", 1.0),
    "kohm": ("ohm", 1e3),
    "kω": ("ohm", 1e3),
    "kΩ": ("ohm", 1e3),
    "w": ("W", 1.0),
    "mw": ("W", 1e-3),
    "kw": ("W", 1e3),
    "j": ("J", 1.0),
    "mj": ("J", 1e-3),
    "uj": ("J", 1e-6),
    "μj": ("J", 1e-6),
    "nj": ("J", 1e-9),
    "c": ("C", 1.0),
    "mc": ("C", 1e-3),
    "uc": ("C", 1e-6),
    "μc": ("C", 1e-6),
    "nc": ("C", 1e-9),
    "f": ("F", 1.0),
    "mf": ("F", 1e-3),
    "uf": ("F", 1e-6),
    "μf": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "pf": ("F", 1e-12),
    "n": ("N", 1.0),
    "mn": ("N", 1e-3),
    "m": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "hz": ("Hz", 1.0),
    "khz": ("Hz", 1e3),
    "mhz": ("Hz", 1e6),
    "rad/s": ("rad/s", 1.0),
    "h": ("H", 1.0),
    "mh": ("H", 1e-3),
    "t": ("T", 1.0),
    "mt": ("T", 1e-3),
    "%": ("%", 1.0),
    "dimensionless": ("dimensionless", 1.0),
}


@dataclass(frozen=True)
class QualityScore:
    answer_correct: bool
    numeric_correct: bool | None
    unit_correct: bool | None
    premise_precision: float | None
    premise_recall: float | None
    formula_correct: bool | None
    hallucinated_premise: bool
    explanation_consistent: bool
    confidence_bucket: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gold_answer(row: dict[str, Any]) -> Any:
    for key in ("gold_answer", "expected_answer", "gold"):
        if key in row:
            return row[key]
    return None


def gold_unit(row: dict[str, Any]) -> str | None:
    return row.get("gold_unit") or row.get("unit")


def _extract_value_unit(text: Any) -> tuple[float | None, str | None]:
    value = str(text or "").replace("µ", "μ").replace("Ω", "Ω")
    match = re.search(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([a-zA-ZμΩ/%]+(?:/[a-zA-Z]+)?|%)?", value, re.I)
    if not match:
        return None, None
    number = float(match.group(1))
    raw_unit = (match.group(2) or "").strip()
    key = raw_unit.lower()
    if raw_unit in {"Ω", "ω"}:
        key = "ω"
    if key not in UNIT_TABLE:
        return number, raw_unit or None
    base_unit, factor = UNIT_TABLE[key]
    return number * factor, base_unit


def _score_physics(row: dict[str, Any], response: QAResponse) -> tuple[bool, bool | None, bool | None]:
    expected = gold_answer(row)
    if expected is None:
        return False, None, None
    actual_value, actual_unit = _extract_value_unit(response.answer)
    expected_value, expected_unit_from_answer = _extract_value_unit(expected)
    expected_unit = gold_unit(row) or expected_unit_from_answer
    if expected_value is None:
        answer_correct = normalize_answer_label(response.answer) == normalize_answer_label(expected)
        return answer_correct, None, None
    expected_base_unit = None
    if expected_unit:
        _, expected_base_unit = _extract_value_unit(f"1 {expected_unit}")
    tolerance = float(row.get("tolerance") or max(1e-6, abs(expected_value) * 1e-3))
    numeric_correct = actual_value is not None and math.isclose(actual_value, expected_value, rel_tol=1e-3, abs_tol=tolerance)
    unit_correct = True if not expected_base_unit else actual_unit == expected_base_unit
    return numeric_correct and unit_correct, numeric_correct, unit_correct


def _score_logic(row: dict[str, Any], response: QAResponse) -> bool:
    expected = gold_answer(row)
    if expected is None:
        return False
    return normalize_answer_label(response.answer) == normalize_answer_label(expected)


def _premise_scores(row: dict[str, Any], response: QAResponse) -> tuple[float | None, float | None, bool]:
    gold = {str(p).upper() for p in row.get("gold_premises") or []}
    predicted = {str(p).upper() for p in response.premises or []}
    available = {
        match.group(1).upper()
        for premise in row.get("premises") or []
        if (match := re.match(r"^(P\d+)\s*:", str(premise).strip(), re.I))
    }
    hallucinated = bool(available and predicted - available)
    if not gold:
        return None, None, hallucinated
    precision = len(predicted & gold) / len(predicted) if predicted else 0.0
    recall = len(predicted & gold) / len(gold)
    return precision, recall, hallucinated


def _formula_correct(row: dict[str, Any], response: QAResponse) -> bool | None:
    expected = row.get("formula_id")
    if not expected:
        return None
    return response.fol == expected


def _explanation_consistent(row: dict[str, Any], response: QAResponse) -> bool:
    explanation = str(response.explanation or "")
    if not explanation.strip():
        return False
    normalized = explanation.lower()
    answer = normalize_answer_label(response.answer)
    if answer == "yes" and "answer is no" in normalized:
        return False
    if answer == "no" and "answer is yes" in normalized:
        return False
    if answer == "unknown" and not any(token in normalized for token in ["unknown", "not enough", "missing", "does not entail", "cannot"]):
        return False
    if row.get("task_type") == "physics":
        if answer == "unknown":
            return any(token in normalized for token in ["unknown", "not enough", "missing", "no deterministic", "unsupported", "cannot"])
        return bool(response.fol and (response.fol in explanation or "=" in explanation or "formula" in normalized or "used" in normalized))
    if response.premises:
        cited = {match.upper() for match in re.findall(r"\bP\d+\b", explanation, re.I)}
        if not set(response.premises).issubset(cited):
            return False
    return True


def confidence_bucket(response: QAResponse) -> str:
    if response.confidence >= 0.85:
        return "high"
    if response.confidence >= 0.6:
        return "medium"
    return "low"


def score_response(row: dict[str, Any], response: QAResponse, latency_ms: float) -> QualityScore:
    if row.get("task_type") == "physics":
        answer_correct, numeric_correct, unit_correct = _score_physics(row, response)
    else:
        answer_correct = _score_logic(row, response)
        numeric_correct = None
        unit_correct = None
    premise_precision, premise_recall, hallucinated = _premise_scores(row, response)
    return QualityScore(
        answer_correct=answer_correct,
        numeric_correct=numeric_correct,
        unit_correct=unit_correct,
        premise_precision=premise_precision,
        premise_recall=premise_recall,
        formula_correct=_formula_correct(row, response),
        hallucinated_premise=hallucinated,
        explanation_consistent=_explanation_consistent(row, response),
        confidence_bucket=confidence_bucket(response),
        latency_ms=latency_ms,
    )
