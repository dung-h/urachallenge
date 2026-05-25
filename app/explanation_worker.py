from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any

from app.physics.formulas import get_formula
from app.schemas import normalize_answer_label

SPECIAL_FORMULA_HINTS: dict[str, tuple[str, str]] = {
    "rlc_resonance_impedance": ("Z = R", "ohm"),
    "transformer_secondary_voltage": ("V_secondary / V_primary = N_secondary / N_primary", "V"),
}


def _norm(text: Any) -> str:
    value = str(text or "").lower()
    value = value.replace("ω", "omega").replace("Ω", "ohm").replace("Ω", "ohm").replace("µ", "u")
    value = re.sub(r"[^a-z0-9./+\-=_\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _answer_hints(answer: str) -> list[str]:
    normalized = normalize_answer_label(answer)
    if normalized == "yes":
        return ["yes", "affirmative"]
    if normalized == "no":
        return ["no", "negative"]
    if normalized == "unknown":
        return [
            "unknown",
            "undetermined",
            "cannot determine",
            "cannot conclude",
            "not enough information",
            "insufficient evidence",
            "insufficient",
        ]
    if re.fullmatch(r"[A-E]", normalized):
        return [normalized.lower(), f"option {normalized.lower()}", f"choice {normalized.lower()}"]

    raw = _norm(answer)
    hints = [raw]
    unit_synonyms = {
        "v": ["v", "volt", "volts"],
        "a": ["a", "amp", "amps", "ampere", "amperes"],
        "ohm": ["ohm", "ohms"],
        "n": ["n", "newton", "newtons"],
        "w": ["w", "watt", "watts"],
        "j": ["j", "joule", "joules"],
        "c": ["c", "coulomb", "coulombs"],
        "f": ["f", "farad", "farads"],
        "hz": ["hz", "hertz"],
        "rad/s": ["rad/s", "radians per second"],
        "%": ["%", "percent", "percentage"],
    }
    parts = [part for part in re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+/=]", raw) if part]
    for part in parts:
        if part not in hints:
            hints.append(part)
        for synonym in unit_synonyms.get(part, []):
            if synonym not in hints:
                hints.append(synonym)
    return [hint for hint in hints if hint]


def _formula_hints(formula_id: str | None, formula_expression: str | None) -> list[str]:
    hints: list[str] = []
    if formula_id:
        hints.append(_norm(formula_id))
    if formula_expression:
        hints.append(_norm(formula_expression))
        tokens = [token for token in re.findall(r"[a-z0-9_]+", _norm(formula_expression)) if token]
        for token in tokens:
            if token not in hints:
                hints.append(token)
    return hints


@dataclass(frozen=True)
class ExplanationTrace:
    trace_version: str
    task_type: str
    request_id: str | None
    question: str
    answer: str
    solver_explanation: str
    fol: str | None = None
    formula_expression: str | None = None
    formula_target_unit: str | None = None
    selected_premise_ids: list[str] = field(default_factory=list)
    selected_premise_texts: list[str] = field(default_factory=list)
    public_cot: list[str] = field(default_factory=list)
    proof_steps: list[dict[str, Any]] = field(default_factory=list)
    physics_variables: dict[str, Any] = field(default_factory=dict)
    solver_used: str | None = None
    confidence: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_explanation_trace(
    *,
    request_id: str | None,
    question: str,
    task_type: str,
    answer: str,
    explanation: str,
    fol: str | None,
    selected_premise_ids: list[str],
    selected_premise_texts: list[str],
    cot: list[str],
    proof_steps: list[Any],
    physics_variables: dict[str, Any] | None = None,
    solver_used: str | None = None,
    confidence: float = 0.0,
) -> ExplanationTrace:
    formula_expression = None
    formula_target_unit = None
    if task_type == "physics" and fol:
        special_hint = SPECIAL_FORMULA_HINTS.get(fol)
        if special_hint:
            formula_expression, formula_target_unit = special_hint
        else:
            try:
                formula = get_formula(fol)
            except KeyError:
                formula = None
            if formula is not None:
                formula_expression = formula.expression
                formula_target_unit = formula.target_unit

    normalized_proof_steps: list[dict[str, Any]] = []
    for step in proof_steps or []:
        if hasattr(step, "to_dict"):
            normalized_proof_steps.append(step.to_dict())
        elif isinstance(step, dict):
            normalized_proof_steps.append(dict(step))
        else:
            normalized_proof_steps.append({"value": str(step)})
    return ExplanationTrace(
        trace_version="v1",
        task_type=task_type,
        request_id=request_id,
        question=question,
        answer=answer,
        solver_explanation=explanation,
        fol=fol,
        formula_expression=formula_expression,
        formula_target_unit=formula_target_unit,
        selected_premise_ids=[pid for pid in selected_premise_ids if pid],
        selected_premise_texts=[text for text in selected_premise_texts if text],
        public_cot=[step for step in cot if str(step or "").strip()],
        proof_steps=normalized_proof_steps,
        physics_variables=physics_variables or {},
        solver_used=solver_used,
        confidence=confidence,
    )


def validate_explanation_rewrite(explanation: str, trace: ExplanationTrace | dict[str, Any]) -> tuple[bool, list[str]]:
    payload = trace.to_payload() if isinstance(trace, ExplanationTrace) else dict(trace)
    errors: list[str] = []
    rewritten = str(explanation or "").strip()
    if not rewritten:
        return False, ["empty_explanation"]

    normalized = _norm(rewritten)
    solver_explanation = _norm(payload.get("solver_explanation") or "")
    answer = str(payload.get("answer") or "").strip()
    answer_hints = _answer_hints(answer)
    if answer_hints and not any(hint in normalized for hint in answer_hints):
        errors.append("missing_answer_reference")

    task_type = str(payload.get("task_type") or "").strip()
    if task_type == "physics":
        formula_hints = _formula_hints(
            str(payload.get("fol") or "").strip() or None,
            str(payload.get("formula_expression") or "").strip() or None,
        )
        if formula_hints and not any(hint in normalized for hint in formula_hints):
            errors.append("missing_formula_reference")
        fol = _norm(payload.get("fol") or "")
        if "perpendicular_bisector" in fol and "perpendicular" not in normalized:
            errors.append("missing_geometry_reference:perpendicular")
    elif task_type == "logic":
        selected_ids = [str(pid).strip().upper() for pid in payload.get("selected_premise_ids") or [] if str(pid).strip()]
        if selected_ids:
            present_ids = {match.upper() for match in re.findall(r"\bP\d+\b", rewritten, re.I)}
            missing_ids = [pid for pid in selected_ids if pid not in present_ids]
            if missing_ids:
                errors.append("missing_premise_ids:" + ",".join(missing_ids))
            foreign_ids = sorted(present_ids - set(selected_ids))
            if foreign_ids:
                errors.append("foreign_premise_ids:" + ",".join(foreign_ids))
        proof_notes = " ".join(_norm(step.get("notes") or "") for step in payload.get("proof_steps") or [] if isinstance(step, dict))
        for protected_phrase in ("missing faculty nomination condition", "existential witness"):
            if (protected_phrase in proof_notes or protected_phrase in solver_explanation) and protected_phrase not in normalized:
                errors.append("missing_solver_trace_phrase:" + protected_phrase)

    if answer.lower() == "yes" and re.search(r"\bno\b", normalized) and "yes" not in normalized:
        errors.append("answer_contradiction:no")
    elif answer.lower() == "no" and re.search(r"\byes\b", normalized) and "no" not in normalized:
        errors.append("answer_contradiction:yes")

    return not errors, errors
