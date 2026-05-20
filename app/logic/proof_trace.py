from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from app.logic.premise_selector import Premise


ALLOWED_RULES = {
    "direct_fact",
    "modus_ponens",
    "multi_condition_rule",
    "threshold_check",
    "contradiction_detected",
    "exception_override",
    "missing_required_condition",
    "invalid_inference_blocked",
    "unknown_due_to_insufficient_evidence",
}

UNKNOWN_EVIDENCE_RULES = {
    "missing_required_condition",
    "invalid_inference_blocked",
    "unknown_due_to_insufficient_evidence",
    "contradiction_detected",
}


@dataclass(frozen=True)
class ProofStep:
    step_id: str
    rule: str
    input_premises: list[str] = field(default_factory=list)
    input_steps: list[str] = field(default_factory=list)
    derived: str = ""
    status: str = "validated"
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def proof_steps_to_dicts(steps: list[ProofStep]) -> list[dict[str, Any]]:
    return [step.to_dict() for step in steps]


def _rule_from_reason(reason: str, answer: str) -> str:
    low = reason.lower()
    if "contradict" in low or "support both" in low:
        return "contradiction_detected"
    if "exception" in low or "negative fact overrides" in low:
        return "exception_override"
    if "missing" in low or "absent" in low:
        return "missing_required_condition"
    if "negated antecedent" in low or "affirming" in low or "conditional premise" in low or "consequent" in low:
        return "invalid_inference_blocked"
    if "modus ponens" in low:
        return "modus_ponens"
    if "threshold" in low:
        return "threshold_check"
    if "academic policy rule" in low or "condition" in low:
        return "multi_condition_rule"
    if answer == "unknown" or "some " in low or "existential" in low or "no deterministic" in low or "insufficient" in low:
        return "unknown_due_to_insufficient_evidence"
    return "direct_fact"


def build_proof_steps(answer: str, selected: list[Premise], reason: str, confidence: float) -> list[ProofStep]:
    rule = _rule_from_reason(reason, answer)
    status = "validated"
    if rule in {"missing_required_condition", "unknown_due_to_insufficient_evidence"}:
        status = "missing"
    elif rule == "invalid_inference_blocked":
        status = "blocked"
    elif rule == "contradiction_detected":
        status = "conflict"
    premise_ids = [premise.id for premise in selected]
    derived = f"answer:{answer}"
    return [
        ProofStep(
            step_id="S1",
            rule=rule,
            input_premises=premise_ids,
            input_steps=[],
            derived=derived,
            status=status,
            confidence=confidence,
            notes=reason,
        )
    ]


def validate_proof_steps(
    steps: list[ProofStep] | list[dict[str, Any]],
    premise_ids: set[str],
    answer: str | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    seen_steps: set[str] = set()
    normalized_steps: list[dict[str, Any]] = [step.to_dict() if isinstance(step, ProofStep) else step for step in steps]
    for step in normalized_steps:
        step_id = str(step.get("step_id") or "")
        rule = str(step.get("rule") or "")
        if not step_id:
            errors.append("missing_step_id")
        elif step_id in seen_steps:
            errors.append(f"duplicate_step_id:{step_id}")
        if rule not in ALLOWED_RULES:
            errors.append(f"invalid_rule:{rule}")
        for premise_id in step.get("input_premises") or []:
            if premise_id not in premise_ids:
                errors.append(f"unknown_input_premise:{premise_id}")
        for input_step in step.get("input_steps") or []:
            if input_step not in seen_steps:
                errors.append(f"unknown_or_forward_input_step:{input_step}")
        derived = str(step.get("derived") or "")
        if not derived.strip():
            errors.append(f"missing_derived:{step_id}")
        for cited_premise in re.findall(r"\bP\d+\b", derived):
            if cited_premise not in premise_ids:
                errors.append(f"unknown_derived_premise:{cited_premise}")
        if step_id:
            seen_steps.add(step_id)
    if answer == "unknown" and normalized_steps:
        if not any(step.get("rule") in UNKNOWN_EVIDENCE_RULES for step in normalized_steps):
            errors.append("unknown_without_missing_blocked_or_insufficient_step")
    if answer == "unknown" and not normalized_steps:
        errors.append("unknown_without_proof_step")
    if answer == "no" and any("exception" in str(step.get("notes") or "").lower() for step in normalized_steps):
        if not any(step.get("rule") == "exception_override" for step in normalized_steps):
            errors.append("exception_without_exception_override_step")
    if answer == "unknown" and any("contradict" in str(step.get("notes") or "").lower() for step in normalized_steps):
        if not any(step.get("rule") == "contradiction_detected" for step in normalized_steps):
            errors.append("contradiction_without_contradiction_step")
    return not errors, errors


def proof_depth(steps: list[ProofStep] | list[dict[str, Any]]) -> int:
    normalized_steps: list[dict[str, Any]] = [step.to_dict() if isinstance(step, ProofStep) else step for step in steps]
    if not normalized_steps:
        return 0
    return max(1, len(normalized_steps))
