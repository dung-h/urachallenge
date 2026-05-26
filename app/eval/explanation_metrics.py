from __future__ import annotations

import re
from typing import Any
from app.explanation_worker import _norm, _formula_hints, _answer_hints
from app.schemas import normalize_answer_label

def check_cites_selected_premise(explanation: str, selected_premises: list[str], task_type: str) -> bool:
    """Checks if logic explanation cites at least one of the selected premises when they exist."""
    if task_type != "logic" or not selected_premises:
        return True
    present_ids = {match.upper() for match in re.findall(r"\bP\d+\b", explanation, re.I)}
    selected_ids = {str(pid).strip().upper() for pid in selected_premises if str(pid).strip()}
    if not selected_ids:
        return True
    return bool(present_ids & selected_ids)

def check_cites_formula(explanation: str, fol: str | None, formula_expression: str | None, task_type: str) -> bool:
    """Checks if physics explanation cites the formula/method when it exists."""
    if task_type != "physics" or not fol:
        return True
    hints = _formula_hints(fol, formula_expression)
    if not hints:
        return True
    normalized_explanation = _norm(explanation)
    for hint in hints:
        if len(hint) == 1:
            if re.search(r"\b" + re.escape(hint) + r"\b", normalized_explanation):
                return True
        elif hint in normalized_explanation:
            return True
    return False

def check_no_hallucinated_premises(explanation: str, selected_premises: list[str], task_type: str) -> bool:
    """Checks that logic explanation does not mention premise IDs not present in the selected set."""
    if task_type != "logic":
        return True
    present_ids = {match.upper() for match in re.findall(r"\bP\d+\b", explanation, re.I)}
    selected_ids = {str(pid).strip().upper() for pid in selected_premises if str(pid).strip()}
    # If any P-ID in explanation is NOT in selected_premises, it is hallucinated
    hallucinated = present_ids - selected_ids
    return not hallucinated

def check_no_hallucinated_formulas(explanation: str, fol: str | None, task_type: str) -> bool:
    """Checks that physics explanation does not mention formula/evidence not in the trace."""
    if task_type != "physics":
        return True
    normalized_explanation = _norm(explanation)
    
    # Check known formulas other than the one used
    from app.physics.formulas import FORMULAS
    for fid in FORMULAS.keys():
        if fol and fid == fol:
            continue
        # Use simpler matching for hallucination check
        fid_norm = fid.lower()
        if fid_norm in normalized_explanation and len(fid_norm) > 5:
            # If the explanation mentions a formula name not in trace
            return False
    return True

def check_answer_consistency(explanation: str, answer: str) -> bool:
    """Checks that the explanation does not contradict the final answer."""
    explanation_norm = _norm(explanation)
    answer_norm = normalize_answer_label(answer)
    
    if answer_norm == "yes" and re.search(r"\bno\b", explanation_norm) and "yes" not in explanation_norm:
        return False
    if answer_norm == "no" and re.search(r"\byes\b", explanation_norm) and "no" not in explanation_norm:
        return False
    if answer_norm == "unknown" and not any(token in explanation_norm for token in ["unknown", "not enough", "missing", "does not entail", "cannot", "insufficient", "unsupported", "contradict"]):
        return False
    return True

def check_unknown_specificity(explanation: str, answer: str, trace_reason: str | None) -> bool:
    """Checks if unknown explanation specifically cites the missing/contradictory condition from the trace."""
    if normalize_answer_label(answer) != "unknown":
        return True
        
    explanation_norm = _norm(explanation)
    
    # Specific reason check
    reasons = ["missing condition", "contradiction", "unsupported topology", "missing quantity", "unrelated circuit", "singular", "insufficient evidence", "lacks a concrete witness", "not triggered", "does not establish"]
    # If trace_reason is known, check for overlap
    if trace_reason:
        trace_reason_norm = _norm(trace_reason)
        # Check if explanation contains keywords from trace_reason
        keywords = [w for w in re.findall(r"\b[a-z]{4,}\b", trace_reason_norm) if w not in {"answer", "unknown", "because", "rule", "used", "evidence"}]
        if keywords and any(kw in explanation_norm for kw in keywords):
            return True
            
    # Generic fallback checks
    return any(r in explanation_norm for r in reasons) or any(w in explanation_norm for w in ["missing", "contradict", "unsupported", "insufficient", "not record", "not provide", "not trigger", "not match"])

def check_prompt_echo_json_leak(explanation: str) -> bool:
    """Checks for prompt echo or raw JSON leaks in the explanation."""
    explanation_norm = _norm(explanation)
    
    # Prompt Echoing patterns
    prompt_patterns = [
        "you are an explanation worker",
        "rewrite the backend trace",
        "without changing the final answer",
        "return json with a single key",
        "explanation_rewrite",
        "do not add new facts",
        "explanation_trace",
        "solver_explanation",
        "trace_version",
        "public_cot",
        "solver_used",
        "selected_premise_ids",
    ]
    for pattern in prompt_patterns:
        if pattern in explanation_norm:
            return False
            
    # JSON Structures / Leakage
    if explanation.strip().startswith("{") or "explanation\"" in explanation_norm or re.search(r"explanation\s*:", explanation_norm):
        return False
        
    return True

def evaluate_explanation_grounding(explanation: str, response: Any, metadata: dict[str, Any]) -> dict[str, bool]:
    """Helper to run all grounding checks and return a breakdown dictionary."""
    task_type = response.task_type
    answer = response.answer
    fol = response.fol
    selected_premises = list(response.premises or [])
    
    # Resolve formula expression if exists
    formula_expression = None
    if task_type == "physics" and fol:
        from app.physics.formulas import FORMULAS
        formula = FORMULAS.get(fol)
        if formula:
            formula_expression = formula.expression
            
    trace_reason = None
    if task_type == "physics":
        trace_reason = metadata.get("fallback_rejected_reason") or (metadata.get("physics_problem_frame", {}).get("geometry") if isinstance(metadata.get("physics_problem_frame"), dict) else None)
    elif task_type == "logic":
        # Extract notes from proof steps
        proof_steps = metadata.get("proof_steps") or []
        if proof_steps and isinstance(proof_steps, list) and isinstance(proof_steps[0], dict):
            trace_reason = proof_steps[0].get("notes")
            
    return {
        "cites_selected_premise": check_cites_selected_premise(explanation, selected_premises, task_type),
        "cites_formula": check_cites_formula(explanation, fol, formula_expression, task_type),
        "no_hallucinated_premises": check_no_hallucinated_premises(explanation, selected_premises, task_type),
        "no_hallucinated_formulas": check_no_hallucinated_formulas(explanation, fol, task_type),
        "answer_consistency": check_answer_consistency(explanation, answer),
        "unknown_specificity": check_unknown_specificity(explanation, answer, trace_reason),
        "no_prompt_echo_or_leak": check_prompt_echo_json_leak(explanation),
    }
