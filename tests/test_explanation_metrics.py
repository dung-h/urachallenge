from __future__ import annotations

from app.eval.explanation_metrics import (
    check_cites_selected_premise,
    check_cites_formula,
    check_no_hallucinated_premises,
    check_no_hallucinated_formulas,
    check_answer_consistency,
    check_unknown_specificity,
    check_prompt_echo_json_leak,
)
from app.schemas import QAResponse

def test_check_cites_selected_premise() -> None:
    # Logic task: premise exists and is mentioned
    assert check_cites_selected_premise("Based on P1 Maya studies", ["P1"], "logic") is True
    # Logic task: premise exists but not mentioned
    assert check_cites_selected_premise("Based on P2 Maya passes", ["P1"], "logic") is False
    # Non-logic task or no premises
    assert check_cites_selected_premise("Some text", [], "physics") is True

def test_check_cites_formula() -> None:
    # Physics task: formula mentioned
    assert check_cites_formula("Using ohms_law we compute...", "ohms_law", "V = I * R", "physics") is True
    # Physics task: formula not mentioned
    assert check_cites_formula("Using some other method...", "ohms_law", "V = I * R", "physics") is False
    # Non-physics task
    assert check_cites_formula("Using something", "ohms_law", "V = I * R", "logic") is True

def test_check_no_hallucinated_premises() -> None:
    # No hallucinated premises (P1 is selected and mentioned)
    assert check_no_hallucinated_premises("Maya passes by P1", ["P1"], "logic") is True
    # Hallucinated premise (P2 is mentioned but not selected)
    assert check_no_hallucinated_premises("Maya passes by P1 and P2", ["P1"], "logic") is False

def test_check_no_hallucinated_formulas() -> None:
    # No hallucination
    assert check_no_hallucinated_formulas("Applying ohms_law to get...", "ohms_law", "physics") is True
    # Hallucination of parallel_resistance
    assert check_no_hallucinated_formulas("Applying ohms_law and parallel_resistance", "ohms_law", "physics") is False

def test_check_answer_consistency() -> None:
    assert check_answer_consistency("The answer is yes", "yes") is True
    assert check_answer_consistency("The answer is no", "yes") is False
    assert check_answer_consistency("The answer is unknown because of missing info", "unknown") is True
    assert check_answer_consistency("The answer is yes", "unknown") is False

def test_check_unknown_specificity() -> None:
    assert check_unknown_specificity("unknown because of missing condition", "unknown", "missing condition") is True
    assert check_unknown_specificity("unknown", "unknown", None) is False

def test_check_prompt_echo_json_leak() -> None:
    # Leak case
    assert check_prompt_echo_json_leak('{"explanation": "answer is yes"}') is False
    # Normal case
    assert check_prompt_echo_json_leak("The student is eligible because they have GPA 3.7.") is True
