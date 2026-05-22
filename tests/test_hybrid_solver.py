from app.logic.hybrid_solver import solve_hybrid
from app.pipeline_config import PipelineConfig
from app.router import predict_with_metadata
from app.schemas import QARequest


def test_hybrid_solver_proves_safe_universal_yes() -> None:
    result = solve_hybrid(
        "Is Tweety a bird?",
        ["P1: All robins are birds.", "P2: Tweety is a robin."],
    )
    assert result.answer == "yes"
    assert result.z3_status == "entailed"
    assert result.conclusion_fol == "bird(tweety)"
    assert "P1 says" in result.explanation
    assert "P2 says" in result.explanation
    assert "Applying P1 to Tweety" in result.explanation
    assert "Therefore the answer is yes" in result.explanation


def test_hybrid_solver_proves_safe_universal_no() -> None:
    result = solve_hybrid(
        "Is Tweety a mammal?",
        ["P1: No robins are mammals.", "P2: Tweety is a robin."],
    )
    assert result.answer == "no"
    assert result.z3_status == "contradicted"
    assert "Applying P1 to Tweety" in result.explanation
    assert "rules out" in result.explanation
    assert "Therefore the answer is no" in result.explanation


def test_hybrid_solver_does_not_flip_required_condition() -> None:
    result = solve_hybrid(
        "Is Mira eligible for the award?",
        [
            "P1: Eligibility for the award requires submitting a portfolio.",
            "P2: Mira submitted a portfolio.",
        ],
    )
    assert result.answer == "unknown"
    assert result.z3_status == "not_entailed"
    assert "P1 says" in result.explanation
    assert "P2 says" in result.explanation
    assert "opposite direction" in result.explanation
    assert "answer is unknown" in result.explanation


def test_router_hybrid_path_uses_z3_without_counting_model_call() -> None:
    response, metadata = predict_with_metadata(
        QARequest(
            task_type="logic",
            question="Is Mira eligible for the award?",
            premises=[
                "P1: Eligibility for the award requires submitting a portfolio.",
                "P2: Mira submitted a portfolio.",
            ],
        ),
        config=PipelineConfig(enable_hybrid_solver=True),
    )
    assert response.answer == "unknown"
    assert response.fol == "eligible_award(mira)"
    assert metadata["solver_used"] == "hybrid_rules_to_z3"
    assert metadata["model_calls"] == 0
