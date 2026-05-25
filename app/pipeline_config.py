from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG = ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    enable_mcq_symbolic: bool = False
    enable_physics_web_search: bool = True
    # Hybrid solver requires an external local LLM endpoint and Z3.
    # Keep off by default so a fresh install can run the API/UI without extra deps.
    enable_hybrid_solver: bool = False
    hybrid_api_url: str = "http://localhost:11434/v1/chat/completions"
    hybrid_model: str = "gemma3:4b"
    fallback_confidence_threshold: float = 0.7
    deterministic_physics_authority: bool = True
    validate_premise_ids: bool = True
    validate_final_json: bool = True
    reasoner_model: str = "deepseek_r1_distill_qwen_7b"
    formatter_model: str = "gemma4_e2b_it"
    secondary_formatter_model: str = "qwen3_4b"
    enable_z3_sidecar: bool = False
    z3_sidecar_mode: str = "experiment_only"
    z3_allowed_domains: tuple[str, ...] = ("academic_policy", "public_logic_sample")


def load_pipeline_config(path: Path = PIPELINE_CONFIG) -> PipelineConfig:
    if not path.exists():
        return PipelineConfig()
    data = yaml.safe_load(path.read_text()) or {}
    values = data.get("pipeline", {}) or {}
    return PipelineConfig(
        enable_mcq_symbolic=bool(values.get("enable_mcq_symbolic", False)),
        enable_physics_web_search=bool(values.get("enable_physics_web_search", True)),
        enable_hybrid_solver=bool(values.get("enable_hybrid_solver", False)),
        hybrid_api_url=str(values.get("hybrid_api_url", "http://localhost:11434/v1/chat/completions")),
        hybrid_model=str(values.get("hybrid_model", "gemma3:4b")),
        fallback_confidence_threshold=float(values.get("fallback_confidence_threshold", 0.7)),
        deterministic_physics_authority=bool(values.get("deterministic_physics_authority", True)),
        validate_premise_ids=bool(values.get("validate_premise_ids", True)),
        validate_final_json=bool(values.get("validate_final_json", True)),
        reasoner_model=str(values.get("reasoner_model", "deepseek_r1_distill_qwen_7b")),
        formatter_model=str(values.get("formatter_model", "gemma4_e2b_it")),
        secondary_formatter_model=str(values.get("secondary_formatter_model", "qwen3_4b")),
        enable_z3_sidecar=bool(values.get("enable_z3_sidecar", False)),
        z3_sidecar_mode=str(values.get("z3_sidecar_mode", "experiment_only")),
        z3_allowed_domains=tuple(values.get("z3_allowed_domains", ["academic_policy", "public_logic_sample"])),
    )
