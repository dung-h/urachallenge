from __future__ import annotations

import os
from typing import Any

from app.llm_client import FallbackClient, OpenAICompatibleLLMClient
from app.pipeline_config import PipelineConfig


def build_runtime_llm_client(config: PipelineConfig, enabled: bool) -> FallbackClient:
    base_url = os.environ.get("URA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    base_url = base_url or "http://127.0.0.1:8001/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    model = os.environ.get("URA_LLM_MODEL") or config.reasoner_model
    timeout = float(os.environ.get("URA_LLM_TIMEOUT", "120"))
    return OpenAICompatibleLLMClient(base_url=base_url, model=model, timeout=timeout, enabled=enabled)


def ensure_runtime_llm_client(
    config: PipelineConfig,
    llm_client: FallbackClient | None,
    enabled: bool,
) -> FallbackClient | None:
    if llm_client is not None:
        return llm_client
    if not enabled:
        return None
    return build_runtime_llm_client(config, enabled=True)


def llm_client_info(llm_client: FallbackClient | None) -> dict[str, Any]:
    if llm_client is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "client_type": type(llm_client).__name__,
        "model": getattr(llm_client, "model", None),
        "base_url": getattr(llm_client, "base_url", None),
    }
