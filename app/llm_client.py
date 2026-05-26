from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json
import re
import time

import httpx
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "configs" / "prompts.yaml"


class LLMClientNotConfigured(RuntimeError):
    """Raised when inference is requested before a local backend is configured."""


ROLE_CONFIG = {
    "router_formatter": "Gemma 4 E2B-it",
    "main_orchestrator": "Qwen3.5-4B",
    "reasoner": "DeepSeek-R1-Distill-Qwen-7B",
    "secondary_formatter": "Qwen3-4B",
}


@dataclass
class LLMResult:
    content: str
    raw_json_validity: bool = False
    repaired_json_validity: bool = False
    error: str | None = None


class FallbackClient(Protocol):
    def orchestrate(self, payload: dict[str, Any]) -> dict[str, Any] | None: ...

    def plan_physics_action(self, payload: dict[str, Any]) -> dict[str, Any] | None: ...

    def plan_logic_action(self, payload: dict[str, Any]) -> dict[str, Any] | None: ...

    def suggest_physics(self, question: str) -> dict[str, Any] | None: ...

    def suggest_logic(self, question: str, premises: list[str]) -> dict[str, Any] | None: ...

    def rewrite_explanation(self, trace: dict[str, Any]) -> str | None: ...


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("prompts", {}) or {}


class OpenAICompatibleLLMClient:
    def __init__(self, base_url: str | None = None, model: str = "local", timeout: float = 120.0, enabled: bool = False) -> None:
        self.base_url = (base_url or "http://127.0.0.1:8001/v1").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        self.prompts = load_prompts()
        self.call_traces: list[dict[str, Any]] = []

    def chat(self, role: str, user: str, max_tokens: int = 256, response_format: bool = False) -> LLMResult:
        if not self.enabled:
            raise LLMClientNotConfigured("LLM fallback is disabled by default for Phase 5 baseline.")
        system = self.prompts.get(role, self.prompts.get("default", "Answer concisely."))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "backend": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "role": role,
            "max_tokens": max_tokens,
            "response_format": "json_object" if response_format else None,
            "system_prompt": system,
            "user_prompt": user,
            "status": "started",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
                obj = response.json()
        except Exception as exc:
            trace.update(
                {
                    "status": "error",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": str(exc),
                }
            )
            self.call_traces.append(trace)
            return LLMResult(content="", error=str(exc))
        choices = obj.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = message.get("content") or message.get("reasoning_content") or ""
        trace.update(
            {
                "status": "ok",
                "latency_ms": (time.perf_counter() - started) * 1000,
                "raw_response": content,
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "usage": obj.get("usage"),
            }
        )
        self.call_traces.append(trace)
        return LLMResult(content=content)

    def _json_chat(self, role: str, user: str, max_tokens: int = 256) -> dict[str, Any] | None:
        result = self.chat(role, user, max_tokens=max_tokens, response_format=True)
        if result.error or not result.content.strip():
            if self.call_traces:
                self.call_traces[-1]["json_validity"] = False
                self.call_traces[-1]["json_parse_error"] = result.error or "empty_response"
            return None
        try:
            parsed = json.loads(result.content)
            if self.call_traces:
                self.call_traces[-1]["json_validity"] = True
                self.call_traces[-1]["repaired_json_validity"] = False
            return parsed
        except Exception as exc:
            first_error = str(exc)
            match = re.search(r"\{.*\}", result.content, re.S)
            if not match:
                if self.call_traces:
                    self.call_traces[-1]["json_validity"] = False
                    self.call_traces[-1]["repaired_json_validity"] = False
                    self.call_traces[-1]["json_parse_error"] = first_error
                return None
            try:
                parsed = json.loads(match.group(0))
                if self.call_traces:
                    self.call_traces[-1]["json_validity"] = False
                    self.call_traces[-1]["repaired_json_validity"] = True
                return parsed
            except Exception as repair_exc:
                if self.call_traces:
                    self.call_traces[-1]["json_validity"] = False
                    self.call_traces[-1]["repaired_json_validity"] = False
                    self.call_traces[-1]["json_parse_error"] = first_error
                self.call_traces[-1]["json_repair_error"] = str(repair_exc)
                return None

    def orchestrate(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._json_chat("main_orchestrator", json.dumps(payload, sort_keys=True), max_tokens=320)

    def plan_physics_action(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._json_chat("physics_agent_planner", json.dumps(payload, sort_keys=True), max_tokens=220)

    def plan_logic_action(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._json_chat("logic_agent_planner", json.dumps(payload, sort_keys=True), max_tokens=220)

    def suggest_physics(self, question: str) -> dict[str, Any] | None:
        prompt = (
            f"Question: {question}\n"
            "Return JSON with target_quantity, optional formula_id, optional expression, variables in SI units, and target_unit. "
            "If you cannot name a known formula_id, return a single evaluable expression for the answer."
        )
        return self._json_chat("physics_formula_assistant", prompt)

    def suggest_logic(self, question: str, premises: list[str]) -> dict[str, Any] | None:
        prompt = "Premises:\n" + "\n".join(premises) + f"\nQuestion: {question}\nReturn JSON with answer, used_premise_ids, and reason_short."
        return self._json_chat("logic_reasoner", prompt)

    def rewrite_explanation(self, trace: dict[str, Any]) -> str | None:
        result = self.chat("explanation_rewrite", json.dumps(trace, sort_keys=True), max_tokens=180, response_format=True)
        if result.error or not result.content.strip():
            return None
        content = result.content.strip()
        try:
            obj = json.loads(content)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("explanation", "text", "content"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
        return content
