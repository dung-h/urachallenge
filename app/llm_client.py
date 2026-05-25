from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json
import re
import subprocess
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


class OpenCodeCLIClient:
    """Fallback worker that asks `opencode run` for JSON proposals.

    OpenCode remains a proposal worker here: outputs are parsed as JSON and then
    validated/recomputed by backend solvers before they can affect `/predict`.
    """

    def __init__(
        self,
        model: str = "ollama/llama3.2:3b",
        timeout: float = 120.0,
        enabled: bool = False,
        project_dir: str | None = None,
        command: str = "opencode",
        agent: str | None = None,
    ) -> None:
        self.model = model if "/" in model else f"ollama/{model}"
        self.timeout = timeout
        self.enabled = enabled
        self.project_dir = project_dir or str(ROOT)
        self.command = command
        self.agent = agent
        self.prompts = load_prompts()
        self.call_traces: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        agent_part = f" --agent {self.agent}" if self.agent else ""
        return f"opencode run{agent_part} --model {self.model}"

    def chat(self, role: str, user: str, max_tokens: int = 256, response_format: bool = False) -> LLMResult:
        if not self.enabled:
            raise LLMClientNotConfigured("OpenCode fallback is disabled by default.")
        system = self.prompts.get(role, self.prompts.get("default", "Answer concisely."))
        prompt = (
            f"System instruction:\n{system}\n\n"
            f"User task:\n{user}\n\n"
            "Return only the requested final content. Do not edit files or run commands."
        )
        if response_format:
            prompt += "\nReturn valid JSON only."
        cmd = [self.command, "run", "--model", self.model]
        if self.agent:
            cmd.extend(["--agent", self.agent])
        cmd.extend(["--format", "json", "--dir", self.project_dir, prompt])
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "backend": "opencode_cli",
            "model": self.model,
            "agent": self.agent,
            "role": role,
            "max_tokens": max_tokens,
            "response_format": "json_object" if response_format else None,
            "system_prompt": system,
            "user_prompt": user,
            "command": " ".join(cmd[:7] + ["..."]),
            "status": "started",
        }
        try:
            completed = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except Exception as exc:
            trace.update({"status": "error", "latency_ms": (time.perf_counter() - started) * 1000, "error": str(exc)})
            self.call_traces.append(trace)
            return LLMResult(content="", error=str(exc))

        content = self._extract_text_from_events(completed.stdout)
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            error = stderr or content or f"opencode exited with {completed.returncode}"
            trace.update(
                {
                    "status": "error",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "returncode": completed.returncode,
                    "raw_response": content,
                    "stderr": stderr[:1000],
                    "error": error[:1000],
                }
            )
            self.call_traces.append(trace)
            return LLMResult(content=content, error=error)

        trace.update(
            {
                "status": "ok",
                "latency_ms": (time.perf_counter() - started) * 1000,
                "raw_response": content,
                "stderr": stderr[:1000],
            }
        )
        self.call_traces.append(trace)
        return LLMResult(content=content)

    @staticmethod
    def _extract_text_from_events(stdout: str) -> str:
        parts: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                parts.append(line)
                continue
            part = event.get("part") if isinstance(event, dict) else None
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _json_objects_from_text(text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", text):
            try:
                parsed, _end = decoder.raw_decode(text[match.start() :])
            except Exception:
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects

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
            candidates = self._json_objects_from_text(result.content)
            if not candidates:
                if self.call_traces:
                    self.call_traces[-1]["json_validity"] = False
                    self.call_traces[-1]["repaired_json_validity"] = False
                    self.call_traces[-1]["json_parse_error"] = first_error
                return None
            try:
                parsed = candidates[-1]
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
        try:
            obj = json.loads(result.content)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            value = obj.get("explanation") or obj.get("text") or obj.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return result.content.strip()


class HuggingFaceLLMClient:
    """A lightweight Hugging Face local/client-backed LLM wrapper.

    This will attempt to use `transformers` locally. If `transformers` or
    required model files are not available, construction raises
    `LLMClientNotConfigured`.
    """

    def __init__(self, model: str = "gpt2", timeout: float = 120.0, enabled: bool = False) -> None:
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        self.prompts = load_prompts()
        self.call_traces: list[dict[str, Any]] = []
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise LLMClientNotConfigured(f"transformers not available: {exc}")

        device = 0 if torch.cuda.is_available() else -1
        try:
            # use text-generation pipeline which supports causal models
            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                device=device,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise LLMClientNotConfigured(f"failed to init HF pipeline: {exc}")

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self.enabled:
            raise LLMClientNotConfigured("HuggingFace LLM fallback is disabled by default.")
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "backend": "huggingface",
            "model": self.model,
            "max_new_tokens": max_new_tokens,
            "user_prompt": prompt,
            "status": "started",
        }
        try:
            out = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=True)
            if not out:
                trace.update({"status": "empty", "latency_ms": (time.perf_counter() - started) * 1000})
                self.call_traces.append(trace)
                return ""
            text = out[0].get("generated_text") or ""
            trace.update({"status": "ok", "latency_ms": (time.perf_counter() - started) * 1000, "raw_response": text})
            self.call_traces.append(trace)
            return text
        except Exception as exc:
            trace.update({"status": "error", "latency_ms": (time.perf_counter() - started) * 1000, "error": str(exc)})
            self.call_traces.append(trace)
            return ""

    def _json_generate(self, prompt: str, max_new_tokens: int = 256) -> dict[str, Any] | None:
        text = self._generate(prompt, max_new_tokens=max_new_tokens)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def orchestrate(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        prompt = self.prompts.get("main_orchestrator", self.prompts.get("default", "Answer concisely."))
        text = self._generate(prompt + "\n\n" + json.dumps(payload, sort_keys=True), max_new_tokens=320)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def plan_physics_action(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        prompt = self.prompts.get("physics_agent_planner", self.prompts.get("default", "Answer concisely."))
        text = self._generate(prompt + "\n\n" + json.dumps(payload, sort_keys=True), max_new_tokens=220)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def plan_logic_action(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        prompt = self.prompts.get("logic_agent_planner", self.prompts.get("default", "Answer concisely."))
        text = self._generate(prompt + "\n\n" + json.dumps(payload, sort_keys=True), max_new_tokens=220)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def suggest_physics(self, question: str) -> dict[str, Any] | None:
        prompt = (
            f"Question: {question}\n"
            "Return JSON with target_quantity, optional formula_id, optional expression, variables in SI units, and target_unit. "
            "If you cannot name a known formula_id, return a single evaluable expression for the answer."
        )
        return self._json_generate(prompt, max_new_tokens=256)

    def suggest_logic(self, question: str, premises: list[str]) -> dict[str, Any] | None:
        prompt = "Premises:\n" + "\n".join(premises) + f"\nQuestion: {question}\nReturn JSON with answer, used_premise_ids, and reason_short."
        return self._json_generate(prompt, max_new_tokens=256)

    def rewrite_explanation(self, trace: dict[str, Any]) -> str | None:
        prompt = self.prompts.get("explanation_rewrite", json.dumps(trace, sort_keys=True))
        text = self._generate(prompt, max_new_tokens=180)
        if not text.strip():
            return None
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("explanation", "text", "content"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
        return text.strip()
