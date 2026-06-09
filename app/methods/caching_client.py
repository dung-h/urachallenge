"""Per-request memoizing wrapper around an LLM client.

Phase F3.1: a single QA request can route a question through MULTIPLE
methods that each end up calling the LLM. With the legacy pipeline wrap
in particular, the planner may invoke `solve_fol_z3` once via the pattern
shortcut and a second time via the legacy-pipeline method. The remote
LLM (Ollama on Colab T4 in our setup) is non-deterministic at T=0 enough
that two identical prompts can return different answers — and that noise
flips ~3 of 60 cases between consecutive runs.

This wrapper caches `(role, system, user, max_tokens, response_format)`
within the lifetime of ONE request. A repeated call with the same
arguments returns the cached `LLMResult` instead of re-rolling the dice.
The wrapper is transparent to ``call_traces``, ``enabled``, ``model``,
``base_url`` introspection so existing audit/runtime code is unchanged.

Cache scope is intentionally per-request: a long-running process should
NOT cache across questions because that would smear context between
unrelated requests.
"""

from __future__ import annotations

import hashlib
from typing import Any


class RequestScopedCachingClient:
    """Cache LLM `chat` calls within one request.

    Usage:

        cached = RequestScopedCachingClient(real_client)
        # Pass `cached` everywhere a client would go for the duration of
        # this request. Drop it at the end so the cache is GC'd.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: dict[str, Any] = {}
        self._hit_count = 0
        self._miss_count = 0

    # --- introspection passthrough -----------------------------------------

    @property
    def enabled(self) -> bool:
        return getattr(self._inner, "enabled", False)

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    @property
    def base_url(self) -> str:
        return getattr(self._inner, "base_url", "")

    @property
    def call_traces(self) -> list:
        return getattr(self._inner, "call_traces", [])

    @property
    def api_key(self) -> str:
        return getattr(self._inner, "api_key", "")

    @property
    def hits(self) -> int:
        return self._hit_count

    @property
    def misses(self) -> int:
        return self._miss_count

    # --- cached chat -------------------------------------------------------

    def _key(self, role: str, user: str, max_tokens: int, response_format: bool) -> str:
        # Hash on role + user + max_tokens + response_format. The system
        # prompt is determined by ``role`` via ``self._inner.prompts`` so it
        # is structurally implied by ``role``.
        blob = f"{role}|{max_tokens}|{int(response_format)}|{user}"
        return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()

    def chat(
        self,
        role: str,
        user: str,
        max_tokens: int = 256,
        response_format: bool = False,
    ) -> Any:
        key = self._key(role, user, max_tokens, response_format)
        if key in self._cache:
            self._hit_count += 1
            return self._cache[key]
        self._miss_count += 1
        result = self._inner.chat(
            role, user, max_tokens=max_tokens, response_format=response_format
        )
        self._cache[key] = result
        return result

    # --- generic generate / complete passthrough ---------------------------

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Any:
        # Same caching rule as chat.
        key = hashlib.sha1(
            f"generate|{max_tokens}|{temperature}|{prompt}".encode("utf-8", errors="replace")
        ).hexdigest()
        if key in self._cache:
            self._hit_count += 1
            return self._cache[key]
        self._miss_count += 1
        if hasattr(self._inner, "generate"):
            result = self._inner.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
        elif hasattr(self._inner, "complete"):
            result = self._inner.complete(prompt)
        else:
            result = self._inner(prompt)
        self._cache[key] = result
        return result

    def __call__(self, prompt: str) -> Any:
        return self.generate(prompt)
