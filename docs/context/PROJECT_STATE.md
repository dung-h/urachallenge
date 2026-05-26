# Project State

Last updated: 2026-05-26

## Current Snapshot

- `/predict` defaults to the local OpenAI-compatible server on `127.0.0.1:8001`.
- Offline server behavior is fail-closed with HTTP `503`, unless `URA_ALLOW_HEURISTIC_FALLBACK=1` is set explicitly.
- Live `llama-server` in WSL has been verified with `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
- Physics and logic now share the bounded agent kernel: LLMs plan/propose/explain, but backend code still verifies the final answer.
- Dead cache/adapters were removed from runtime code. The runtime client path is now narrowed to the OpenAI-compatible/vLLM route.
- Repeatable live smoke exists at `scripts/real_smoke_tests.py`.

## Recent Verified Checks

- `python -m compileall -q app tests scripts/real_smoke_tests.py`
- `pytest -s -q tests/test_router.py tests/test_explanation_worker.py`
- Live `/predict` smoke via `TestClient` returned `200` for one logic sample and one physics sample.

## Current Recommendation

- Keep the live vLLM default and fail-closed gating.
- Do not reintroduce Ollama, HuggingFace, or OpenCode runtime adapters unless there is a specific benchmark reason.
- Keep this file short. Put older history in reports or git history instead of expanding the handoff context.
