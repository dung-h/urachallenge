# URA EXACT Challenge

Local neuro-symbolic QA system for educational logic and physics questions.

The production path is:

```text
/predict request
-> local OpenAI-compatible LLM server on 127.0.0.1:8001
-> LLM planner/proposal/explanation worker
-> deterministic backend verification
-> Pydantic-validated final JSON
```

Raw model output is never the final authority.

## Requirements

- WSL Ubuntu
- Python 3.10+
- `.venv` in the repo
- A local open-weight model served through an OpenAI-compatible API

The verified local model is `Qwen/Qwen2.5-7B-Instruct` via `llama-server` GGUF:

```text
models/gguf/qwen25_7b_instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

## Setup

Run from WSL:

```bash
cd /mnt/d/ura_challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Start The LLM Server

```bash
bash scripts/start_vllm_server.sh
```

The script starts the first available real backend:

- Python `vLLM`, if installed
- `third_party/llama.cpp/llama-server` with the Qwen2.5 GGUF model

Production does not silently start a mock model. For local development only:

```bash
URA_ALLOW_MOCK_LLM_SERVER=1 bash scripts/start_vllm_server.sh
```

## Start The API

In a second WSL shell:

```bash
cd /mnt/d/ura_challenge
bash scripts/start_api.sh
```

Default production environment:

```bash
URA_LLM_BASE_URL=http://127.0.0.1:8001/v1
URA_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
URA_ALLOW_HEURISTIC_FALLBACK=0
```

If the LLM server is offline, `/predict` returns HTTP `503` by default. Heuristic fallback is opt-in only:

```bash
URA_ALLOW_HEURISTIC_FALLBACK=1 bash scripts/start_api.sh
```

## Smoke Test

```bash
export PYTHONPATH=.
python scripts/real_smoke_tests.py
```

Expected result: one logic case and one physics case return HTTP `200` through `/predict`.

Useful URLs:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/demo
```

## Architecture Rules

- LLMs are workers for planning, proposals, search intent, and explanation rewrite.
- Backend code validates the answer, units, premise IDs, trace, and final JSON.
- Logic and physics share a bounded agent kernel.
- Physics formula/method search is allowed, but accepted answers must pass backend verification.
- Closed-source hosted LLM APIs are not part of the submission path.
- Ollama, HuggingFace direct loading, and OpenCode CLI are not runtime adapters in the production path.

## Development Checks

```bash
source .venv/bin/activate
python -m compileall -q app tests scripts/real_smoke_tests.py
pytest -s -q tests/test_router.py tests/test_explanation_worker.py
```

## Context For Agents

Before changing runtime behavior, read:

```text
docs/context/AGENT_ONBOARDING.md
docs/context/PROJECT_STATE.md
docs/context/CURRENT_WORK.md
docs/context/KNOWN_FAILURES.md
```

Keep `PROJECT_STATE.md` and `CURRENT_WORK.md` short. They are handoff snapshots, not changelogs.
