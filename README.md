# URA EXACT Challenge

Local neuro-symbolic QA system for educational logic and physics questions.
LLM **translates + explains**. Z3 / SymPy **decides**. Backend **validates + assembles** JSON.
Raw model output is never the final authority.

## Current scores (hard_eval_50 via planner-on path)

| Domain  | Score     |
|---------|-----------|
| Physics | 25/25 (100%) |
| Logic   | 25/25 (100%) |
| **Total**   | **50/50 (100%)** |

Random-batch generalization (35-case un-tuned slice): **31/35 (88.6%)**.

## Architecture

```text
/predict  →  MethodPlanner  →  scored Method shortlist
                              ├─ logic.fol_z3        (Z3 FOL prover + self-refine)
                              ├─ logic.patterns      (pre-rewrite + Z3)
                              ├─ logic.bfs           (backward chaining)
                              ├─ physics.equation_graph  (SymPy graph solver, primary)
                              ├─ physics.retrieval_grounded (search-grounded formulas)
                              ├─ physics.qualitative (direction/monotonic reasoning)
                              └─ (legacy_fallback)   (forward chaining / legacy solver)
                              ↓
                       faithfulness gate → coverage gate → self-consistency vote
                              ↓
                       backend assembles + validates JSON  (app/schemas.py)
```

Enable method planner with `URA_USE_METHOD_PLANNER=1` (required for production path).

## Requirements

- WSL Ubuntu (mirrored-mode networking — see below)
- Python 3.10+, `.venv` in repo root
- Single GPU ≥ 8 GB VRAM (RTX 4060 tested)
- Open-weight LLM ≤ 8B served via OpenAI-compatible API on port 8001

## Setup

```bash
# From WSL
cd /mnt/d/URA_challenge
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Networking — CRITICAL (WSL mirrored mode)

WSL mirrored mode: loopback (`127.0.0.1`) does **not** reach servers bound on `0.0.0.0`.
Use the WSL LAN IP instead:

```bash
WSL_IP=$(hostname -I | awk '{print $1}')   # e.g. 192.168.1.5
```

Verify alignment:

```bash
hostname -I
ss -tlnp | grep :800
curl -s http://$(hostname -I | awk '{print $1}'):8001/v1/models | head -c 200
grep URA_LLM_BASE_URL .env
```

## Start LLM Server

**Option A — vLLM (Qwen2.5-3B AWQ, local GPU):**

```bash
source .venv/bin/activate
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct-AWQ --host 0.0.0.0 --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 2048 --quantization awq
```

Note: `VLLM_USE_V1=0` required — flashinfer ABI mismatch with torch 2.6 crashes V1 engine.

**Option B — llama-cpp-python (GGUF, local GPU):**

```bash
python3 -m llama_cpp.server \
  --model models/<file>.gguf --host 0.0.0.0 --port 8001 \
  --n_gpu_layers -1 --n_ctx 4096 --chat_format gemma
```

**Option C — Cloudflare Quick Tunnel → Ollama on Colab T4 (recommended for 7B quality):**

1. Open `scripts/colab_remote_llm.ipynb` in Google Colab (T4 runtime).
2. Run all cells — Ollama starts serving `qwen2.5:7b-instruct`, Cloudflare tunnel prints URL.
3. Copy the tunnel URL (e.g. `https://xxxx.trycloudflare.com`) into `.env`:

```bash
URA_LLM_BASE_URL=https://xxxx.trycloudflare.com/v1
URA_LLM_MODEL=qwen2.5:7b-instruct
```

Tunnel hostname rotates on every Colab restart — update `.env` each time.

Only ONE LLM server may hold the GPU at a time. Stop the previous before starting a new one:

```bash
pkill -9 -f vllm.entrypoints   # kill vLLM
# or kill the llama_cpp process
nvidia-smi                       # confirm VRAM freed
```

## Start API Server

```bash
WSL_IP=$(hostname -I | awk '{print $1}')
source .venv/bin/activate
URA_LLM_MODEL='qwen2.5:7b-instruct' \
URA_LLM_BASE_URL="http://$WSL_IP:8001/v1" \
URA_USE_METHOD_PLANNER=1 \
URA_ENABLE_QUALITATIVE_PARSER=1 \
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For Cloudflare tunnel, set `URA_LLM_BASE_URL` to the tunnel URL directly (no WSL_IP needed):

```bash
URA_LLM_MODEL='qwen2.5:7b-instruct' \
URA_LLM_BASE_URL='https://xxxx.trycloudflare.com/v1' \
URA_USE_METHOD_PLANNER=1 \
URA_ENABLE_QUALITATIVE_PARSER=1 \
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

After any code change in `app/`, restart the API server (no `--reload`).

Verify the server is up:

```bash
curl -s http://localhost:8000/health
```

A stale non-project server returns 404 on `/predict` or shows `provider: mock` in health. Kill and restart.

## Run Evals

**Full 50-case benchmark (hard_eval_50):**

```bash
source .venv/bin/activate
python scripts/deep_test_planner.py
# Results in: reports/deep_test_planner_summary.md + reports/deep_test_planner_cases.jsonl
```

**Random generalization batch (35 fresh cases):**

```bash
python scripts/random_batch_eval.py --limit 35
# Results in: reports/random_batch_results.jsonl
```

**Hard eval v2 (focused regression):**

```bash
python scripts/hard_eval_v2.py
```

**Planner vs legacy comparison:**

```bash
python scripts/eval_planner_vs_legacy.py
# Results in: reports/eval_planner_vs_legacy.json
```

All evals run against `data/Logic_Based_Educational_Queries.corrected.json` (corrected labels).
Never use `data/Logic_Based_Educational_Queries.json` (original — 53% mislabeled).

## Smoke Test

```bash
source .venv/bin/activate
export PYTHONPATH=.
python scripts/real_smoke_tests.py   # HTTP 200 on logic + physics via /predict
```

Health and demo endpoints:

```
http://localhost:8000/health
http://localhost:8000/demo
```

## Unit Tests

```bash
source .venv/bin/activate
pytest -q tests/test_logic_solver.py tests/test_dsl_compiler.py tests/test_fol_z3_pipeline.py
```

Known pre-existing failures (do not chase):
- `test_baseline_accuracy_floors.py::test_logic_first_50_baseline_accuracy_at_least_55_percent` — legacy path 50% < 55% floor
- `test_negation_scope.py::test_negated_consequent_entails_no[generic_class_consequent]` — returns `unknown` on generic class, pre-existing
- `test_physics_coulomb_geometry_regressions.py::test_series_parallel_resistors_network_variation_1` — pre-existing from session 1

## Architecture Rules

- LLM = translator + explainer only. Z3 / SymPy = decider. Backend = validator + JSON assembler.
- New reasoning shapes → register a `Method` in `app/methods/`. Never add `if`-branches to `app/router.py` or `app/logic/solver.py`.
- No per-question text overrides or substring-match heuristics (AGENTS.md §20.1).
- No closed-source / commercial LLM APIs. Open-weight ≤ 8B only.
- `unknown` is correct when solver cannot derive an answer. Never guess.
- All Python via `.venv`. WSL only. Tunnel URL in `.env`, not `127.0.0.1`.

## Key Files

| Path | Role |
|------|------|
| `app/methods/planner.py` | MethodPlanner — shortlist, walk, gate, discover |
| `app/methods/impl/physics_equation_graph.py` | SymPy graph solver (primary physics path) |
| `app/methods/impl/logic_fol_z3.py` | FOL + Z3 with self-refine loop |
| `app/methods/discovery.py` | Level-6 runtime discovery (logic + physics) |
| `app/logic/solver.py` | Forward chaining + contradiction guard |
| `app/logic/fol_z3_pipeline.py` | FOL → Z3 pipeline |
| `app/physics/retrieval_grounded_method.py` | Search-grounded formula extraction |
| `app/physics/unit_converter.py` | SI unit normalization |
| `models/methods.json` | Persistent discovered physics methods |
| `models/logic_patterns.json` | Persistent discovered logic rewrite patterns |
| `data/Logic_Based_Educational_Queries.corrected.json` | Corrected eval dataset (use this) |
| `reports/session_method_centric_full_summary.md` | Master session report — append here |

## Context For Agents

Read in this order before any code change:

```
AGENTS.md §24 (North Star), §25 (Resume Guide)
CLAUDE.md §0 (read order), §5 (session history), §9 (resume guide)
reports/session_method_centric_full_summary.md
reports/deep_test_planner_summary.md
docs/context/PROJECT_STATE.md
docs/context/CURRENT_WORK.md
docs/context/KNOWN_FAILURES.md
```

Do not create new fine-grained session reports. Append to `reports/session_method_centric_full_summary.md`.
