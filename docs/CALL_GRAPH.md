<!--
NOTE: This document is a repo-maintained snapshot.
Update it when moving entrypoints, renaming modules, or changing major flows.
-->

# Call Graph (Repo Map)

This doc describes what each module does and how requests flow through the system.

## Entry Points

### `app/main.py`

- Purpose: FastAPI app bootstrap.
- Owns: exception handlers, `/health`, `/` redirect.
- Delegates: mounts `app/router.py`.

### `app/router.py`

- Purpose: HTTP API surface and runtime orchestration.
- Endpoints:
  - `POST /predict` (main)
  - `GET /trace/{request_id}` (trace retrieval)
  - `GET /demo` (UI)
- Core function: `predict_with_metadata(request, config, llm_client, write_trace)`.

Key functions:
- `route_task(request)`
- `predict_with_metadata(...)`
- `trace(request_id)`

## Top-Level Runtime Flow (`/predict`)

1. Input schema: `app/schemas.py` (`QARequest`).
2. Normalize + guardrails: `app/runtime_workflow.py` (`InputNormalizer`) and `app/guardrails.py`.
3. Route and plan:
   - Default: deterministic `route_and_plan(...)` in `app/runtime_workflow.py`.
   - Optional: LLM planner `LLMOrchestrator.plan(...)` in `app/runtime_workflow.py` (gated by `PipelineConfig.use_llm_orchestrator`).
4. Ensure LLM client if allowed:
   - `app/runtime_clients.py` -> `app/llm_client.py` (`OpenAICompatibleLLMClient`, OpenAI-compatible local endpoint, typically vLLM).
   - Wrap with budget gate: `BudgetGatedClient` in `app/runtime_workflow.py`.
5. Execute task solver:
   - Physics: `app/physics/solver.py::solve`
   - Logic: `app/logic/solver.py::solve`
6. Assemble validated response + metadata:
   - `app/runtime_trace.py::assemble_response` + confidence signals in `app/confidence.py`.
   - Optionally rewrite explanation via `app/runtime_trace.py::maybe_rewrite_explanation` calling `app/explanation_worker.py`.
7. Persist trace (optional): `app/runtime_trace.py::write_trace`.

## Physics Subsystem

### `app/physics/solver.py`

- Purpose: main physics controller.
- Primary path: deterministic parse -> adapter solve -> unit/dimension verification.
- Fallbacks (controlled by orchestration plan + budgets): method-search retrieval + agent loop.

Key functions (high-signal):
- `solve(...)`
- verifier boundary helpers: `_units_equivalent(...)`, `_dimensional_agreement(...)`

Key dependencies:
- Parsing: `app/physics/parser.py` -> `ParsedPhysicsProblem`.
- Adapter architecture:
  - `app/physics/ir.py`, `app/physics/dimensions.py`, `app/physics/equation_graph.py`
  - Adapters: `app/physics/adapters/*` via `app/physics/adapters/registry.py::default_adapters()`
- Search-backed method evidence:
  - `app/physics/method_search.py` (retrieval, proposal extraction, backend verification)
- Agent rescue loop:
  - `app/agent_runtime.py` + `app/agent_kernel.py` + `app/agent_tools.py`
- Explanation templates: `app/physics/templates.py`
- Units: `app/physics/unit_converter.py`

### Adapter Call Order

Defined in `app/physics/adapters/registry.py`:
- `MeasurementAdapter`
- `CircuitAdapter`
- `ElectrostaticsVectorAdapter`
- `MechanicsAdapter`

## Logic Subsystem

### `app/logic/solver.py`

- Purpose: central logic solve entry point.
- Paths:
  - Policy reasoning: `app/logic/policy_reasoner.py`.
  - Deterministic forward chaining / MCQ tooling: internal helper modules (`app/logic/_*.py`).
  - Optional Z3/FOL verification: `app/logic/_fol_bridge.py` and `app/logic/fol_z3_pipeline.py` (guarded by config/allowed domains).
  - Agent rescue loop: `app/logic/agent_runtime.py` + `app/logic/agent_tools.py` + shared `app/agent_kernel.py`.

Key functions:
- `solve(...)`
- deterministic fast path: `solve_forward_chaining(...)`

### `app/logic/premise_selector.py`

- Purpose: normalize premises, select relevant premise IDs, detect hallucinated premise references.

### `app/logic/proof_trace.py`

- Purpose: build and validate proof steps for transparency + verifier evidence.

## Explanation Worker

### `app/explanation_worker.py`

- Purpose: build trace-grounded explanations deterministically and validate LLM rewrites.
- Used by: `app/runtime_trace.py` and (for verified paths) `app/logic/fol_z3_pipeline.py`.

## LLM + Budgets

### `app/llm_client.py`

- Purpose: OpenAI-compatible client wrapper (vLLM/llama-server), JSON validity tracking, prompt templates (`configs/prompts.yaml`).

### `app/runtime_workflow.py`

- Purpose: normalization, deterministic routing/planning, LLM orchestrator hook, call budgeting.

## Config

- `configs/pipeline.yaml`: pipeline defaults (Z3 sidecar, model IDs, etc.)
- `app/pipeline_config.py`: YAML loader + env overrides.

## Module Import Graph (Mermaid)

Full module-level import graph snapshot:
- `docs/graphs/module_import_graph.mmd`

## Suspected Cleanup Candidates (Needs Confirmation)

This section is intentionally conservative: it lists areas to review, not files to delete.

- `scripts/analysis/*`: looks research/analysis oriented. Likely safe to exclude from runtime deployment, but keep if you still use them for eval.
- `scripts/install_llama_cpp_cuda.py`: only needed if llama.cpp backend is still used. If vLLM is the only backend going forward, we can consider removing or moving it to an archived/dev-only area.
- `scripts/mock_openai_server.py`: useful for offline tests; keep unless it is unused and redundant with other mocks.
