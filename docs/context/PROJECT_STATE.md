# Project State

Last updated: 2026-06-05

## Current Snapshot

- `/predict` defaults to the local OpenAI-compatible server on `127.0.0.1:8001`.
- **3B AWQ Live Smoke**: `Qwen/Qwen2.5-3B-Instruct-AWQ` now runs under vLLM on RTX 4060 8GB when bound to `0.0.0.0` and called through the WSL IP rather than `127.0.0.1`. Smoke achieved 4/4; the 6-case synthetic slice achieved 5/6 with one logic FOL/Z3 false negative.
- **Refined Orchestration Routing**: Physics extraction, web search, and explanation rewrites are now strictly routed from `orchestration_plan` options.
- **Reproducible Default Mode**: Live web search is disabled by default to ensure reproducible local corpus benchmarks, unless `URA_ENABLE_WEB_METHOD_SEARCH=1` is explicitly set.
- **Hardened Fail Policy**: connection/unreachable errors fail closed (HTTP `503`), while live planner invalid JSON results in a graceful deterministic fallback with a `planner_invalid_json` warning in metadata.
- **Latency & Budget Controls**: Request-level constraints (`max_model_calls`, `max_agent_steps`, `max_search_calls`) are enforced within the agent kernel and search retrieval layers.
- Physics and logic share the bounded agent kernel: LLMs plan/propose/explain, but backend code still verifies the final answer.
- Repeatable live smoke exists at `scripts/real_smoke_tests.py`.
- **Latest Solver Pass**: Physics parser now handles induced-EMF `L * ΔI / Δt`, AC RMS-current equation-style inputs, resonance current `V / R`, simple parallel-lamp current, dielectric point-charge scaling, solenoid turn density, magnetic-field energy, least-count / absolute-relative error, line-charge electric field, and a series `R1` from `V^2 / P - R2` shortcut. Logic policy routing was tightened so non-policy VR-style prompts are no longer misclassified by the academic-policy heuristic, and missing-condition policy cases for "meet all requirements" now return `no` instead of `unknown`. The logic LLM rescue path now allows validated binary overrides but blocks MCQ-style overrides unless deterministic verification agrees. The live llama-server launch script now disables prompt cache by default (`--cache-ram 0`) to avoid KV-cache exhaustion during longer eval batches. On the latest small live logic sample after these changes, logic moved to `0.40` with remaining errors dominated by `other`, `unknown`, and a smaller polarity-flip bucket.
- **Logic Safety Patch**: Conditional negation parsing now scopes negation to the consequent claim, so antecedent-only negation no longer flips the rule. MCQ `cannot_prove` cases may still try LLM rescue, but accepted MCQ answers remain backend-verified. The experimental FOL/Z3 path no longer uses Python `eval()` on LLM-generated FOL; it uses a restricted parser for the small supported FOL subset and remains disabled unless explicitly requested via sidecar config.
- **Physics Root-Cause Recovery Patch**: Recent physics work moved several failure clusters from local shortcuts toward reusable solver structures. `app/physics/scene_parser.py` now supports target field points and general Coulomb vector summation for labeled triangle/perpendicular-bisector/rectangle-style geometry. `app/physics/circuit_solver.py` introduces a small `SeriesRLCPhasorIR` based on SPICE/MNA-style phasor impedance, with `app/physics/solver.py` using it for labeled `R`/`XL`/`XC` scaled-frequency current and resonance reactance recovery. The solver also gained a backend multi-answer executor for measurement average/mean-absolute-error and capacitor energy+charge prompts, fixed least-count relative error to use the full least count, added mass units, guarded qualitative lookup from hijacking numeric compute prompts, fixed scorer decimal parsing for values like `0.101`, and added deterministic yes/no resonance handling.
- **Physics IR Architecture Started**: Added shared `app/physics/ir.py`, `app/physics/dimensions.py`, `app/physics/equation_graph.py`, and adapter modules under `app/physics/adapters/`. `solver.py` now uses `default_adapters()` from `adapters/registry.py`. Measurement multi-answer logic runs through `MeasurementAdapter`; RLC phasor cases run through `CircuitAdapter`; Coulomb coordinate/vector scenes run through `ElectrostaticsVectorAdapter`; non-EM mechanics examples run through `MechanicsAdapter`. Mechanics coverage now includes speed, Newton's second law, acceleration, torque, kinetic energy, momentum, and simple constant-acceleration final velocity/displacement. This is the intended path for future mechanics/thermal/waves/fluids extensions rather than adding broad handlers in `solver.py`.
- **Adapter Traceability**: Adapter solutions now record selected adapter, adapter score, candidate adapter scores, and adapter-specific solution trace in `PhysicsSolution.search_trace[0]["physics_adapter"]`. This preserves auditability while moving solver logic behind adapters.

## Recent Verified Checks

- `python -m compileall -q app tests scripts/real_smoke_tests.py`
- `pytest -q tests/test_explanation_worker.py tests/test_router.py` (all tests passing perfectly offline with `URA_ALLOW_HEURISTIC_FALLBACK=1`)
- `pytest -q tests/test_logic_solver.py tests/test_academic_policy_reasoner.py tests/test_policy_unknown_missing_conditions.py` (all targeted logic regressions passing)
- `python -m py_compile app/logic/solver.py app/physics/parser.py app/physics/formulas.py app/physics/solver.py app/physics/templates.py app/eval/scorers.py`
- `pytest -q tests/test_logic_solver.py -k 'llm_rescue_can_override_unknown or llm_rescue_does_not_override_unknown_mcq'` (2 passed)
- live llama-server restart via `scripts/start_vllm_server.sh` with `--cache-ram 0`
- Targeted logic safety checks passed for conditional negation, missing MCQ symbolic modules, and restricted FOL/Z3 parsing; no full test suite was run for this patch.
- Physics targeted regression after root-cause patch: `pytest -q tests/test_physics_coulomb_geometry_regressions.py tests/test_eval_scorers.py` -> `42 passed`; `python3 -m py_compile app/physics/circuit_solver.py app/physics/solver.py` passed. Full suite was not rerun in this shell because previous collection lacked WSL deps `fastapi` and `z3`.
- Physics IR/adapters targeted regression: `pytest -q tests/test_physics_ir_architecture.py tests/test_physics_coulomb_geometry_regressions.py tests/test_eval_scorers.py` -> `53 passed`; py-compile passed for updated mechanics adapter, dimensions, and unit converter.
- 3B AWQ smoke/benchmark: `wsl -e bash /mnt/d/URA_challenge/scripts/_tmp_run_smoke_benchmark_3b.sh` -> smoke 4/4, synthetic benchmark 5/6; artifacts in `outputs/predictions/`, summaries in `reports/smoke_test_report.md` and `reports/benchmark_run_summary.md`.

## Current Recommendation

- Maintain the default reproducible mode for benchmarking.
- Keep the live vLLM default and fail-closed connection gating.
- Root-cause the `logic_002` FOL/Z3 false negative before treating the 3B AWQ benchmark as solved.
- Keep this file short. Put older history in reports or git history instead of expanding the handoff context.
- Next physics architecture work should avoid adding more broad `if` handlers in `solver.py`; add shared `PhysicsProblemIR`, dimensions, equation graph, and adapter interfaces, then migrate `scene_parser.py`, `circuit_solver.py`, and measurement programs behind those adapters.
