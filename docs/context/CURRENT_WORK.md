# Current Work

Last updated: 2026-06-07 Asia/Bangkok

## Active Workstream — Method-Centric Reasoning Architecture (North Star §24)

The system has been refactored from a fixed routed pipeline into a
**method-centric reasoning architecture**: every reasoning unit conforms
to the `app.methods.types.Method` Protocol; a `MethodPlanner` selects
applicable methods, gates results with faithfulness + coverage, and
falls back to the legacy pipeline when no method decides. New
capabilities (LLM cache, runtime method discovery, SymPy equation graph)
are wired in.

Activation: set `URA_USE_METHOD_PLANNER=1`. The legacy pipeline is the
default and is preserved verbatim.

### Reasoning ladder position (per AGENTS.md §24)

```
Level 0  raw LLM
Level 1  LLM + JSON validation
Level 2  LLM translate → symbolic prover                  ✓ pre-existing
Level 3  + faithfulness + coverage gates                   ✓ live
Level 4  + general equation-graph solver (physics)         ✓ live (Phase F5)
Level 5  + meta-reasoning planner                          ✓ live
Level 6  + runtime method discovery (search→register)      ✓ live (persisted)
```

### Methods registered (default library, 7 built-ins)

```
logic.pattern_rewrite_then_fol_z3   logic_retrieval   builtin
logic.legacy_pipeline               logic_symbolic    builtin
physics.qualitative_reasoner        physics_formula   builtin
physics.equation_graph              physics_numeric   builtin
physics.legacy_pipeline             physics_formula   builtin
physics.conceptual_lookup           physics_retrieval builtin
physics.retrieval_grounded          physics_retrieval builtin
```

Plus persistently-registered `physics.discovered.*` methods (one per
runtime-discovered formula) under `models/methods.json`.

### Logic pattern store (5 seed patterns)

```
seed.unless                  X unless Y                -> if not Y, then X.
seed.except_when             X except when Y           -> if not Y, then X.
seed.provided_that           Provided that X, Y        -> if X, then Y.
seed.only_if                 X only if Y               -> if not Y, then not X.
seed.however_does_not_apply  However, X does not apply to Y -> if Y, then not X.
```

### Latest Phase F-* Reports

* `reports/method_centric_architecture_design.md` — Phase A/E design.
* `reports/phase_f1_f2_planner_wired_report.md` — F1 wire + F2
  faithfulness inside FOL+Z3 refine loop.
* `reports/phase_f3_planner_full_eval_report.md` — F3 full 60-case eval
  (planner 45 vs legacy 50, mostly LLM noise).
* `reports/phase_f3_1_f4_light_report.md` — F3.1 LLM cache + F4-light
  expanded pattern seeds.
* `reports/phase_f5_equation_graph_report.md` — F5 equation-graph
  Method (Level 4).

### Currently Running (2026-06-07 ~01:50)

`scripts/eval_planner_vs_legacy.py` — full 60-case re-run with F5 +
F3.1 + F4-light + Phase A-E architecture active. Output goes to
`reports/planner_vs_legacy_summary.md`.

## Pre-Method-Centric Workstream (still relevant)

* **Refined Runtime Architecture**: orchestration_plan options route
  physics extraction, web search, explanation rewrites — preserved.
* **Reproducible Default Mode**: live web search disabled by default
  (`URA_ENABLE_WEB_METHOD_SEARCH=1` to opt in).
* **Hardened Fail Policy**: connection errors fail closed (HTTP 503).
* **Latency & Budget Limits**: `max_model_calls`, `max_agent_steps`,
  `max_search_calls` enforced via CallBudget.
* **3B AWQ vLLM**: still the production-default model when
  `URA_LLM_BASE_URL=http://192.168.1.5:8001/v1`. Switched to `qwen2.5:7b-instruct`
  via Ollama+Cloudflare tunnel for the recent Method-centric work
  because vLLM 0.22.x's FlashInfer kernel crashes on T4.
* **Logic Safety Patches**: conditional negation scoped to consequent;
  `_negates_condition` upgraded to XOR polarity (catches negated
  antecedent for "X unless Y" + "fact is Y"); `_match_all_rule` skips
  premises containing "unless"/"except" so the conditional path
  handles them.

## Next Recommended Action

Per AGENTS.md §24 phase plan:

* **F6** — final model recommendation report consolidating
  {3B local vLLM, 7B remote Ollama} × {planner-on, planner-off} ×
  {with discovery, without discovery}. Awaits the in-flight eval.
* **F4-full** — active LLM-driven discovery for new logic shapes.
  Scaffold lives in `app/methods/logic_patterns.py` and
  `app/methods/discovery.py::discover_logic_method`.
* Address LLM noise floor at the source — investigate Ollama T=0
  determinism settings or move to a smaller, deterministic local model
  for benchmark runs.

## Files To Read Before Touching the Architecture

```
app/methods/__init__.py              package overview
app/methods/types.py                 Method Protocol, MethodResult, MethodTrace
app/methods/library.py               registry + persistence
app/methods/planner.py               meta-reasoning loop
app/methods/runtime.py               request-level wiring
app/methods/faithfulness.py          atom-coverage + round-trip checks
app/methods/coverage.py              silent-drop gate
app/methods/discovery.py             runtime method discovery
app/methods/logic_patterns.py        seed pattern store
app/methods/caching_client.py        per-request LLM cache
AGENTS.md §24                        North Star + phase plan
```
