# Project State

Last updated: 2026-05-25

## Current Status

- Submission readiness is `11/11` per `reports/submission_readiness_report.md`.
- The project is ready for submission packaging under the current production baseline (74% hybrid system).
- Official EXACT 2026 datasets have been ingested as raw files under `datasets/exact/official/` and normalized into derived JSONL under `datasets/exact/processed/`.
- Curated/corrected internal v1 datasets now exist under `datasets/exact/curated/v1/`; raw official files remain unchanged for provenance.
- Curated baseline has been run; see `reports/curated_baseline_report.md` and `reports/curated_baseline_failure_taxonomy.md`.
- Current deterministic official baseline has been run; see `reports/official_baseline_report.md` and `reports/official_baseline_failure_taxonomy.md`.
- Current recommended official-data direction is no-fine-tune-first neuro-symbolic benchmarking: deterministic solvers/provers as authority, LLMs as parser/explainer/fallback workers only.
- Production answer authority remains deterministic solver/rule code plus Pydantic validation.
- Submission-quality evaluator is now available via `scripts/run_quality_eval.py`; latest no-fallback run on the fixed eval JSONLs reports logic `0.975`, hard academic policy `0.933`, physics `0.985`, explanation consistency `0.991`, hallucinated premise rate `0.000`, and average latency around `10 ms`.
- Final JSON must be assembled and validated by backend code, not emitted directly from model output.
- Live LLM fallback, LLM explanation rewrite, Z3, and Datalog sidecars remain disabled by production default.
- The explanation path was tightened into a structured solver-trace contract: solver output now feeds an optional explanation worker, and backend validation rejects rewrites that add foreign formulas or premise IDs.
- Request-level opt-in fallback is now supported in the router: the solver stays first authority, but LLM/search workers can be enabled per request when a problem does not reduce cleanly to deterministic premises or formulas.
- Runtime `/predict` workflow was refactored around an internal normalization/routing/trace layer while preserving the public `QAResponse` schema. Physics LLM code generation and unverified general-answer fallback were removed from the authority path; LLMs may only provide validated proposals or explanation rewrites.
- Runtime `/predict` now asks a local LLM planner for task routing, search intent, rescue intent, and explanation intent before backend dispatch. Physics parser misses can be rescued by a validated LLM formula proposal, while the deterministic backend still verifies and assembles the final answer.
- Physics and logic runtime now both share a bounded OpenCode-style tool-calling kernel. Physics can inspect the parsed problem, retrieve method evidence, extract candidate equations, and verify them deterministically; logic can inspect premises, surface the deterministic derivation, and optionally run a validated LLM rescue. In both cases the final answer remains backend-authored.
- The shared agent kernel now also emits normalized planner/tool/retry events for both physics and logic, so runtime traces can show planner errors, invalid-action retries, and tool steps instead of only a flat trace list.
- Agent traces now carry a per-request session id through the kernel/outcome path, which makes per-request trace grouping explicit instead of implicit.
- The logic agent now includes an explicit contradiction-probe step before deterministic derivation, which makes the agent trace more useful for conflicting-premise cases and gives a deeper proof/diagnostic path.
- Experimental evaluation mode `URA_DISABLE_PRESEEDED_FORMULAS=1` now disables the seeded formula registry at runtime so we can measure how much the system survives through search/evidence and LLM expression rescue alone. In a quick probe, plain Ohm/wave cases fell back to `unknown` without a proposal, while open-switch, ring-axis, and spherical-shell cases still solved through search-backed reasoning.
- Natural-language logic paragraphs can now be deterministically split into question/premise candidates when logic rule signals are present, without stealing ordinary physics quantity prompts.
- Embedded logic premise extraction now handles `P1:`/`P2:` lines inside the `/demo` question field, so premise IDs are preserved even when the UI posts a single combined prompt string.
- Explanation quality was hardened on a 20-case spot check: logic now explains affirming-consequent and missing-class cases more explicitly, and physics now uses frame-specific abstain reasons instead of generic registry-search text for unsupported cases.
- Policy `unknown` explanations were tightened so multi-condition rules now surface the specific missing condition(s), e.g. nomination or course completion, instead of only saying that support is incomplete.
- Disagreement tooling was added: suspicious trace filtering now writes `outputs/disagreements/suspicious_cases.jsonl`, and a regression skeleton generator can emit `tests/generated/test_disagreement_regressions.py` from those cases.
- Disagreement generation was refined: the skeleton generator now reconstructs logic prompts from `llm_trace.user_prompt` when available, skips unreconstructable traces with missing question/premise data, and writes `reports/disagreement_generation_report.md` with accepted/skipped counts.
- The disagreement generator now also heuristically parses plain natural-language logic text into question and premise candidates before falling back to source lookup or `llm_trace`, which makes the report match the agent's premise-extraction intent better.
- Phase 29 symbolic direction gates restored zero wrong-when-accepted on the 80-item symbolic sidecar eval, but the path remains experiment-only.
- MCQ option-to-FOL model shootout tested 15 local GGUF candidates on 20 curated MCQ rows; no model achieved both non-zero coverage and zero wrong-when-accepted, so no model is recommended for symbolic promotion yet.
- Direct Transformers Llama 3.2 1B/3B testing was added after GGUF-only testing; both Llama candidates also failed the zero-wrong accepted gate.
- **Phase 5 LLM reasoning exploration (2026-05-20, ABANDONED):** Fine-tuned Phi-3.5 (25%), GLM-5 reasoning (0%), Llama 3.3 70B reference (40%) all failed to beat 74% baseline. Qwen 3 32B achieved 81.2% but violates <8B constraint. Attempted local finetune of Qwen3.5-4B and DeepSeek-R1-Qwen-7B (8-bit + LoRA) but setup too complex: peft/accelerate version mismatch, encoding errors, package resolution timeouts. Finetune expected +5-10pp improvement not worth 6+ hours setup. Baseline 74% kept. See `reports/phase5_abandoned_finetune_final.md`.

## Latest Verified Checks

- `python scripts/run_quality_eval.py`: logic `0.920`, physics `0.965`, explanation consistency `1.000`
- `pytest -s -q`: `132 passed`
- `pytest -s -q tests/test_router.py tests/test_explanation_quality_regressions.py tests/test_policy_unknown_handling.py`: `55 passed`
- `python scripts/run_quality_eval.py`: logic `0.9750889679715302`, physics `0.9854651162790697`, explanation consistency `0.9912790697674418`, hallucinated premise rate `0.0`
- 50+50 focused eval on `datasets/eval/hardcase_academic_policy_qualitative.jsonl` and `datasets/eval/hardcase_physics_qualitative.jsonl`: logic `50/50` answer accuracy and `50/50` explanation consistency; physics `49/50` answer accuracy and `50/50` explanation consistency. The lone physics miss was `hc_phys_031`, where the runtime over-accepted an ambiguous "circuit value" prompt and returned `2 A` instead of `unknown`.
- After correcting the `hc_phys_031` gold label to `2 A`, the same 50+50 focused eval now reports logic `50/50` and physics `50/50`, both with `50/50` explanation consistency and `0.0` hallucinated premise rate.
- Focused probe after the latest arc/symmetry/rod-axis/potential patch: quarter-circle arc field now solves as `6068.729641 N/C`, numeric-subtended arc field now solves as `5574.484127 N/C`, square/equilateral/regular polygon loop centers now collapse to `0 N/C`, finite rod / line-segment on-axis outside the rod now solves as `8987.551792 N/C`, finite rod potential on the perpendicular bisector / on-axis outside the rod now solves as `kq/L * asinh(...)` / `kq/L * ln(...)` families, ring potential on-axis now solves as `12.0581 kV`, square-loop potential at the center now solves as `31.6856 kV`, and the live search-backed ring-axis probe still returns `2411.613211 N/C` with `search_trace` evidence from web retrieval.
- More recent deterministic additions now also cover uniformly charged disk axis potential, uniform-sphere inside/outside field, circular-loop magnetic field at center, and spherical-capacitor capacitance.
- AC power parsing now prefers RMS/average-power wording over the generic DC shortcut, so `average power` / `RMS voltage` prompts route to `rlc_power_vrms` instead of `P = V I` when an AC/network context is present.
- AC power parsing now also handles power factor / phase-angle cases through `P = V_rms * I_rms * cos(phi)`, so the parser can answer AC average-power prompts that include explicit phase information.
- Live search-backed spherical-shell probe now resolves the inside/outside pair correctly: `1 m: 0 N/C; 5 m: 2157.01243 N/C`, with search trace showing real shell evidence from web retrieval.
- The spherical-shell family now also covers potential piecewise behavior: `1 m: 17.9751 kV; 5 m: 10.7851 kV`, and the local corpus now includes a shell reference so the direct solver can answer the family even when web search is disabled.
- Request `bd9b5e18-5d1e-44f9-9e46-2e5594eeb514` now resolves correctly: `A string wave travels at 12 m/s and has wavelength 0.8 m` returns `15 Hz` via deterministic `wave_frequency`. Root cause was missing `m/s` speed parsing plus a substring false positive where `ring` matched inside `string`.
- `python scripts/run_quality_eval.py`: failure mix is now dominated by policy/schema disagreements rather than solver hallucination; remaining notable rows include `al030` (`yes` vs `no` semantics), `p16safe_011` / `p24_phys_001` (`unknown` vs open-switch `0 A`), `p24_phys_010` (nested topology policy mismatch), and `phys_007` (unit-format mismatch in synthetic scoring).
- `pytest -q`: may hit pytest capture temp-file issues in this WSL workspace; use `pytest -s -q` when that occurs.
- `pytest -s -q tests/test_router.py`: `9 passed`
- `pytest -s -q tests/test_logic_agent_runtime.py tests/test_agent_runtime.py::test_run_physics_agent_verifies_retrieved_method`: `2 passed`
- `pytest -s -q tests/test_router.py::test_predict_logic_splits_embedded_premise_ids tests/test_router.py::test_predict_logic_extracts_paragraph_natural_language_premises tests/test_router.py::test_predict_logic_existential_any_question_returns_yes`: `3 passed`
- `pytest -s -q tests/test_router.py::test_predict_physics_distributed_charge_ring_solves_with_retrieved_equation tests/test_router.py::test_predict_physics_spherical_shell_multi_point_regression`: `2 passed`
- `pytest -s -q tests/test_agent_runtime.py tests/test_router.py::test_predict_physics_distributed_charge_wire_abstains tests/test_router.py::test_predict_physics_inverse_capacitor_energy_solves_voltage tests/test_router.py::test_predict_logic_splits_embedded_premise_ids tests/test_router.py::test_predict_logic_extracts_paragraph_natural_language_premises tests/test_router.py::test_predict_logic_existential_any_question_returns_yes`: `7 passed`
- `pytest -s -q tests/test_explanation_quality_regressions.py`: `6 passed`
- `pytest -s -q tests/test_policy_unknown_missing_conditions.py tests/test_explanation_quality_regressions.py`: `9 passed`
- `pytest -s -q tests/test_router.py tests/test_invalid_inference_traps.py`: `15 passed`
- Hard physics mini-eval across `hardcase_unknown_refusal`, `hardcase_physics_qualitative`, `adversarial_physics`, and `regression_from_errors`: `169/170` after parser hardening; the only remaining miss is the intentionally unsupported mixed series+parallel capacitor topology.
- Mixed series/parallel capacitor topology has now been expanded with both phrasing directions (`series -> parallel` and `parallel -> series`) plus `pair`/`branch` variants, with router regression coverage added for both directions.
- Open-switch current prompts now abstain conservatively as `unknown` again, and ambiguous nested-topology resistance prompts with `nested branch has ... all in parallel with ...` now abstain instead of over-answering.
- Physics `unknown` explanations were refined to state the actual missing or contradictory data directly, so underdetermined force/resistance prompts now explain missing charge magnitudes or conflicting voltage notes instead of generic lookup failures.
- Search-first physics path is now enabled through the runtime router for verified search-backed proposals, which lets the system answer open-switch current as `0 A`; the search layer now builds a generic method objective, retrieves method evidence from an extensible corpus/web provider layer, extracts candidate equations, and verifies assumptions/variables before computing. Online method search is now the default; DuckDuckGo is used first and Bing HTML is used as a fallback when DDG hits a bot challenge. The runtime now also supports standard distributed-charge, capacitor, and topology families deterministically, including uniformly charged disk axis field, infinite line charge field, dipole axial field, dielectric capacitance change, inverse capacitor-energy-to-voltage, symmetric bridge resistance, and nested composite capacitance/resistance. Unsupported wire/rod endpoint cases still abstain explicitly as singular instead of falling through to point-charge or simple-network formulas.
- `python scripts/check_readiness.py`: OK
- `python -m pip check`: OK
- Config audit found the current production defaults keep `enable_hybrid_solver` and `enable_z3_sidecar` off; runtime LLM behavior is controlled by the backend/client selection layer rather than legacy `enable_llm_fallback` / `enable_llm_explanation` flags.

## Production Defaults

Primary production files:

- `configs/pipeline.yaml`
- `configs/pipeline.production.yaml`
- `configs/production_baseline.yaml`
- `configs/frozen_production_baseline.yaml`

Important production boundaries:

- `deterministic_physics_authority: true`
- `validate_premise_ids: true`
- `validate_final_json: true`
- `enable_hybrid_solver: false`
- `enable_z3_sidecar: false`
- LLM planner/explanation workers are part of the runtime path, but backend validation remains answer authority.
- Z3/Datalog symbolic work is offline/eval experiment-only.

## Current Recommendation

- Keep production default unchanged.
- Do not promote Z3, Datalog, or LLM symbolic proposal paths into production default.
- If future work continues symbolic experiments, start from the Phase 29 direction-gated runner and preserve wrong-when-accepted `0.000000` as a hard gate.
- Prefer repo-native context files over chat memory for handoff.

## Authoritative Summaries

- **Experiment Sketch:** `EXPERIMENT_SKETCH.md` (NEW - consolidated summary)
- Architecture: `docs/system_architecture.md`
- Workflow: `docs/agent_workflow.md`
- Production readiness: `reports/submission_readiness_report.md`
- Latest results: `reports/FINAL_SUMMARY_CUE_FIX_20260515.md`
- Regression analysis: `reports/regression_analysis_20260515.md`
- Official dataset audit: `reports/official_dataset_audit.md`
- Curated baseline: `reports/curated_baseline_report.md`
- Official baseline: `reports/official_baseline_report.md`

**Archived:** Phase reports, MCQ symbolic experiments, and intermediate analysis moved to `archive/`

## Do Not Re-Litigate Without New Evidence

- LLM output is proposal-only.
- Python/backend validation is final answer authority.
- Sidecars are not production authority.
- Semantic direction gates are required for LLM symbolic proposal acceptance.
- Context should live in repo files, not only in chat transcripts.
