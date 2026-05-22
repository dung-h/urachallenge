# Project State

Last updated: 2026-05-22

## Current Status

- Submission readiness is `11/11` per `reports/submission_readiness_report.md`.
- The project is ready for submission packaging under the current production baseline (74% hybrid system).
- Official EXACT 2026 datasets have been ingested as raw files under `datasets/exact/official/` and normalized into derived JSONL under `datasets/exact/processed/`.
- Curated/corrected internal v1 datasets now exist under `datasets/exact/curated/v1/`; raw official files remain unchanged for provenance.
- Curated baseline has been run; see `reports/curated_baseline_report.md` and `reports/curated_baseline_failure_taxonomy.md`.
- Current deterministic official baseline has been run; see `reports/official_baseline_report.md` and `reports/official_baseline_failure_taxonomy.md`.
- Current recommended official-data direction is no-fine-tune-first neuro-symbolic benchmarking: deterministic solvers/provers as authority, LLMs as parser/explainer/fallback workers only.
- Production answer authority remains deterministic solver/rule code plus Pydantic validation.
- Submission-quality evaluator is now available via `scripts/run_quality_eval.py`; latest no-fallback run on the fixed eval JSONLs reports logic `0.920`, hard academic policy `0.933`, physics `0.965`, explanation consistency `1.000`, hallucinated premise rate `0.000`, and average latency under `10 ms`.
- Final JSON must be assembled and validated by backend code, not emitted directly from model output.
- Live LLM fallback, LLM explanation rewrite, Z3, and Datalog sidecars remain disabled by production default.
- The explanation path was tightened into a structured solver-trace contract: solver output now feeds an optional explanation worker, and backend validation rejects rewrites that add foreign formulas or premise IDs.
- Request-level opt-in fallback is now supported in the router: the solver stays first authority, but LLM/search workers can be enabled per request when a problem does not reduce cleanly to deterministic premises or formulas.
- Phase 29 symbolic direction gates restored zero wrong-when-accepted on the 80-item symbolic sidecar eval, but the path remains experiment-only.
- MCQ option-to-FOL model shootout tested 15 local GGUF candidates on 20 curated MCQ rows; no model achieved both non-zero coverage and zero wrong-when-accepted, so no model is recommended for symbolic promotion yet.
- Direct Transformers Llama 3.2 1B/3B testing was added after GGUF-only testing; both Llama candidates also failed the zero-wrong accepted gate.
- **Phase 5 LLM reasoning exploration (2026-05-20, ABANDONED):** Fine-tuned Phi-3.5 (25%), GLM-5 reasoning (0%), Llama 3.3 70B reference (40%) all failed to beat 74% baseline. Qwen 3 32B achieved 81.2% but violates <8B constraint. Attempted local finetune of Qwen3.5-4B and DeepSeek-R1-Qwen-7B (8-bit + LoRA) but setup too complex: peft/accelerate version mismatch, encoding errors, package resolution timeouts. Finetune expected +5-10pp improvement not worth 6+ hours setup. Baseline 74% kept. See `reports/phase5_abandoned_finetune_final.md`.

## Latest Verified Checks

- `python scripts/run_quality_eval.py`: logic `0.920`, physics `0.965`, explanation consistency `1.000`
- `pytest -s -q`: `98 passed` for the current lightweight workspace test set
- `pytest -q`: `164 passed`
- `python scripts/check_readiness.py`: OK
- `python -m pip check`: OK
- Config audit found no `enable_llm_fallback: true` or `enable_z3_sidecar: true` in `configs/*.yaml`.

## Production Defaults

Primary production files:

- `configs/pipeline.yaml`
- `configs/pipeline.production.yaml`
- `configs/production_baseline.yaml`
- `configs/frozen_production_baseline.yaml`

Important production boundaries:

- `enable_llm_fallback: false`
- `enable_llm_explanation: false`
- `deterministic_physics_authority: true`
- `validate_premise_ids: true`
- `validate_final_json: true`
- `enable_z3_sidecar: false`
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
