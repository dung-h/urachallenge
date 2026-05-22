# Current Work

Last updated: 2026-05-22

## Active Workstream

Official EXACT Dataset Integration Workstream.

Additional active work: solver authority + LLM explanation-worker hardening + request-level fallback opt-in. Production defaults remain unchanged; LLM output remains proposal-only and explanation rewrites are validated against solver trace.

Latest quality-eval work (2026-05-22): added repo-local submission-quality evaluator and tightened deterministic policy/physics guards. Current quality run reports logic answer accuracy `0.920`, hard academic policy `0.933`, physics answer accuracy `0.965`, explanation consistency `1.000`, hallucinated premise rate `0.000`, and average no-fallback latency under `10 ms`; see `reports/quality_eval_summary.md`.

## Latest LLM/Unit Finding (2026-05-13)

- Unit tokenizer coverage was measured in `reports/unit_tokenization_coverage.md` with `scripts/analyze_unit_tokenization.py`.
- Gemma GGUF tokenizers are competitive with Qwen on unit symbols: `Ω`/`Ω` are single-token, `ohm` is one token for Gemma versus two for Qwen, and superscript scientific notation is slightly less fragmented for Gemma.
- Qwen tokenizers are better for slash units like `N/C`, `V/m`, and `rad/s`.
- Tokenizer coverage is not task accuracy; Gemma still needs a direct hard-case physics run before judging usefulness.

Status: Official physics and logic zip files were extracted, audited, normalized into derived JSONL files, evaluated with the current deterministic production-style baseline, curated into an internal corrected v1 dataset, and re-evaluated with a curated baseline scorer. Production defaults remain unchanged.
Current implementation note: `/predict` now builds a structured explanation trace from solver output, lets the optional LLM explanation worker rewrite only from that trace, and falls back to the solver explanation when validation fails. Requests can also opt into LLM/search fallback when a task does not reduce cleanly to deterministic premises or formulas.

Latest router/logic fix: embedded natural-language logic blocks now extract unnumbered premise lines such as `All robots are machines.`, `Observation: Device X is safe.`, and `Question: ...`; logic routing now wins over physics keyword hints such as `power` when universal/conditional premise structure is present. The deterministic solver also handles `All X need Y` chains without requiring LLM fallback.

Latest mini-regression hardening: deterministic `/predict` now passes the 19-case no-fallback stress set covering modus tollens, no-overlap chaining, generalized MCQ contrapositive, external-knowledge traps, router keyword traps, Ohm current target, and LC angular-frequency routing.

## Scope

Ingest official EXACT 2026 datasets and prepare safe derived artifacts for module-level benchmarking before any model acquisition or fine-tuning.

Allowed changes:

- `datasets/exact/official/**` raw extracted official files
- `datasets/exact/processed/**` derived JSONL/manifest files
- `scripts/audit_official_exact_datasets.py`
- `scripts/prepare_official_exact_jsonl.py`
- `reports/official_dataset_audit.md`
- `reports/official_dataset_integration_plan.md`
- `reports/dataset_fix_policy.md`
- `reports/curated_dataset_v1_report.md`
- `reports/curated_baseline_report.md`
- `reports/curated_baseline_failure_taxonomy.md`
- `reports/curated_baseline_debug_report.md`
- `reports/curated_baseline_deep_diagnosis.md`
- `reports/curated_baseline_improvement_report.md`
- `reports/official_baseline_report.md`
- `reports/official_baseline_failure_taxonomy.md`
- `docs/context/**` index/state updates

Out of scope:

- production configs
- frozen eval datasets and scorers
- model downloads
- fine-tuning
- prompt changes
- final response schema changes
- enabling live fallback, Z3, or Datalog by default

## Stop Conditions

- Any change would modify frozen eval datasets or scorers while changing solver/model behavior.
- Any change would enable live fallback, Z3, or Datalog by default.
- Any change would treat raw LLM/model output as final JSON authority.
- Any model download or fine-tune would be required before a separate model-acquisition phase.

## Next Recommended Action

Use curated v1 as the main engineering dataset and harden one module at a time without changing production defaults:

**Completed (2026-05-10):**
1. ✅ Physics target detection fixes (field/charge/energy bugs)
2. ✅ Physics formula matching guards (coulomb_force/electric_field_kq_r2/power_p_v2r abstain on complex cases)
3. ✅ Resultant force formulas (collinear/perpendicular/angle) - 55 rows covered
4. ✅ Measurement error formulas (absolute/relative/propagation) - 2 rows covered, needs expansion
5. ✅ Dielectric capacitor transform formulas - 0 rows matched, detection needs refinement
6. ✅ Magnetism unit support (H/T/Wb/area) + basic formulas (inductor/solenoid) - 20 rows covered
7. ✅ Logic MCQ detection broadened (which conclusion/statement/following)

**Result:** Physics accuracy improved from 0.035 to 0.096 (+2.7x, +106 correct rows). Logic accuracy stable at 0.34.

**MCQ symbolic model shootout (2026-05-10):** Tested all 15 local GGUF candidates on 20 curated MCQ rows with the option-to-FOL proposal path. No model passed the promotion gate of non-zero coverage plus `wrong_when_accepted = 0`; see `reports/mcq_symbolic_model_shootout.md`.

**Next priorities:**
1. Refine dielectric transform detection (18 rows identified, 0 matched).
2. Expand measurement error propagation coverage (36 rows identified, 2 matched).
3. Add physics gold-consistency diagnostic overlay for unambiguous deterministic rows.
4. Multi-charge net force geometry (needs vector solver or templates).
5. Parse official logic FOL and measure syntax coverage before adding prover behavior.
6. Add option-to-claim parsing benchmark for logic MCQ rows.
7. Only after module metrics are known, benchmark local LLM parser candidates.
8. Before any larger MCQ symbolic model run, add stricter acceptance gates for translated claims; current model proposals accept wrong answers when coverage is non-zero.
