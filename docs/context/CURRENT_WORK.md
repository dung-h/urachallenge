# Current Work

Last updated: 2026-05-25

## Active Workstream

Official EXACT Dataset Integration Workstream.

Additional active work: solver authority + LLM explanation-worker hardening + runtime orchestration cleanup. Production defaults keep backend validation as answer authority; LLM output remains proposal-only and explanation rewrites are validated against solver trace.
Latest shared-agent-kernel work (2026-05-25): `/predict` now has a bounded tool-calling loop for both physics and logic. Physics can inspect the parsed frame, retrieve method evidence, extract candidate equations, and verify them before a deterministic backend answer is accepted. Logic can inspect premises, expose the deterministic derivation, and run a validated LLM rescue when allowed. If no proposal verifies, the system returns the deterministic unknown with trace rather than treating the LLM as answer authority.
Latest agent-trace hardening (2026-05-25): the shared kernel now records normalized planner/tool/retry events, including planner errors and invalid-action retries, so trace consumers can distinguish failed planning from executed tool steps.
Latest session-trace hardening (2026-05-25): the agent kernel now stamps a per-request session id through event/outcome traces, which makes it easier to group all tool steps from one `/predict` request.
Latest logic-agent depth update (2026-05-25): the logic agent now probes for contradictions before deterministic derivation, so conflicting-premise cases leave a clearer trace and can short-circuit to a reasoned unknown instead of a flat fallback.

Latest quality-eval work (2026-05-24): added repo-local submission-quality evaluator and tightened deterministic policy/physics guards. Current quality run reports logic answer accuracy `0.9750889679715302`, hard academic policy `0.933`, physics answer accuracy `0.9854651162790697`, explanation consistency `0.9912790697674418`, hallucinated premise rate `0.000`, and average no-fallback latency around `10 ms`; see `reports/quality_eval_summary.md`.

## Latest LLM/Unit Finding (2026-05-13)

- Unit tokenizer coverage was measured in `reports/unit_tokenization_coverage.md` with `scripts/analyze_unit_tokenization.py`.
- Gemma GGUF tokenizers are competitive with Qwen on unit symbols: `Ω`/`Ω` are single-token, `ohm` is one token for Gemma versus two for Qwen, and superscript scientific notation is slightly less fragmented for Gemma.
- Qwen tokenizers are better for slash units like `N/C`, `V/m`, and `rad/s`.
- Tokenizer coverage is not task accuracy; Gemma still needs a direct hard-case physics run before judging usefulness.

Status: Official physics and logic zip files were extracted, audited, normalized into derived JSONL files, evaluated with the current deterministic production-style baseline, curated into an internal corrected v1 dataset, and re-evaluated with a curated baseline scorer. Production defaults remain unchanged.
Current implementation note: `/predict` now builds a structured explanation trace from solver output, lets the local LLM explanation worker rewrite only from that trace, and falls back to the solver explanation when validation fails. The local LLM also participates in planning/rescue in the runtime path, but backend validation remains the authority for final answers.
Latest planner-first orchestration (2026-05-24): `/predict` now asks a local LLM planner for task routing plus search/rescue/explanation decisions before backend dispatch. Physics parser unknowns can be rescued by a validated LLM formula proposal, but the deterministic backend still verifies and assembles the final answer.
Latest seeded-formula off-switch probe (2026-05-24): `URA_DISABLE_PRESEEDED_FORMULAS=1` now suppresses the seeded physics formula registry for runtime evaluation. In a quick probe, direct Ohm/wave questions fell back to `unknown` without a proposal, while an LLM expression proposal still solved Ohm's law (`10 V`) and search-backed open-switch/ring/shell cases still solved through validated evidence.

Latest router/logic fix: embedded natural-language logic blocks now extract unnumbered premise lines such as `All robots are machines.`, `Observation: Device X is safe.`, and `Question: ...`; logic routing now wins over physics keyword hints such as `power` when universal/conditional premise structure is present. The deterministic solver also handles `All X need Y` chains without requiring LLM fallback.

Latest `/demo` hardening: the router now also extracts embedded `P1:`/`P2:` lines when the UI posts the entire logic prompt through the `question` field, which keeps premise IDs and produces the more specific solver explanation for the Maya scholarship case.

Latest explanation hardening: a 20-case manual spot check was used to tighten wording and parser coverage. Logic explanations now call out affirming-consequent and missing-class cases more directly, while physics explanations now correctly handle capacitor energy, capacitance from `Q/V`, transformer secondary voltage, and solenoid magnetic field.

Latest policy unknown hardening: multi-condition academic policy rules now surface the missing clause explicitly, such as `nominated` or `completed MA200`, instead of collapsing to a generic insufficient-evidence message.

Latest disagreement tooling: added `scripts/filter_suspicious_cases.py` to score trace artifacts and `scripts/generate_disagreement_test_skeletons.py` to turn suspicious cases into pytest skeletons, with a generated regression file under `tests/generated/test_disagreement_regressions.py`.

Latest disagreement generation refinement: skeleton generation now skips traces that cannot be reconstructed because they have no usable question/premise data, reconstructs logic prompts from `llm_trace.user_prompt` when available, and writes a sidecar report at `reports/disagreement_generation_report.md` that separates accepted versus skipped cases.

Latest natural-language premise parsing refinement: the disagreement generator now also heuristically splits plain natural-language logic text into `question` plus premise candidates, so it can reconstruct prompts that were not explicitly tagged with `P1:`/`P2:` markers before falling back to source dataset or `llm_trace`.

Latest mini-regression hardening: deterministic `/predict` now passes the 19-case no-fallback stress set covering modus tollens, no-overlap chaining, generalized MCQ contrapositive, external-knowledge traps, router keyword traps, Ohm current target, and LC angular-frequency routing.

Latest runtime authority refactor: `/predict` now uses an internal `InputNormalizer` + `TaskRouter` + typed runtime trace builder. Public `QAResponse` is unchanged. Physics LLM code generation and unverified general reasoning fallback were removed from the authority path; request-level fallback can still provide validated formula/premise proposals. Natural-language logic paragraphs are split into premise candidates only when logic rule signals are present, so normal physics prompts are not stolen by the logic route.

Latest hard-physics sweep (2026-05-24): a 170-row mixed physics eval across `hardcase_unknown_refusal`, `hardcase_physics_qualitative`, `adversarial_physics`, and `regression_from_errors` now reaches `169/170` answer-match under unit-aware checking. The only remaining miss is the intentionally unsupported mixed series+parallel capacitor topology; new parser coverage now handles direct capacitance read-off, `F=qE`, counted resistor repetition, and energy/voltage polarity cases.

Latest search-first tweak (2026-05-24): the router now treats verified search-backed physics proposals as first-class results for supported families. The search layer now builds a generic method objective, retrieves method evidence through an extensible corpus/web provider layer, extracts candidate equations, and verifies assumptions/variables before computing. Open-switch and ambiguous nested-topology prompts now abstain conservatively as `unknown`. Online method search is still the default; DuckDuckGo is tried first, then Bing HTML is used as a fallback when DDG returns a bot challenge. A uniformly charged ring axis-field question now solves from retrieved evidence, the transformer turns-ratio case now solves through web-backed verification, and unsupported wire/rod endpoint cases now abstain explicitly as singular instead of falling through to a vague search failure.
Latest shell gap fix (2026-05-24): the search planner now asks shell-specific queries, the parser no longer short-circuits thin spherical-shell questions into the generic distributed-charge abstain path, and the router now answers the multi-point spherical-shell prompt as `1 m: 0 N/C; 5 m: 2157.01243 N/C` with web shell evidence in the trace.
Latest shell-potential expansion (2026-05-24): thin spherical-shell potential is now covered too. Direct solver tests and the router both pass for the single-point and multi-point shell potential prompts, with the web-backed path returning `1 m: 17.9751 kV; 5 m: 10.7851 kV` and the local corpus ensuring the family still works when the env flag is off.
Latest wave-frequency fix (2026-05-24): request `bd9b5e18-5d1e-44f9-9e46-2e5594eeb514` failed because `m/s` was not parsed as speed and `ring` matched as a substring of `string`, causing a string-wave problem to be framed as distributed charge. Runtime now parses speed units, supports deterministic `wave_frequency` (`f = v / wavelength`), and uses exact structural-term matching for method-frame proposals.

Latest physics expansion (2026-05-24): the physics solver now also supports uniformly charged disk axis field, infinite line charge field, dipole axial field, dielectric capacitance change, inverse capacitor-energy-to-voltage, symmetric bridge resistance, nested composite capacitance/resistance, uniformly charged disk axis potential, uniform-sphere inside/outside field, circular-loop magnetic field at center, and spherical-capacitor capacitance deterministically. Candidate gates still abstain on genuinely missing-information charge/voltage traps instead of inventing unsupported simple relations.
Latest AC-power parser fix (2026-05-25): average-power / RMS-voltage wording in AC or network contexts now routes to the RMS power formula when resistance is present, instead of taking the generic DC `P = V I` shortcut.
Latest AC-power extension (2026-05-25): explicit power-factor / phase-angle prompts now route to `ac_power_vi_cos_phi`, so average-power AC cases with `cos(phi)` or a phase angle no longer require a custom workaround.
Latest mixed-capacitance hardening (2026-05-25): the mixed series/parallel capacitor parser now covers both `series -> parallel` and `parallel -> series` phrasing variants, including `pair`/`branch` wording, and has regression coverage for both directions.
Latest physics refusal tightening (2026-05-25): open-switch current prompts now abstain as `unknown` instead of over-answering `0 A`, and ambiguous nested-topology resistance prompts with `nested branch has ... all in parallel with ...` now abstain instead of over-answering `2 ohm`.
Latest physics explanation refinement (2026-05-25): underdetermined physics cases now surface the specific missing or contradictory data in the unknown explanation, so force/resistance prompts mention missing charge magnitudes or conflicting voltage notes instead of generic lookup wording.

Latest verification after runtime refactor: `pytest -s -q` reports `132 passed`; `pytest -s -q tests/test_router.py tests/test_explanation_quality_regressions.py tests/test_policy_unknown_handling.py` reports `55 passed`; `python scripts/run_quality_eval.py` reports logic answer accuracy `0.9750889679715302`, physics answer accuracy `0.9854651162790697`, explanation consistency `0.9912790697674418`, hallucinated premise rate `0.0`, and average latency around `10 ms`.
Latest focused 50+50 eval (2026-05-25): logic `50/50` answer accuracy and `50/50` explanation consistency on the first 50 hard academic-policy cases; physics `50/50` answer accuracy and `50/50` explanation consistency on the first 50 hard physics cases after correcting the `hc_phys_031` gold label to `2 A`.
Latest unseen-physics expansion (2026-05-24): the solver now handles a generic circular-arc-at-center family, including quarter-circle and numeric subtended-angle prompts, plus symmetry-zero centers for square, rectangular, equilateral-triangle, and regular-polygon wire loops. It also now handles finite rod / line-segment electric fields on the axis outside the rod, both from the center and from one end, finite rod electric potential on the perpendicular bisector and on-axis outside the rod, plus ring potential on-axis and square-loop potential at the center. More recent deterministic additions also cover uniformly charged disk axis potential, uniform-sphere inside/outside field, circular-loop magnetic field at center, and spherical-capacitor capacitance. The live web-backed ring-axis probe still returns `2411.613211 N/C` with `search_trace` evidence, so the search path remains active after the deterministic arc/symmetry/rod/potential additions.

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
