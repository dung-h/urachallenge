# Hard Eval v2 — Root-Cause Fixes Report

**Date**: 2026-06-06
**Status**: COMPLETED (physics qualitative + new domain adapters + logic reasoners)

## Summary

Applied root-cause structural fixes (no question-text overrides, AGENTS.md §20
compliant) across the physics and logic pipelines. Overall hard_eval_v2 score
improved from **35/60 (58.3%)** to **49/60 (81.7%)**.

| Task | Before | After |
|------|--------|-------|
| Physics | 14/30 (46.7%) | 25/30 (83.3%) |
| Logic | 21/30 (70.0%) | 24/30 (80.0%) |
| **Total** | **35/60 (58.3%)** | **49/60 (81.7%)** |

## Physics Qualitative Fixes (solver.py, qualitative_parser.py)
- Moved `other_vars_constant` check after formula lookup (formula-aware).
- Prepositional-phrase guard in `_direction_near()` ("from a point charge").
- Implicit constants (k, charges) recognized so E-field/Coulomb resolve.
- Passivity-based formula tie-break (P=V²/R preferred over P=V*I).
- Exact change-factor computation via SymPy exponent analysis.
- Cases fixed: phys_01, 02, 05, 06.

## New Domain Adapters (deterministic, AGENTS.md §13.2)

### Optics adapter (app/physics/adapters/optics.py — NEW)
- Snell's law refraction: θ2 = asin(n1·sin(θ1)/n2). Extracts refractive
  indices ("n=1.5") and bare angles ("30°") with general regex.
- Thin lens / spherical mirror: 1/f = 1/d_o + 1/d_i; mirror f = R/2.
- Cases fixed: phys_10 (refraction 19.47°), phys_13 (lens 60cm), phys_14 (mirror 60cm).

### Fluids adapter (app/physics/adapters/fluids.py — NEW)
- Archimedes submerged fraction: f = ρ_object / ρ_fluid.
- Case fixed: phys_11 (buoyancy 0.6).

### Mechanics adapter extensions (app/physics/adapters/mechanics.py)
- Perfectly inelastic collision (momentum conservation): v = (m1v1+m2v2)/(m1+m2),
  with opposite-direction sign detection.
- Charged-particle acceleration speed (work-energy qU=½mv²): v = sqrt(2qU/m).
- Work against gravity on an incline: W = m·g·d·sin(θ).
- Cases fixed: phys_18 (electron 5.93e6 m/s), phys_22 (work 250J), phys_23 (momentum 0.8 m/s).

### Unit/dimension support (unit_converter.py, dimensions.py)
- Added density (kg/m³, g/cm³), volume (m³, L, mL), temperature (K) and
  specific-heat (J/(kg·K)) units with proper SI dimensions.

## Retrieval-Grounded LLM Method Solver (NEW — app/physics/retrieval_grounded_method.py)

Implements the "retrieve the solving METHOD, not just a formula" strategy
(user request; AGENTS.md §3.1b search allowance, §13.2 deterministic verify).

**Why**: the old ungrounded LLM rescue chose formulas from the model's unaided
memory and got them wrong (e.g. v=qU/m instead of v=sqrt(2qU/m) for electron
speed; W=F·d·cos θ instead of m·g·d·sin θ for work against gravity). Meanwhile
the search path retrieved the right reference pages (Snell's law, etc.) but the
regex extractor could not bind variables when the parser left target_quantity
empty.

**Pipeline**:
1. Retrieve method evidence via web search (reuses method_search retrieval).
2. Ground an LLM call with the retrieved snippets; the LLM returns a structured
   method: target quantity + unit, the formula solved for the target, and each
   variable bound to an SI value from the question.
3. Backend RE-COMPUTES with `safe_eval_expression` (LLM arithmetic never
   trusted) and applies an acceptance gate: dimensional consistency with the
   requested target + finite/plausible magnitude. Abstains (None) otherwise →
   falls through to the agent loop.

**Wiring**: runs in `_solve_impl` after deterministic adapters fail, before the
broad agent loop, only when `use_search` and web search are enabled.

**Supporting changes**:
- Extended `safe_eval_expression` to support sin/cos/tan/asin/acos/atan/log/
  log10/exp/radians/degrees/abs (needed for optics/wave/thermal methods).
- Greek-symbol normalization so LLM variable keys (e.g. `λ`) match the
  normalized expression tokens (`lam`).

**Verified end-to-end** (out-of-adapter questions, real LLM + web search):
- "wave: f=50 Hz, λ=4 m → speed" → 200 m/s (formula v=f·λ retrieved), conf 0.7
- "rod thermal expansion, α=1.2e-5, ΔT=50 → Δlength" → 1.2e-3 m
  (formula Δl=l·α·ΔT retrieved), conf 0.6

This generalizes to unseen physics topics: any question whose method is on the
web can now be solved with a retrieved + backend-verified formula, instead of
abstaining or trusting ungrounded model arithmetic.



### Disjunction & conditional parsing
- Eligibility disjunction matcher ("To be eligible, one must have A or B").
- "unless" rewrite ("X unless Y" ≡ "if not Y, then X").
- Exclusive-or guard ("but not both") so XOR abstains instead of unsound "yes".
- Case fixed: logic_15 (disjunction).

### Transitive comparison reasoner (app/logic/_comparison_reasoner.py — NEW)
- Builds a directed comparison graph per dimension (height/size/age/...).
- Handles chained comparisons, reverse queries, superlatives, and transitive
  equality (union-find). Entity normalization strips "object/item/..." prefixes.
- Cases fixed: logic_03 (transitive_5 → no), logic_23 (tallest → Diana),
  logic_24 (A=B,B=C ⊢ A=C → yes).

## Infrastructure / Environment

### vLLM + flashinfer incident (documented in AGENTS.md)
- After an abrupt machine shutdown, vLLM failed with "Engine core init failed".
- Root cause traced: the V1 engine imports `flashinfer.sampling`, and the
  installed `torch_c_dlpack_ext` native lib has an undefined symbol (ABI
  mismatch vs torch 2.6) → `OSError`.
- Fix: launch vLLM with `VLLM_USE_V1=0` (V0 engine, no flashinfer). Loads fine.
- Documented the mandatory `.venv` rule and the V0-engine launch command in AGENTS.md.

### Bug fix
- Removed a leftover `print(f"DEBUG PAYLOAD: ...")` in app/llm_client.py.

## Tests
- tests/test_logic_solver.py: 27 passed
- tests/test_physics_qualitative.py: 24 passed
- tests/test_physics_numeric_unit_agreement_properties.py + ir_architecture + schemas: 24 passed
- No new regressions introduced.

## Remaining failures (future work)
- phys_28 parallel-plate (multi-variable qualitative factor: area×2, sep/2 → C×4)
- phys_29 series-vs-parallel power comparison (needs circuit-comparison reasoner)
- phys_30 SI unit of magnetic flux (conceptual lookup; LLM returned "Tesla")
- phys_03/04 capacitor + dielectric κ (needs dielectric-insertion reasoning)
- logic_09 unless (parse fixed, but token-matching can't equate "continues to
  rain" with "does not stop raining" — semantic gap, LLM/Z3 territory)
- logic_12 contradiction (mutual-exclusivity semantics)
- logic_13 policy 3-of-4 threshold; logic_16 XOR (abstains, needs full XOR eval);
  logic_19 causal chain; logic_21/22 MCQ elimination

## Pre-existing failures (NOT caused by this work, confirmed via git stash)
- test_physics_coulomb_geometry_regressions.py::test_series_parallel_resistors_network_variation_1
- test_negation_scope.py::test_negated_consequent_entails_no[generic_class_consequent]

## Cleanup Done
- ✓ Deleted all temp scripts (_tmp_*.py, test_*.py, fix_*.py, debug_*.py)
- ✓ Kept: reports/physics_qualitative_fix_report.md, reports/hard_eval_v2_fixes_report.md
- Workspace tidiness: MAINTAINED


---

## Session 3 addendum: conceptual-lookup retrieval + routing + determinism fixes

**Date**: 2026-06-06

### 1. Conceptual-lookup retrieval-grounded path (NEW)
Extended `app/physics/retrieval_grounded_method.py` with `solve_conceptual_lookup`:
non-numeric questions (SI unit names, definitions) are answered by the LLM
grounded in retrieved web snippets, then **verified by evidence-grounding** —
the answer string must appear in the retrieved references, otherwise it abstains.
This is the non-numeric analogue of backend recomputation (the LLM proposes, the
evidence-grounding check verifies). Wired into `_solve_impl` after the numeric
retrieval-grounded method.

### 2. Magnetic-flux unit correctness bug (formula_registry.py)
`lookup_qualitative` had `if "magnetic" in q: return "Tesla"`, which wrongly
answered "magnetic flux" (Weber) as Tesla. Fixed to distinguish:
- magnetic flux / flux → **Weber**
- magnetic flux density / magnetic field → **Tesla**
This is a correctness fix to an over-broad substring rule (AGENTS.md §20).

### 3. Determinism fix: registry lookup + open-switch no longer gated by `use_search`
Two PURE LOCAL deterministic facts were gated behind the LLM orchestrator's
non-deterministic `use_search` planning choice, so identical questions answered
inconsistently between runs:
- The qualitative registry lookup ("SI unit of X") → extracted to
  `_qualitative_registry_hit`, runs unconditionally (defers to the SymPy
  monotonic reasoner when a change-shape is detected).
- Open-switch current ("switch is open → 0 A") → runs unconditionally.
Both now return stable, physically-correct answers regardless of the planner.

### 4. Routing fix: conceptual physics questions (runtime_workflow.py)
"What is the SI unit of X?" carries no numeric signal and was falling through to
the LOGIC solver. Added a `has_physics_concept_signal` (unit-of / measured-in
phrasing + a physics-quantity term) so these route to PHYSICS.

### 5. safe_eval_expression extended
Added sin/cos/tan/asin/acos/atan/log/log10/exp/radians/degrees/abs so retrieved
optics/wave/thermal methods evaluate. Greek-symbol normalization aligns LLM
variable keys (e.g. `λ`) with normalized expression tokens.

### Verified end-to-end (API)
- "SI unit of magnetic flux" → Weber (was Tesla)
- "SI unit of energy" → Joule (was unknown — routed to logic)
- "switch is open, what current?" → 0 A (was fl'unknown' depending on planner)
- buoyancy → 0.6; refraction → 19.47°; lens/mirror → 0.6 m
- out-of-adapter via retrieval-grounded: wave speed v=fλ → 200 m/s; thermal Δl=lαΔT → 1.2e-3 m

### Benchmark (clean run, hard_eval_v8)
Physics 25/30 (83.3%), Logic 24/30 (80.0%), **Total 49/60 (81.7%)**.

### Tests
- tests/test_physics_qualitative.py: 24 passed
- tests/test_physics_scene_abstention.py + numeric/ir/schemas: passing
- tests/test_router.py: open-switch test updated to assert the correct 0 A;
  2 remaining router failures (planner_source = llm vs deterministic) are
  PRE-EXISTING and environment-dependent (vLLM availability), confirmed via
  baseline stash.

### Infrastructure note
- After abrupt shutdowns, restart vLLM with `VLLM_USE_V1=0` (flashinfer ABI bug)
  and the API server from `.venv`. A stale non-project server on :8000 (mock
  provider, 404 on /predict) must be killed before starting ours.


---

## Session 4: random-batch generalization probe + root-cause fixes

**Date**: 2026-06-06

Ran a fresh 35-question batch (`scripts/random_batch_eval.py`) NOT in hard_eval_v2
to surface generalization gaps. Score improved **24/35 → 30/35** (Physics 14→18,
Logic 10→12; remaining "fails" include 2 grader false-negatives now fixed).

### Root causes found and fixed

1. **`eligibility_words` NameError (app/logic/solver.py)** — CRITICAL latent bug.
   `solve_forward_chaining` referenced an undefined `eligibility_words`, crashing
   forward chaining on every rule-firing case (silently caught → wrong fallback
   answers). Defined it as the intended status/eligibility stem set. Fixed
   negated-antecedent modus ponens ("If Leo did not eat → hungry" + "Leo did not
   eat" → yes) and improved many logic regression tests (a stashed-baseline run
   could not even collect these suites).

2. **Shadowing dead-code stubs (app/logic/_subject_chain.py)** — `_has_universal_no_conflict`,
   `_universal_positive_support`, `_universal_negative_support` each had a second
   `(*args, **kwargs) -> None` stub defined AFTER the real implementation, so the
   stub won at module load and disabled contradiction detection + universal
   support. Removed the three shadowing stubs (kept the unused
   `_universal_contrapositive_support` stub, which has no real impl and is never
   called). Restored the "All metals conductive / No metals conductive" →
   unknown contradiction behavior.

3. **`_explicit_negative_premise` mis-fired on conditional rules** — a "not" in a
   rule's antecedent ("If Leo did not eat, then ...") was read as an explicit
   negative fact about the consequent, returning a WRONG "no". Added a guard to
   exclude if/all/every/each/some/no/then rules. Now abstains/forward-chains
   instead of asserting the wrong negative.

4. **Physics target detection (app/physics/parser.py)** — "what is the resistor's
   value" / "value of the resistor" now map to target=resistance (rp_phys_01).

5. **Coulomb force from two charges (app/physics/parser.py)** — added a candidate
   computing F = k·q1·q2/r² from "two point charges of 2 μC each, 10 cm apart"
   (handles explicit pair and "each"/equal single value) (rp_phys_12).

6. **Qualitative "radius" vocab (app/physics/qualitative_parser.py)** — added
   "radius" → r so "if the radius is tripled, how does the field change" → 1/9
   (rp_phys_15).

7. **Inelastic collision "at rest" (app/physics/adapters/mechanics.py)** — when a
   second mass is present and the text says "at rest"/"stationary", the missing
   second velocity is treated as 0 m/s, enabling momentum conservation
   (rp_phys_20).

8. **Transitive temporal "before/after" (app/logic/_comparison_reasoner.py)** —
   added a temporal precedence graph so multi-hop "A before B before C before D,
   did A happen before D?" → yes (rp_log_10).

9. **"neither A nor B" (app/logic/_subject_chain.py)** — added a structural
   handler: "X is neither A nor B" → "Is X an A?" = no. Uses raw content tokens
   (the clean variant filters role nouns like doctor/nurse) (rp_log_11).

10. **Grader false-negatives (scripts/random_batch_eval.py)** — the local grader
    now normalizes SI prefixes (720 μJ == 7.2e-4 J, 2 mF == 2e-3 F).

### Tests
- tests/test_logic_solver.py: 27 passed
- tests/test_physics_qualitative.py: 24 passed
- 115 passed across core physics+logic suites; only the PRE-EXISTING
  `series_parallel_resistors_network_variation_1` and
  `test_negated_consequent_entails_no[generic_class_consequent]` remain (both
  confirmed pre-existing via baseline stash).
- Net effect on logic regression suites: removing the shadowing stubs + the
  eligibility_words fix turned a baseline that couldn't even collect into 55
  passing (10 remaining failures are pre-existing stub-disabled features:
  existential-witness, contrapositive, negation-scope-in-rules).

### Known remaining (future work)
- rp_log_03 negated-antecedent via the LLM path returns "unknown" (deterministic
  path gives "yes"); the wrong "no" is fixed. Dispatch ordering in the LLM path
  reaches a conditional-abstain handler before forward chaining.
- rp_log_04 "unless" via API still abstains (semantic gap: "code was not entered"
  vs "the code is entered" negation matching).


---

## Session 5: Logic-LM-style self-refinement loop + FOL soundness guard

**Date**: 2026-06-06
**Strategic context**: After repeated whack-a-mole, surveyed the literature
(Logic-LM EMNLP'23 arxiv:2305.12295; neuro-symbolic <15B+Z3 arxiv:2509.12645;
PAL arxiv:2211.10435; Physics Reasoner arxiv:2412.13791). Consensus: stop
extending heuristics; make the LLM a TRANSLATOR evaluated by the symbolic
solver, and add a SELF-REFINEMENT loop driven by solver error messages.

### 1. Self-refinement loop (app/logic/fol_z3_pipeline.py)
`_solve_fol_z3_dsl` now retries translation (bounded by
`URA_FOL_Z3_REFINE_ROUNDS`, default 2) when the first attempt yields a
non-decisive Z3 verdict (undetermined/abstained/error/low-confidence) or leaves
premises unsupported. `_build_refinement_feedback` composes structural feedback
(missing chain link, atom-name mismatch between query predicate and clause
conclusion, unsupported premises, direction-gate rejection) and feeds it back to
the LLM. Z3 remains the decision authority; only the translation is refined.
- Fixed **rp_log_13 "Is Dana admitted?" (3-of-4 threshold policy)** → yes via
  `self_refinement_rounds=1` (failed across multiple prior turns).
- Latency stays within the ~30s budget (Dana: 12.2s, 3 calls).

### 2. FOL contradiction soundness guard (app/logic/solver.py) — PRE-EXISTING BUG
Found (independent of refinement): when ground premises are mutually
contradictory ("bridge is safe" + "bridge is not safe"), the LLM-FOL translator
silently DROPS the conflicting premise and Z3 entails a spurious definite
verdict ("yes"). Added a `_premises_contain_contradiction` guard that skips the
LLM-FOL fast path (via `_SkipFolPath`) so the deterministic solver detects the
conflict and abstains (Req 11.1). rp_log_13 "Is the bridge safe?" → unknown
(was wrongly "yes", even with refinement disabled — so this is a real soundness
fix, not a refinement artifact).

### 3. Grader bug fix (scripts/random_batch_eval.py)
The SI-prefix normalization wrongly treated a lone unit letter as a prefix
("2 m" → 2 milli = 0.002; "3.6 N" → nano). Now only scales when the prefix is
followed by a base-unit letter (mF, μJ, kΩ). Physics regraded 18/20 → 20/20.

### Random-batch result (35 fresh questions, clean grader)
Physics 20/20, Logic 11/15, **Total 31/35** (was 24/35 at the start of the
random-batch effort).

### Honest limit (model capability)
rp_log_03 (negated antecedent), rp_log_04 (unless), rp_log_05 (only_if) remain
"unknown" via the API: the 3B FOL translator systematically mis-handles negation
atom alignment, and refinement feedback cannot fix what the small model cannot
translate. This matches the literature (Logic-LM used much larger translators).
The deterministic BFS solves some of these in isolation, but a conditional-status
handler intercepts before forward chaining in the full LLM flow. These are the
true ceiling cases; abstaining is sound (§20.4) and the wrong answers are gone.

### Tests
- 43 passed across logic_solver / contradiction_exception / mcq_llm_veto /
  verifier_abstain_paths. No regressions from refinement or the guard.
