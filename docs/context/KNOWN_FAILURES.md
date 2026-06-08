# Known Failures

Use this before changing fallback, symbolic proposals, sidecars, validation, datasets, or scoring.

## Raw LLM Output as Final JSON

Failure mode:

- A model emits plausible JSON that is malformed, unsupported, or inconsistent with validated evidence.

Mitigation:

- Final JSON must be assembled by backend code.
- Validate with Pydantic `QAResponse`.
- Track raw JSON validity and repaired JSON validity when benchmarking model output.

## Live Fallback as Production Authority

Failure mode:

- Live model fallback can produce unsupported answers, hallucinated premises, or inconsistent explanations.

Mitigation:

- Keep live fallback disabled by default.
- If tested, treat it as experiment-only and validate all outputs.
- Do not use raw fallback output as final answer.

## Necessary Condition Converted to Sufficient Rule

Failure mode:

- Text such as `eligibility requires GPA at least 3.5` is translated as `GPA at least 3.5 implies eligibility`.
- This caused accepted-wrong symbolic sidecar results in Phase 27B and was diagnosed in Phase 28.

Mitigation:

- Preserve Phase 29 semantic-direction gates.
- Reject positive target rules derived from necessary-only wording unless sufficient language is present.
- Keep source direction metadata in proposal evaluation artifacts.

## Blocker or Exception Converted to Positive Eligibility

Failure mode:

- Text describing blockers, exceptions, or disqualifiers is translated as positive eligibility.

Mitigation:

- Use `app/logic/symbolic_direction.py` detection.
- Reject blocker/exception wording represented as positive `all_of_implies` target rules.

## Baseline-Unknown Override

Failure mode:

- Sidecar or symbolic proposal overrides a deterministic baseline unknown answer with an unsupported positive answer.

Mitigation:

- Preserve baseline-unknown override gates.
- Allow override only when proof rules are sourced from sufficient-rule language.

## Misleading Small-Limit Experiments

Failure mode:

- Phase 27 limit-2 symbolic model search appeared safe but failed at Phase 27B larger limit.

Mitigation:

- Use staged limits before recommendation.
- Do not promote candidates from tiny sample results.
- Treat wrong-when-accepted as a hard safety metric.

## MCQ Option-to-FOL Proposals Accept Wrong Answers

Failure mode:

- A local model translates an MCQ option into a parseable FOL claim, the prover accepts the claim, but the accepted option is not the gold answer.
- The 2026-05-10 15-model shootout found no model with both non-zero coverage and zero wrong-when-accepted on the first 20 curated MCQ rows.

Mitigation:

- Do not promote option-to-FOL model proposals without stricter acceptance gates.
- Preserve wrong-when-accepted as a hard gate.
- Add semantic direction and baseline-unknown override gates before running larger MCQ symbolic model benchmarks.

## Context Drift Across Agents

Failure mode:

- A new agent reads a subset of reports or chat history and misses current production boundaries.

Mitigation:

- Start from `docs/context/AGENT_ONBOARDING.md`.
- Keep `docs/context/PROJECT_STATE.md` and indexes current.
- Use ADRs for decisions that should not be casually re-litigated.
