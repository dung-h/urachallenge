# ADR 0002: LLM Output Is Proposal-Only

Status: Accepted
Date: 2026-05-01

## Context

LLMs are useful for extraction, candidate reasoning, and explanation drafting, but they can hallucinate premises, invert rule direction, make arithmetic errors, or emit invalid JSON.

## Decision

LLM output is proposal-only unless independently verified by deterministic backend logic and final schema validation.

This applies to:

- live fallback answers
- symbolic proposal JSON
- premise selection candidates
- explanation rewrites
- structured extraction outputs

## Consequences

- Keep live LLM fallback disabled by production default.
- Keep LLM explanation rewrite disabled by production default.
- Log raw output validity and repaired output validity in benchmarks.
- Do not use model-reported confidence as system confidence.
- Derive confidence from parser, solver, unit, premise, verification, and validation signals.

## Links

- `configs/pipeline.yaml`
- `docs/live_fallback_findings.md`
- `reports/phase_25_llm_symbolic_proposal_report.md`
- `reports/phase_28_llm_symbolic_root_cause_report.md`
