# ADR 0004: Semantic Direction Gates Required

Status: Accepted
Date: 2026-05-01

## Context

Phase 27 symbolic model search produced a promising small-limit result, but Phase 27B invalidated it with non-zero wrong-when-accepted. Phase 28 diagnosed the core failure: LLM symbolic proposals converted necessary conditions into sufficient rules, especially cases like `Scholarship eligibility requires GPA at least 3.5`.

## Decision

Any LLM symbolic proposal acceptance path must include deterministic semantic-direction gates.

The gates reject:

- positive target rules derived from necessary-only wording without sufficient language
- single-condition positive eligibility/status rules from necessary-only or ambiguous sources
- blocker or exception wording represented as positive `all_of_implies` target rules
- baseline-unknown overrides unless proof rules are sourced from sufficient-rule language

## Consequences

- Preserve `app/logic/symbolic_direction.py` behavior when changing symbolic proposal validation.
- Keep direction metadata in evaluation artifacts.
- Treat wrong-when-accepted as a hard safety gate.
- Do not rely on tiny sample limits for symbolic proposal safety claims.

## Links

- `reports/phase_27b_symbolic_best_candidate_report.md`
- `reports/phase_28_llm_symbolic_root_cause_report.md`
- `reports/phase_29_symbolic_direction_gate_report.md`
- `app/logic/symbolic_direction.py`
- `app/logic/symbolic_proposal_validator.py`
