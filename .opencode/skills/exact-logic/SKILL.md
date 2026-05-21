---
name: exact-logic
description: Logic reasoning guardrails for URA EXACT items, including premise IDs, unknown handling, MCQ traps, and academic policy thresholds.
---

# EXACT Logic Skill

Use this skill for logic, academic policy, premise entailment, MCQ, and unknown/refusal cases.

## Required Discipline

- Use only the given premises.
- Cite only premise IDs present in the input.
- Prefer `unknown` when premises do not entail the claim.
- Never turn necessary-only language into a sufficient rule.
- Never override deterministic `unknown` with `yes` or `no` from a model proposal alone.

## Common Traps

- Affirming consequent: `If A then B`, `B`, therefore `A` is invalid.
- Denying antecedent: `If A then B`, `not A`, therefore `not B` is invalid.
- Existential premise: `Some X are Y` does not prove a named X is Y.
- MCQ unknown: choose the option that represents insufficient information when no option is entailed.
- Academic threshold: compare facts such as GPA, CPA, credits, attendance against the rule.
- Exception/blocker: explicit negative or exception premises can override a positive policy.

## Trace Review

Check:

- `selected_premises`
- `proof_steps`
- `proof_step_validity`
- `fallback_used`
- `llm_trace`

If `llm_trace` proposes a stronger answer than the deterministic solver, require backend validation or a focused patch plus test before accepting it.

Load this skill at most once per item. After reading it, proceed to the app helper or trace inspection instead of reloading the skill.
