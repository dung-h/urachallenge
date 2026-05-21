---
name: exact-physics
description: Physics and electrical reasoning rules for URA EXACT items using solver-first formulas, unit conversion, and validated computation traces.
---

# EXACT Physics Skill

Use this skill for electrical or physics word problems.

## Required Discipline

- Use deterministic Python/app solver arithmetic as authority.
- Do not do final arithmetic in model text when a solver trace exists.
- Check `formula_id`, `physics_variables`, `confidence`, and `cot` in trace.
- Preserve units in final answers.
- If parsing fails, report the formula or parser gap instead of guessing.

## Formula Areas

Expected formulas include:

- Ohm law: `V = I * R`
- Power: `P = V * I`, `P = I^2 * R`, `P = V^2 / R`
- Resistance: series, parallel, composite series/parallel
- Capacitance: charge, energy, series, parallel
- Electric field and force formulas
- Unit conversions such as mV, mA, kohm, microfarad

## Trace Review

Use `/trace/{request_id}` to verify:

- `formula_id`
- `physics_variables`
- `answer`
- `explanation`
- `model_calls`
- `fallback_accepted`

If `model_calls=0`, the deterministic solver handled the item.
Load this skill at most once per item. After reading it, proceed to the app helper or trace inspection instead of reloading the skill.
