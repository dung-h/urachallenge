---
name: exact-debug
description: Failure analysis workflow for URA EXACT outputs: reproduce, inspect trace, classify failure, patch narrowly, and run focused tests.
---

# EXACT Debug Skill

Use this skill when a challenge item is wrong, unsafe, ambiguous, or surprising.

## Workflow

1. Reproduce through `scripts/exact_agent_request.py`.
2. Inspect `/trace/{request_id}`.
3. Classify the failure:
   - routing error
   - parser miss
   - wrong formula
   - unit conversion gap
   - invalid logic inference
   - unsafe fallback proposal
   - explanation-only issue
4. Patch only the smallest relevant module.
5. Add or update a focused regression.
6. Run the narrow pytest set that covers the changed behavior.

## Do Not

- Do not edit unrelated modules.
- Do not change public `QAResponse` schema without explicit approval.
- Do not modify frozen datasets or scorers while changing solver behavior.
- Do not enable production fallback defaults.
- Do not hide a failing model proposal; preserve it in trace.

## Report Format

Report:

- item id or request id
- expected answer
- actual answer
- trace URL
- failure class
- files touched
- focused tests run

Load this skill at most once per failure analysis. After reading it, reproduce or inspect the trace instead of reloading the skill.
