---
name: exact-challenge
description: URA EXACT challenge operating contract for using OpenCode as an agent while keeping backend solver output as final answer authority.
---

# EXACT Challenge Contract

Use this skill for any URA EXACT challenge question, benchmark sample, answer check, or failure analysis.

## Authority

- The local app is the answer authority.
- Call `POST /predict` for challenge items.
- Read `GET /trace/{request_id}` before presenting final conclusions.
- Raw model output is proposal-only, including OpenCode's own reasoning.
- Final answer must come from validated backend response or from a code patch plus focused tests.

## Output Contract

The minimum response fields are:

- `answer`
- `explanation`

Encouraged trace fields are:

- `premises`
- `cot`
- `fol`
- `confidence`
- `task_type`
- `raw_json_validity`
- `repaired_json_validity`

Do not invent extra final JSON fields.

## Workflow

1. Use `scripts/exact_agent_request.py` with a JSON payload.
2. Inspect `response`, `trace`, `trace_url`, `model_calls`, and `solver_used`.
3. If `response.answer` is not `unknown`, verify the explanation cites only actual premises or formulas.
4. If `response.answer` is `unknown`, do not change it to `yes` or `no` from model intuition alone.
5. If the output looks wrong, classify the failure and use `exact-debug`.

If you already loaded this skill for the current item, do not load it again.
For concrete answer checks, move directly to the helper command after reading this section.
When using OpenCode's `bash` tool, include both a short `description` and the `command`.

## Safety Rules

- Do not use web search or web fetch for self-contained logic/physics items.
- Do not edit frozen datasets or scorers.
- Do not enable live fallback in production config.
- Do not trust model confidence.
- Include the trace URL when reporting a manually tested item.
