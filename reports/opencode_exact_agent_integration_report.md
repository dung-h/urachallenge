# OpenCode EXACT Agent Integration Report

Date/time: 2026-05-21 Asia/Saigon

## What Was Done

- Added project-local OpenCode EXACT skills under `.opencode/skills/`.
- Added repo-local OpenCode config at `.opencode/opencode.json`.
- Added `scripts/exact_agent_request.py` for consistent `/predict` plus `/trace` calls.
- Kept backend `/predict` as answer authority; OpenCode/model output remains proposal-only.

## Files Changed

- `.opencode/opencode.json`
- `.opencode/skills/exact-challenge/SKILL.md`
- `.opencode/skills/exact-logic/SKILL.md`
- `.opencode/skills/exact-physics/SKILL.md`
- `.opencode/skills/exact-debug/SKILL.md`
- `scripts/exact_agent_request.py`
- `.gitignore`

## Commands Run

- `python -m json.tool .opencode/opencode.json`
- YAML frontmatter validation for all `.opencode/skills/*/SKILL.md`
- `python -m py_compile scripts/exact_agent_request.py`
- Helper smoke for `acad_001`, `sr019`, `physx_001`, `sr005`
- OpenCode skill discovery smoke with `ollama/qwen2.5:7b`
- OpenCode helper smoke with `ollama/qwen2.5:7b --pure`
- `python -m pytest -s -q tests/test_phase_6_fallback.py tests/test_router.py tests/test_schemas.py tests/test_policy_unknown_handling.py tests/test_invalid_inference_traps.py`

## Metrics Collected

| Check | Result |
| --- | --- |
| Skill frontmatter | 4/4 valid |
| Helper smoke | 4/4 pass |
| OpenCode skill discovery | pass |
| OpenCode pure helper smoke | pass |
| OpenCode exact-runner unattended smoke | partial; model/tool schema issues remain |
| 11-case app/helper batch | 11/11 pass |
| Focused pytest | 29 passed |

## Known Issues

- `gemma3:4b` does not support OpenCode tools through the tested Ollama path.
- `qwen2.5:7b` supports tools, but as `exact-runner` it can omit the required bash `description` field or fall into repeated skill loads.
- `opencode run --pure --model ollama/qwen2.5:7b` can call the helper successfully when explicitly instructed to include `description` and `command`.

## Next Recommended Action

- Treat OpenCode as an assisted operator for now, not a fully unattended authority.
- Keep using `scripts/exact_agent_request.py` for traceable manual/OpenCode checks.
- If OpenCode is promoted further, test a stronger local tool-calling model or a custom OpenCode command wrapper.

## Cleanup Done

- Added root ignore rules for `.opencode/node_modules/` and local OpenCode package/lock files.
- Confirmed no lingering `opencode run` process remained after smoke tests.
- Did not touch `clones/`.
