# Agent Onboarding

This repository uses repo-native context. Do not rely on chat history as project memory.

Agents should self-bootstrap from these context files. Users do not need to run context scripts before normal requests.

## Read First

Read these files before making project decisions:

1. `AGENTS.md`
2. `docs/context/PROJECT_STATE.md`
3. `docs/context/CURRENT_WORK.md`
4. `docs/context/TASK_ROUTER.md`
5. `docs/context/CONTEXT_MANIFEST.yaml`
6. `docs/context/REPORT_INDEX.md`
7. `docs/context/KNOWN_FAILURES.md`

Read these when changing architecture, authority boundaries, or experiment policy:

1. `docs/adr/0001-solver-first-authority.md`
2. `docs/adr/0002-llm-output-is-proposal-only.md`
3. `docs/adr/0003-sidecars-remain-experiment-only.md`
4. `docs/adr/0004-semantic-direction-gates-required.md`
5. `docs/adr/0005-repo-native-agent-context.md`

## Default Operating Rules

- Work in WSL at `/mnt/d/URA_challenge` unless explicitly instructed otherwise.
- Treat `AGENTS.md` as canonical project policy.
- Do not modify `AGENTS.md` unless explicitly requested.
- Do not enable live LLM fallback, Z3, or Datalog by production default.
- Do not treat raw LLM output as final JSON.
- Do not change final `QAResponse` schema without explicit approval.
- Do not modify frozen datasets or scorers while changing solver/model behavior.
- Do not download models unless the user explicitly requests model acquisition.
- Prefer short context/index files first; read detailed reports only when the index points to them.
- Read raw logs or JSONL only when debugging a specific failure.

## Bootstrap Flow

1. Read the files in `Read First`.
2. Classify the request with `docs/context/TASK_ROUTER.md`.
3. Stay within the initial budget in `docs/context/CONTEXT_MANIFEST.yaml` when possible.
4. If editing code, inspect the relevant files before proposing changes.
5. If editing docs only, keep context files concise and update indexes consistently.
6. After substantive work, update `docs/context/PROJECT_STATE.md`, `docs/context/CURRENT_WORK.md`, and any relevant index.

## Agent Self-Bootstrap Contract

For short or ambiguous prompts, the agent must route the task itself:

1. Use `TASK_ROUTER.md` to classify the prompt.
2. Use `CONTEXT_MANIFEST.yaml` to identify route-required files and budget limits.
3. Read only route-required files first.
4. Do not broad-scan reports, logs, outputs, or source files before classification.
5. Use `casual_or_unclear` and ask one clarifying question if the route or current work is ambiguous.

The user is not expected to run `scripts/context_router.py` or `scripts/context_lint.py` for normal interaction. Those scripts are optional helpers for verification, CI, maintenance, and cases where the agent wants to double-check routing.

## Context Budget Guidance

- Use `docs/context/PROJECT_STATE.md` for current status.
- Use `docs/context/TASK_ROUTER.md` to classify short prompts.
- Use `docs/context/CONTEXT_MANIFEST.yaml` for machine-readable budgets and route constraints.
- Use `docs/context/REPORT_INDEX.md` to choose reports.
- Use `docs/context/PHASE_HISTORY_INDEX.md` for historical phase summary.
- Use `docs/context/DECISION_INDEX.md` before changing durable decisions.
- Use `docs/context/KNOWN_FAILURES.md` before touching symbolic sidecars, validators, fallback, or scoring.
- Use `docs/context/CONTEXT_HEALTH.md` to detect stale context.
- Avoid reading every report in `reports/`; use targeted reads.

## Reusable Standard

For applying this context engineering pattern to other projects, use `docs/agent_context_standard/README.md` and the templates under `docs/agent_context_standard/templates/`.
