# ADR 0005: Repo-Native Agent Context

Status: Accepted
Date: 2026-05-01

## Context

Long-running agent sessions generate large chat context and many tool outputs. A future agent should not need the original chat transcript to understand current state, constraints, decisions, and next actions.

## Decision

Persistent agent context lives in the repository.

Canonical policy remains in `AGENTS.md`. Current state, indexes, known failures, handoff procedures, and tool-specific adapters live in documentation files:

- `docs/context/**`
- `docs/adr/**`
- `docs/runbooks/**`
- thin adapter files such as `CLAUDE.md` and `.github/copilot-instructions.md`

Tool adapters must be short pointers to canonical context, not duplicated policy sources.

## Consequences

- New agents start from `docs/context/AGENT_ONBOARDING.md` after `AGENTS.md`.
- Long reports stay in `reports/`, but indexes guide when to read them.
- Chat handoffs are useful but not sufficient; durable state must be written to repo files.
- Context files must not contradict production safety boundaries.

## Links

- `AGENTS.md`
- `docs/context/AGENT_ONBOARDING.md`
- `docs/context/PROJECT_STATE.md`
- `docs/runbooks/session-handoff.md`
