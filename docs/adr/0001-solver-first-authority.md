# ADR 0001: Solver-First Authority

Status: Accepted
Date: 2026-05-01

## Context

The project must answer educational QA tasks with structured JSON containing at least `answer` and `explanation`. The system handles deterministic physics problems and premise-bound logic problems. Raw model output can be malformed, unsupported, or numerically wrong.

## Decision

The backend is the final answer authority.

- Physics arithmetic is computed by deterministic Python formulas.
- Logic answers are derived from supplied premises and validated premise IDs.
- Final JSON is assembled by backend code and validated with Pydantic `QAResponse`.
- Model output may assist extraction, proposal, or explanation experiments but is not final authority.

## Consequences

- Do not submit raw model output directly.
- Do not let an LLM perform final arithmetic when deterministic formulas apply.
- Preserve final response validation and premise ID validation.
- Any future model integration must pass through backend normalization and validation.

## Links

- `AGENTS.md`
- `docs/system_architecture.md`
- `app/schemas.py`
- `app/router.py`
