# ADR 0003: Sidecars Remain Experiment-Only

Status: Accepted
Date: 2026-05-01

## Context

Z3 and Datalog sidecars can provide high-precision verification for some logic problems. Phase 23 hardened Z3 to zero wrong-when-accepted on the tested policy, and Phase 24 showed Datalog could match Z3 on a hard subset. Coverage remained limited, and these paths introduce operational and validation complexity.

## Decision

Z3 and Datalog sidecars remain offline/eval experiment-only and are disabled by production default.

They must not become production answer authority without a new explicit approval and a documented validation phase.

## Consequences

- `enable_z3_sidecar` remains `false` by default.
- Datalog is not part of the production `/predict` authority path by default.
- Sidecar outputs may be used in reports, traces, and experiments.
- Future promotion requires larger validation, regression gates, and explicit user approval.

## Links

- `reports/phase_23_z3_precision_hardening_report.md`
- `reports/phase_24_alt_approach_shootout_report.md`
- `reports/phase_29_symbolic_direction_recommendation.md`
- `configs/pipeline.yaml`
