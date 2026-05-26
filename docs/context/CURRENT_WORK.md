# Current Work

Last updated: 2026-05-26

## Active Workstream

- **Live vLLM orchestration and runtime cleanup**: the default runtime uses `127.0.0.1:8001`, fails closed with `503` when offline, and only falls back to heuristics when explicitly opted in.
- **Shared agent kernel stability**: physics and logic both use the bounded planner/tool/retry loop. Backend verification remains the answer authority.
- **Context hygiene**: keep `PROJECT_STATE.md` and this file trimmed to the latest verified state. Older details should live in reports or git history.

## Latest Verification

- Live `/predict` smoke via `TestClient` returned `200` for one logic sample and one physics sample.
- The repeatable smoke entrypoint is `scripts/real_smoke_tests.py`.
- Compile/import checks passed after the cache/adapter cleanup.

## Next Recommended Action

- Keep the live smoke path healthy.
- Preserve fail-closed behavior and do not reintroduce dead adapters.
- Only expand these files when a new verified milestone changes runtime behavior.
