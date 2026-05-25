Logic module

Core runtime files:
- `solver.py`: main logic solver and orchestration
- `premise_selector.py`: premise normalization and selection
- `proof_trace.py`: proof step objects and validators
- `templates.py`: explanation templates
- `policy_patterns.py`, `policy_reasoner.py`, `thresholds.py`: academic policy helpers

Optional runtime paths:
- `enable_hybrid_solver`: external hybrid path with Z3 + local model, guarded by config
- `enable_z3_sidecar`: optional Z3-sidecar experiment path
- `enable_mcq_symbolic`: symbolic MCQ helper path

The module is intentionally split by responsibility; keep new logic helpers in focused files rather than extending `solver.py` further.
