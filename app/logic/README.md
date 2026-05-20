Logic module — core vs experimental

Core files (used by default):
- solver.py: main logic solver and orchestration
- premise_selector.py: normalize/select premises
- proof_trace.py: proof step objects and validators
- templates.py: explanation templates
- thresholds.py / policy_patterns.py / policy_reasoner.py: policy-specific helpers

References to optional/missing components:
- z3_sidecar: optional external Z3 sidecar (enabled via pipeline config);
- fol_translator: translator to FOL (used by some hybrid flows);
- mcq_symbolic: symbolic MCQ solver helper.

Notes:
- `z3_sidecar`, `fol_translator`, and `mcq_symbolic` may not be present in the repo; imports are guarded. If you want a strict cleanup, I can either:
  1) Move experimental files into `app/logic/experimental/` and keep only core files, or
  2) Add `app/logic/EXPERIMENTAL.md` documenting missing dependencies and how to enable them.
Which option do you want?
