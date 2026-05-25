Physics module — core vs experimental

Core files (used by default):
- solver.py: deterministic-first physics pipeline (entry point)
- formulas.py: formula registry and compute functions
- formula_registry.py: qualitative lookup & helpers
- parser.py: extraction of variables and units from text
- problem_frame.py: lightweight frame inference for search/ranking guardrails
- method_search.py: method/equation evidence retrieval and verification helpers
- expression_eval.py: safe arithmetic evaluator for verified equation proposals
- unit_converter.py: parse/format SI quantities
- templates.py: explanation templates

Runtime boundaries:
- LLM/search workers may propose formulas, expressions, or methods only when opt-in runtime settings allow them.
- Final arithmetic and final JSON remain backend validated.
- LLM-generated Python code execution is not part of the runtime path.

Notes:
- `solver.py` is still intentionally the public entry point, but parser/search/verifier helpers should continue moving into focused modules instead of growing new special cases there.
