Physics module — core vs experimental

Core files (used by default):
- solver.py: deterministic-first physics pipeline (entry point)
- formulas.py: formula registry and compute functions
- formula_registry.py: qualitative lookup & helpers
- parser.py: extraction of variables and units from text
- unit_converter.py: parse/format SI quantities
- templates.py: explanation templates

Experimental / optional (loaded conditionally):
- v2/: structured v2 architecture with scenarios and qgraph (used when available)
- llm_extractor.py / llm_extract_ollama.py: LLM-based extraction helpers (Ollama-specific helper exists)
- llm_reasoning.py: LLM self-reasoning fallback for qualitative questions
- rag_solver.py: retrieval-augmented routines for qualitative answers
- search_solver.py / search_assisted_solver.py: search + LLM helpers
- smart_extractor.py: geometry/structured extractor
- multi_charge_solver.py: multi-charge specific solver
- midpoint_solver.py: regex-based midpoint pattern solver

Notes:
- Experimental modules are imported inside `try` blocks and are **disabled by default** unless their dependencies (transformers, Ollama, search backends, etc.) are present.
- If you want a cleanup action, I can either move these experimental files into `app/physics/experimental/` or add stubs. Which do you prefer?
