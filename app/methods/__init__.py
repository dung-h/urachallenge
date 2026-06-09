"""Method-centric reasoning architecture for the URA EXACT system.

This package introduces a **first-class Method abstraction**: a typed,
inspectable, composable unit of solving capability that wraps the existing
physics adapters and logic verifiers behind a uniform interface.

Architecture layers (AGENTS.md §13, §22):

    Question
      → Planner          (LLM agent: chooses Method(s), decides search,
                          decides abstain) — app/methods/planner.py
      → MethodLibrary    (registry of Method instances, scored by history) —
                          app/methods/library.py
      → Method.solve()   (Method-specific reasoner; LLM may translate, Z3 /
                          SymPy / equation graph decides)
      → Faithfulness +   (round-trip the translation; reject if input was
                          dropped) — app/methods/faithfulness.py
        Coverage gates    (every premise/quantity has a fate) —
                          app/methods/coverage.py
      → MethodDiscovery  (Level 6: when no method fits, search the web,
                          extract a candidate, validate, register) —
                          app/methods/discovery.py
      → Backend assembles & validates final JSON (unchanged authority)

The planner is LLM-driven so the system can do meta-reasoning ("I have method
X for this kind of problem; let me try it; if it abstains, search for a
different method"). The Method library is persistent so a method discovered
on question N is available from question N+1 onward (Level 6 self-extension).

Methods do NOT decide the final answer alone. Z3 / SymPy / safe_eval still
verify; the Method just packages "how to translate this kind of problem and
whom to hand it to". This keeps the AGENTS.md §13 invariant intact: LLM
translates, deterministic backend decides, schemas validate.
"""

from app.methods.types import (
    Method,
    MethodResult,
    MethodTrace,
    MethodFamily,
    MethodApplicability,
    MethodSource,
)
from app.methods.library import MethodLibrary, get_default_library

__all__ = [
    "Method",
    "MethodResult",
    "MethodTrace",
    "MethodFamily",
    "MethodApplicability",
    "MethodSource",
    "MethodLibrary",
    "get_default_library",
]
