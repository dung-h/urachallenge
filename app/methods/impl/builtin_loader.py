"""Register every built-in Method into a ``MethodLibrary``.

Called once when the default library is built (``get_default_library``).
Built-ins are reloaded from code on every process start so a code change
is the single source of truth for them.

Registration policy (post-Phase F3 measurement)
-----------------------------------------------
Phase F3's full eval found that decomposing logic / physics into many
small methods regressed -6 vs the legacy fixed pipeline because the small
methods miss post-processing gates the legacy ``solve_logic`` /
``solve_physics`` already enforce. We therefore wrap the entire legacy
pipelines as single Methods (``LegacyLogicMethod`` / ``LegacyPhysicsMethod``)
and only ALSO register the structural shortcuts that produce a strict
WIN over legacy on a measured slice:

  * ``PhysicsQualitativeMethod``  — 0.1 s vs 15 s on qualitative cases.
  * ``LogicPatternRewriteMethod``  — turns "X unless Y" into "if not Y, then X"
                                     before the LLM translator sees it.
  * ``DiscoveredPhysicsMethod``s   — runtime Level-6 discoveries; gated by
                                     ``DiscoveredPhysicsMethod.score_match``
                                     to require ``quantity_count > 0``.

Hand-coded ``PhysicsAdapterMethod`` wrappers and the small
``LogicFolZ3Method`` / ``LogicBfsMethod`` standalone methods are NOT
registered — their behaviors are subsumed by the legacy pipeline wrap.
"""

from __future__ import annotations

from app.methods.library import MethodLibrary
from app.methods.impl.legacy_solve_methods import (
    LegacyLogicMethod,
    LegacyPhysicsMethod,
)
from app.methods.impl.logic_patterns_method import LogicPatternRewriteMethod
from app.methods.impl.physics_equation_graph import PhysicsEquationGraphMethod
from app.methods.impl.physics_qualitative import PhysicsQualitativeMethod
from app.methods.impl.physics_retrieval import (
    PhysicsConceptualLookupMethod,
    PhysicsRetrievalMethod,
)


def register_builtins(library: MethodLibrary) -> None:
    """Register every in-tree Method into ``library``.

    Order is documentary only — the planner sorts methods by score, not by
    registration order.
    """
    # Logic: pattern rewrite (high score when it fires) → legacy pipeline.
    library.register(LogicPatternRewriteMethod())
    library.register(LegacyLogicMethod())

    # Physics: qualitative shortcut → equation-graph (Level 4) → legacy.
    # Equation-graph score peaks at 0.85 when target_quantity AND ≥3 quantities
    # are present, so it outranks the legacy pipeline (0.40) on multi-knowns
    # numeric problems while leaving 1-2-quantity problems to the legacy path
    # that has more domain-specific heuristics.
    library.register(PhysicsQualitativeMethod())
    library.register(PhysicsEquationGraphMethod())
    library.register(LegacyPhysicsMethod())
    library.register(PhysicsConceptualLookupMethod())
    library.register(PhysicsRetrievalMethod())
