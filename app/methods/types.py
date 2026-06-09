"""Core types for the method-centric reasoning architecture.

Defines the uniform contract every Method (whether it wraps a physics adapter,
a Z3-backed logic clause set, an LLM-grounded retrieved formula, or a freshly
discovered procedure) must implement.

Design notes
------------
* **Methods are stateless reasoners.** State (success rate, last used) lives
  in the ``MethodLibrary`` so the same ``Method`` instance can be shared
  across requests safely.
* **Methods never finalize the answer.** They produce a ``MethodResult`` with
  trace + confidence; the planner / backend decide acceptance.
* **Methods declare their applicability.** ``can_handle`` + ``score_match``
  let the planner pick without LLM-prompt wallclock.
* **Methods carry provenance.** ``source`` and ``signature`` make the library
  auditable and let MethodDiscovery (Level 6) deduplicate retrieved methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class MethodFamily(str, Enum):
    """Coarse categorization for routing and analytics.

    Kept small on purpose: families let the planner reason at a high level
    ("try a logic-symbolic method first") without enumerating every concrete
    method. Add a new family only when a genuinely new reasoning shape arrives
    (e.g. probabilistic, planning-based).
    """

    LOGIC_SYMBOLIC = "logic_symbolic"        # FOL+Z3, BFS, rule-based
    LOGIC_RETRIEVAL = "logic_retrieval"      # Looked-up logic patterns
    PHYSICS_FORMULA = "physics_formula"      # Hand-coded adapter / formula registry
    PHYSICS_RETRIEVAL = "physics_retrieval"  # retrieval_grounded_method
    PHYSICS_NUMERIC = "physics_numeric"      # SymPy equation graph / IR solver
    META = "meta"                            # planner / abstain / orchestration


class MethodSource(str, Enum):
    """Where a Method came from. Important for trust gates and Level-6 audit."""

    BUILTIN = "builtin"               # Hand-written, in-tree
    DISCOVERED_VERIFIED = "discovered_verified"   # Found by MethodDiscovery, passed validation
    DISCOVERED_PROVISIONAL = "discovered_provisional"  # Found, awaiting more evidence
    USER_REGISTERED = "user_registered"           # Registered via API / config


@dataclass(frozen=True)
class MethodApplicability:
    """How well a Method matches a question.

    Returned by ``Method.score_match``. ``score`` ∈ [0, 1]. ``why`` is a short
    structural reason (NOT the question text — see AGENTS.md §20.1) the planner
    and audits use to explain selection.
    """

    score: float
    why: str
    confident: bool = False  # When True, planner can skip other candidates


@dataclass
class MethodTrace:
    """Audit trail of what a Method did. Always populated, even on abstain.

    Backend authority requires every step be inspectable:
      * ``inputs_seen``: which premises / quantities were considered.
      * ``inputs_dropped``: which were intentionally NOT used (with reason).
      * ``llm_calls``: count + roles ("translator", "explainer", ...).
      * ``backend_steps``: what Z3 / SymPy / safe_eval actually executed.

    The Coverage gate inspects ``inputs_dropped`` — a method that silently
    ignores a premise without listing why fails the gate.
    """

    method_id: str
    inputs_seen: list[str] = field(default_factory=list)
    inputs_dropped: list[dict[str, str]] = field(default_factory=list)
    llm_calls: int = 0
    llm_roles: list[str] = field(default_factory=list)
    backend_steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def drop(self, item_id: str, reason: str) -> None:
        """Record an input that the method intentionally did not use."""
        self.inputs_dropped.append({"id": item_id, "reason": reason})

    def step(self, message: str) -> None:
        """Record a backend reasoning step."""
        self.backend_steps.append(message)

    def note(self, message: str) -> None:
        """Record a free-form observation."""
        self.notes.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "inputs_seen": list(self.inputs_seen),
            "inputs_dropped": list(self.inputs_dropped),
            "llm_calls": int(self.llm_calls),
            "llm_roles": list(self.llm_roles),
            "backend_steps": list(self.backend_steps),
            "notes": list(self.notes),
            "elapsed_ms": float(self.elapsed_ms),
        }


@dataclass
class MethodResult:
    """The output every Method produces.

    ``answer`` may be ``None`` when the method abstains; in that case the
    planner tries the next method. ``confidence`` MUST be computed from
    backend signals (AGENTS.md §16) — never asked of the LLM.
    """

    method_id: str
    family: MethodFamily
    answer: str | None
    explanation: str
    confidence: float
    trace: MethodTrace
    # Optional structured payloads — populated when relevant:
    used_premise_ids: list[str] = field(default_factory=list)
    formula_id: str | None = None
    numeric_value: float | None = None
    numeric_unit: str | None = None
    z3_status: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    error: str | None = None
    # When True, the method has its OWN backend witness (e.g. SymPy
    # backwards-substitution into every equation, or a Z3 proof) that the
    # answer is self-consistent. The planner uses this to weight the
    # method-vs-method consistency vote: a primary with a backend witness
    # outranks a verifier that has no witness of its own.
    backend_verified: bool = False

    @property
    def decisive(self) -> bool:
        """A decisive result has an answer and was not an abstain/error."""
        return (
            self.answer is not None
            and not self.abstained
            and self.error is None
            and str(self.answer).strip().lower() != "unknown"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "family": self.family.value,
            "answer": self.answer,
            "explanation": self.explanation,
            "confidence": float(self.confidence),
            "used_premise_ids": list(self.used_premise_ids),
            "formula_id": self.formula_id,
            "numeric_value": self.numeric_value,
            "numeric_unit": self.numeric_unit,
            "z3_status": self.z3_status,
            "abstained": bool(self.abstained),
            "abstain_reason": self.abstain_reason,
            "error": self.error,
            "trace": self.trace.to_dict(),
        }


@runtime_checkable
class Method(Protocol):
    """Uniform reasoner contract.

    A Method is anything that can:
      1. Decide whether it applies to a given problem (``score_match``).
      2. Produce a traced, scored result (``solve``).
      3. Identify itself uniquely so the library can deduplicate (``signature``).

    It is intentionally minimal so wrapping an existing ``PhysicsAdapter``,
    a logic ``solve_fol_z3`` call, or a freshly retrieved Wikipedia-derived
    procedure all fit. Concrete implementations live in ``app/methods/impl/``.
    """

    method_id: str
    family: MethodFamily
    source: MethodSource

    def signature(self) -> str:
        """A stable hash / structural fingerprint used for deduplication.

        For built-in methods this is typically the method_id; for discovered
        methods it is a hash of (formula expression, target unit, variables,
        applicability heuristic) so two different LLM extractions of the same
        underlying procedure collapse to one library entry.
        """
        ...

    def score_match(self, problem: Any) -> MethodApplicability:
        """How well this method fits the given problem (0..1).

        ``problem`` is the typed input bundle (a ``LogicProblem`` or
        ``PhysicsProblem`` IR). Implementations should NOT inspect raw question
        text strings beyond what is structurally summarized in the IR.
        """
        ...

    def solve(self, problem: Any, *, llm_client: Any | None = None,
              budget: Any | None = None) -> MethodResult:
        """Run the method on the problem.

        ``llm_client`` is optional; methods that need it must check and abstain
        gracefully when it is absent. ``budget`` is the request-level
        ``CallBudget`` so the method's LLM calls participate in the global
        cap+deadline.
        """
        ...


# A simple structural helper: sort-key for the planner when ranking applicable
# methods. Built-ins win ties over discovered, then by score, then by family
# preference order. This keeps method choice deterministic given the same set
# of applicable methods.
_FAMILY_PREFERENCE: list[MethodFamily] = [
    MethodFamily.LOGIC_SYMBOLIC,
    MethodFamily.PHYSICS_FORMULA,
    MethodFamily.PHYSICS_NUMERIC,
    MethodFamily.LOGIC_RETRIEVAL,
    MethodFamily.PHYSICS_RETRIEVAL,
    MethodFamily.META,
]


def planner_sort_key(method: Method, applicability: MethodApplicability) -> tuple:
    """Stable ordering key for the planner.

    Order: applicability score (desc), source preference (builtin first),
    family preference, method_id for tiebreak.
    """
    source_rank = {
        MethodSource.BUILTIN: 0,
        MethodSource.USER_REGISTERED: 1,
        MethodSource.DISCOVERED_VERIFIED: 2,
        MethodSource.DISCOVERED_PROVISIONAL: 3,
    }.get(method.source, 4)
    try:
        family_rank = _FAMILY_PREFERENCE.index(method.family)
    except ValueError:
        family_rank = len(_FAMILY_PREFERENCE)
    return (
        -float(applicability.score),
        source_rank,
        family_rank,
        str(method.method_id),
    )
