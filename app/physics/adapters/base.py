"""Base interface and schemas for physics domain adapters.

This module defines the AdapterSolution schema and the PhysicsAdapter protocol,
which specialized adapters must implement to solve domain-specific physics
problems (e.g. circuits, mechanics, electrostatics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.physics.equation_graph import EquationGraph
from app.physics.ir import PhysicsProblemIR


@dataclass(frozen=True)
class AdapterSolution:
    """Represents a solution returned by a domain adapter.

    Attributes:
        answer: The solved value or text.
        explanation: Natural language explanation of the solution.
        formula_id: ID of the formula or method used.
        variables: Variables used in the computation.
        cot: Chain-of-thought calculation steps.
        confidence: Confidence score of the solution.
        trace: Full trace metadata of the solution process.
    """
    answer: str
    explanation: str
    formula_id: str
    variables: dict[str, float] = field(default_factory=dict)
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: dict[str, object] = field(default_factory=dict)


class PhysicsAdapter(Protocol):
    """Protocol defining the interface for domain-specific physics adapters.

    Attributes:
        name: Name of the adapter.
    """
    name: str

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines how confidently this adapter can solve the given problem.

        Args:
            ir: The intermediate representation of the problem.

        Returns:
            A confidence score between 0.0 (cannot handle) and 1.0 (perfect fit).
        """
        ...

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph from the problem parameters.

        Args:
            ir: The intermediate representation of the problem.

        Returns:
            An EquationGraph instance, or None if the problem is invalid/incomplete.
        """
        ...

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the physics problem using domain-specific rules/algorithms.

        Args:
            ir: The intermediate representation of the problem.

        Returns:
            The AdapterSolution if successful, or None if the problem cannot be solved.
        """
        ...

