"""Graph-based equation solver for physics.

This module represents variables and equations as a dependency graph
and provides forward-propagation solvers to find target values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.physics.dimensions import Dimension, dimensions_compatible


@dataclass(frozen=True)
class EquationVariable:
    """Represents a variable in the equation graph.

    Attributes:
        name: Name of the variable.
        dimension: Dimensional analysis vector.
        value: Numeric value if known.
        provenance: Description of where the value came from.
    """
    name: str
    dimension: Dimension | None = None
    value: float | None = None
    provenance: str | None = None


@dataclass(frozen=True)
class EquationNode:
    """Represents an equation or formula node in the graph.

    Attributes:
        id: Unique identifier for the equation.
        expression: Text representation of the formula.
        output: Variable name of the output of this equation.
        inputs: Variable names of the inputs needed for the equation.
        output_dimension: Dimensional analysis vector of the output.
        compute: Function that computes the output given input values.
        evidence: Context or citation for this equation.
    """
    id: str
    expression: str
    output: str
    inputs: tuple[str, ...]
    output_dimension: Dimension | None = None
    compute: Callable[[dict[str, float]], float] | None = None
    evidence: str | None = None


@dataclass
class EquationGraph:
    """A graph of variables and equation nodes that propagates known values.

    Attributes:
        variables: Dict mapping variable names to their EquationVariable state.
        equations: List of EquationNode equations in the graph.
        target: The variable name we want to solve for.
        trace: Running audit log of the solving process.
    """
    variables: dict[str, EquationVariable] = field(default_factory=dict)
    equations: list[EquationNode] = field(default_factory=list)
    target: str | None = None
    trace: list[str] = field(default_factory=list)

    def add_known(self, name: str, value: float, dimension: Dimension | None = None, provenance: str | None = None) -> None:
        """Adds a known variable value to the graph.

        Args:
            name: Name of the variable.
            value: Numeric value of the variable.
            dimension: Dimensional vector.
            provenance: Description of where the value came from.
        """
        self.variables[name] = EquationVariable(name=name, value=value, dimension=dimension, provenance=provenance)

    def add_equation(self, equation: EquationNode) -> None:
        """Adds an equation node to the graph.

        Args:
            equation: The EquationNode to add.
        """
        self.equations.append(equation)

    def solve_forward(self) -> tuple[float, list[str]] | None:
        """Runs forward propagation to compute the target variable.

        Repeatedly applies equations whose inputs are fully known until
        the target is solved or no more equations can be applied.

        Returns:
            A tuple of (solved_value, trace_messages) if solved, or None.
        """
        if self.target is None:
            return None
        known = {name: var.value for name, var in self.variables.items() if var.value is not None}
        trace = list(self.trace)
        changed = True
        while changed:
            changed = False
            if self.target in known:
                return float(known[self.target]), trace
            for equation in self.equations:
                if equation.output in known or equation.compute is None:
                    continue
                if not all(name in known for name in equation.inputs):
                    continue
                output_var = self.variables.get(equation.output)
                if output_var and not dimensions_compatible(output_var.dimension, equation.output_dimension):
                    trace.append(f"Rejected {equation.id}: output dimension mismatch")
                    continue
                known[equation.output] = equation.compute({name: float(known[name]) for name in equation.inputs})
                trace.append(f"Applied {equation.id}: {equation.expression}")
                changed = True
        return None

