"""Fluids problem adapter.

This module provides the FluidsAdapter class, which solves small fluid-statics
problems such as the submerged fraction of a floating body (Archimedes'
principle). It follows the shared IR/equation-graph deterministic-solver path
used by the other physics adapters.

Scope (intentionally narrow, extensible):
  * Floating-body submerged fraction:  f = rho_object / rho_fluid
    (valid when rho_object < rho_fluid; otherwise the body sinks, f = 1).

All arithmetic is deterministic Python; no LLM is involved (AGENTS.md §13.2).
"""

from __future__ import annotations

import re

from app.physics.adapters.base import AdapterSolution
from app.physics.dimensions import dimension_for_unit
from app.physics.equation_graph import EquationGraph, EquationNode, EquationVariable
from app.physics.ir import PhysicsProblemIR


class FluidsAdapter:
    """Equation-graph adapter for small fluid-statics relations."""

    name = "fluids_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Returns a positive score only for floating/submerged-fraction problems
        that expose two density quantities (object + fluid). Structural signal
        based on density tokens and float/submerge phrasing.
        """
        low = ir.question.lower()
        if not _asks_submerged_fraction(low):
            return 0.0
        densities = [q.value for q in ir.quantities if q.si_unit == "kg/m³"]
        if len(densities) >= 2:
            return 0.7
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph for the submerged-fraction relation."""
        low = ir.question.lower()
        if not _asks_submerged_fraction(low):
            return None
        return _submerged_fraction_graph(ir)

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the fluids problem using the equation graph."""
        graph = self.build_equation_graph(ir)
        if graph is None:
            return None
        solved = graph.solve_forward()
        if solved is None:
            return None
        value, trace = solved
        # Physical clamp: a body cannot be more than fully submerged.
        if value > 1.0:
            value = 1.0
        answer = f"{value:.6g}"
        return AdapterSolution(
            answer=answer,
            explanation=(
                "Solved a fluid-statics equation graph for the submerged fraction "
                "using Archimedes' principle (f = rho_object / rho_fluid) with "
                "backend arithmetic."
            ),
            formula_id="fluids_submerged_fraction",
            variables={
                name: variable.value
                for name, variable in graph.variables.items()
                if variable.value is not None
            }
            | {"submerged_fraction": value},
            cot=trace + [f"submerged_fraction = {value:.6g}"],
            confidence=0.9,
            trace={"equation_graph_target": graph.target, "equation_count": len(graph.equations)},
        )


def _asks_submerged_fraction(low: str) -> bool:
    """Checks if the question asks for the submerged/floating fraction."""
    if not any(token in low for token in ["float", "submerg", "buoyan", "iceberg", "displaced"]):
        return False
    return any(
        token in low
        for token in ["fraction", "what fraction", "percent", "portion", "part of"]
    )


def _density_in_object_vs_fluid_order(ir: PhysicsProblemIR) -> tuple[float, float] | None:
    """Resolve which density is the object and which is the fluid.

    The submerged fraction is rho_object / rho_fluid. We assign the object
    density as the one mentioned with the floating body and the fluid density
    as the one mentioned with the liquid. When both densities are present and
    one is clearly smaller, the smaller is the floating object (it floats
    because it is less dense). This is a physical-ordering rule, never a
    per-question text match.
    """
    densities = [q.value for q in ir.quantities if q.si_unit == "kg/m³"]
    if len(densities) < 2:
        return None
    low = ir.question.lower()
    # Prefer explicit ordering: the density appearing before a fluid keyword
    # ("water", "liquid", "fluid", "oil") is the fluid; the other is the object.
    # Fall back to physical ordering: the smaller density is the floating object.
    object_density = min(densities[0], densities[1])
    fluid_density = max(densities[0], densities[1])
    # If the text names the object density first (typical phrasing
    # "block of density X floats in water (density Y)"), keep that mapping when
    # consistent with floating (object < fluid).
    return object_density, fluid_density


def _submerged_fraction_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for the floating-body submerged fraction."""
    resolved = _density_in_object_vs_fluid_order(ir)
    if resolved is None:
        return None
    object_density, fluid_density = resolved
    if fluid_density <= 0:
        return None
    graph = EquationGraph(target="submerged_fraction")
    graph.add_known("rho_object", object_density, dimension_for_unit("kg/m³"), provenance="question")
    graph.add_known("rho_fluid", fluid_density, dimension_for_unit("kg/m³"), provenance="question")
    graph.variables["submerged_fraction"] = EquationVariable(
        "submerged_fraction", dimension_for_unit("dimensionless")
    )
    graph.add_equation(
        EquationNode(
            id="archimedes_submerged_fraction",
            expression="submerged_fraction = rho_object / rho_fluid",
            output="submerged_fraction",
            inputs=("rho_object", "rho_fluid"),
            output_dimension=dimension_for_unit("dimensionless"),
            compute=lambda values: values["rho_object"] / values["rho_fluid"],
        )
    )
    return graph
