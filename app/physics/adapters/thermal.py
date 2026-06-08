"""Thermal / calorimetry problem adapter.

Solves sensible-heat calorimetry deterministically (AGENTS.md §13.2):

  * Heat to change temperature:  Q = m * c * ΔT
    where m is mass (kg), c is specific heat (J/(kg·K)), and ΔT is the
    temperature change. ΔT is dimensionally identical in °C and K, so a change
    "from 20°C to 80°C" is ΔT = 60 K.

Temperatures with the "°C"/"°F"/"K" notation and the "from X to Y" phrasing are
extracted with conservative regex since "°C" is not a standard parsed unit.
All arithmetic is deterministic Python.
"""

from __future__ import annotations

import re

from app.physics.adapters.base import AdapterSolution
from app.physics.dimensions import dimension_for_unit
from app.physics.equation_graph import EquationGraph, EquationNode, EquationVariable
from app.physics.ir import PhysicsProblemIR
from app.physics.unit_converter import format_best_unit


class ThermalAdapter:
    """Deterministic adapter for sensible-heat calorimetry (Q = m·c·ΔT)."""

    name = "thermal_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Positive score for "heat to raise/change temperature" problems that
        expose a mass, a specific heat, and a resolvable temperature change."""
        low = ir.question.lower()
        if not _asks_heat(low):
            return 0.0
        if _extract_thermal_inputs(ir) is not None:
            return 0.75
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds the Q = m·c·ΔT equation graph."""
        inputs = _extract_thermal_inputs(ir)
        if inputs is None:
            return None
        mass, specific_heat, delta_t = inputs
        graph = EquationGraph(target="heat")
        graph.add_known("mass", mass, dimension_for_unit("kg"), provenance="question")
        graph.add_known("specific_heat", specific_heat, dimension_for_unit("J/(kg·K)"), provenance="question")
        graph.add_known("delta_t", delta_t, dimension_for_unit("K"), provenance="temperature change")
        graph.variables["heat"] = EquationVariable("heat", dimension_for_unit("J"))
        graph.add_equation(
            EquationNode(
                id="calorimetry_sensible_heat",
                expression="heat = mass * specific_heat * delta_t",
                output="heat",
                inputs=("mass", "specific_heat", "delta_t"),
                output_dimension=dimension_for_unit("J"),
                compute=lambda values: values["mass"] * values["specific_heat"] * values["delta_t"],
            )
        )
        return graph

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the calorimetry problem using the equation graph."""
        graph = self.build_equation_graph(ir)
        if graph is None:
            return None
        solved = graph.solve_forward()
        if solved is None:
            return None
        value, trace = solved
        answer = format_best_unit(value, "J")
        return AdapterSolution(
            answer=answer,
            explanation=(
                "Solved a calorimetry equation graph for the heat required using "
                "Q = m·c·ΔT with backend arithmetic."
            ),
            formula_id="thermal_sensible_heat",
            variables={
                name: variable.value
                for name, variable in graph.variables.items()
                if variable.value is not None
            }
            | {"heat": value},
            cot=trace + [f"heat = {value:.6g} J"],
            confidence=0.9,
            trace={"equation_graph_target": graph.target, "equation_count": len(graph.equations)},
        )


def _asks_heat(low: str) -> bool:
    """Checks if the question asks for heat to change temperature."""
    if "heat" not in low:
        return False
    return any(
        token in low
        for token in ["temperature", "raise", "warm", "heat up", "cool", "from", "degree", "°"]
    )


def _extract_temperatures(question: str) -> list[float]:
    """Extract temperature magnitudes from "20°C", "80 °C", "300 K", "from 20 to 80".

    Returns the numeric values in their stated scale. Since calorimetry uses a
    temperature *difference*, and a change of 1°C equals a change of 1 K, we do
    not convert absolute values; we only difference them.
    """
    temps: list[float] = []
    # Explicit temperature tokens with a degree/scale marker.
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(?:°\s*[cf]|degrees?\s*(?:c|f|celsius|fahrenheit)?|k(?:elvin)?)\b", question, re.I):
        try:
            temps.append(float(m.group(1)))
        except ValueError:
            continue
    return temps


def _extract_delta_t(question: str) -> float | None:
    """Resolve the temperature change ΔT from the question.

    Handles:
      * "from X to Y"  → |Y - X|
      * an explicit "by N degrees" change
      * two bare temperature readings → their difference
    """
    low = question
    # "from X ... to Y" with optional degree markers.
    m = re.search(
        r"from\s+([0-9]+(?:\.[0-9]+)?)\s*(?:°\s*[cf]|degrees?\s*(?:c|f|celsius|fahrenheit)?|k)?\s*"
        r"to\s+([0-9]+(?:\.[0-9]+)?)",
        low,
        re.I,
    )
    if m:
        try:
            t1 = float(m.group(1))
            t2 = float(m.group(2))
            return abs(t2 - t1)
        except ValueError:
            pass
    # "by N degrees" explicit change.
    m = re.search(r"by\s+([0-9]+(?:\.[0-9]+)?)\s*(?:°|degrees?|k)\b", low, re.I)
    if m:
        try:
            return abs(float(m.group(1)))
        except ValueError:
            pass
    # Two temperature readings → difference.
    temps = _extract_temperatures(question)
    if len(temps) >= 2:
        return abs(temps[1] - temps[0])
    return None


def _extract_thermal_inputs(ir: PhysicsProblemIR) -> tuple[float, float, float] | None:
    """Resolve (mass_kg, specific_heat, delta_t) for Q = m·c·ΔT."""
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    specific_heats = [q.value for q in ir.quantities if q.si_unit == "J/(kg·K)"]
    if not masses or not specific_heats:
        return None
    delta_t = _extract_delta_t(ir.question)
    if delta_t is None or delta_t <= 0:
        return None
    return masses[0], specific_heats[0], delta_t
