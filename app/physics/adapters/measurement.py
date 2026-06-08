"""Measurement problem adapter.

This module provides the MeasurementAdapter class, which solves multi-output
physics tasks such as calculating the average and mean absolute error of
readings, or computing both capacitor energy and charge.
"""

from __future__ import annotations

from app.physics.adapters.base import AdapterSolution
from app.physics.equation_graph import EquationGraph
from app.physics.ir import PhysicsProblemIR
from app.physics.unit_converter import format_best_unit


def _format_number(value: float) -> str:
    """Formats a float to a concise string representation."""
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


class MeasurementAdapter:
    """Adapter for multi-output measurement problems.

    This adapter handles queries requiring multiple output values, such as
    averaging readings with mean absolute error, or evaluating capacitor energy
    and accumulated charge simultaneously.
    """
    name = "measurement_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        low = ir.question.lower()
        if "average" in low and ("absolute error" in low or "mean absolute error" in low):
            return 0.95
        if "capacitor" in low and "energy" in low and _asks_for_charge_output(low):
            return 0.9
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph for the problem.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            None, as this adapter does not use equation graphs.
        """
        # Multi-output measurement programs are currently direct executors. The
        # graph hook exists so later adapters can expose the same trace contract.
        return None

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the multi-output measurement problem.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            An AdapterSolution, or None if the problem cannot be solved.
        """
        low = ir.question.lower()

        if "average" in low and ("absolute error" in low or "mean absolute error" in low):
            grouped: dict[tuple[str, str], list[float]] = {}
            for quantity in ir.quantities:
                grouped.setdefault((quantity.si_unit, quantity.raw.split()[-1] if quantity.raw.split() else quantity.si_unit), []).append(_display_value(quantity.value, quantity.raw))
            if not grouped:
                return None
            values = max(grouped.values(), key=len)
            if len(values) < 2:
                return None
            average = sum(values) / len(values)
            mean_abs_error = sum(abs(value - average) for value in values) / len(values)
            return AdapterSolution(
                answer=f"{_format_number(average)}; {_format_number(mean_abs_error)}",
                explanation="Computed the average of the measurements and then averaged the absolute deviations from that average.",
                formula_id="multi_answer_average_absolute_error",
                variables={"average": average, "mean_absolute_error": mean_abs_error, "n": float(len(values))},
                cot=[f"Measurements: {values}", f"Average = {average:.6g}", f"Mean absolute error = {mean_abs_error:.6g}"],
                confidence=0.95,
                trace={"program": "average_and_mean_absolute_error", "quantity_count": len(values)},
            )

        if "capacitor" in low and "energy" in low and _asks_for_charge_output(low):
            capacitances = [quantity.value for quantity in ir.quantities if quantity.si_unit == "F"]
            voltages = [quantity.value for quantity in ir.quantities if quantity.si_unit == "V"]
            if not capacitances or not voltages:
                return None
            c = capacitances[0]
            v = voltages[0]
            energy = 0.5 * c * v * v
            charge = c * v
            return AdapterSolution(
                answer=f"{format_best_unit(energy, 'J')}; {format_best_unit(charge, 'C')}",
                explanation="Computed capacitor energy with E = 0.5*C*V^2 and charge with Q = C*V.",
                formula_id="multi_answer_capacitor_energy_charge",
                variables={"C": c, "V": v, "E": energy, "Q": charge},
                cot=[f"C={c:.6g} F, V={v:.6g} V", f"E={energy:.6g} J", f"Q={charge:.6g} C"],
                confidence=0.95,
                trace={"program": "capacitor_energy_and_charge"},
            )
        return None


def _display_value(si_value: float, raw: str) -> float:
    """Extracts the display numeric value from raw quantity text, falling back to SI value."""
    parts = raw.split()
    if not parts:
        return si_value
    try:
        return float(parts[0])
    except ValueError:
        return si_value


def _asks_for_charge_output(low: str) -> bool:
    """Checks if the question text asks for both energy and charge output."""
    return any(
        phrase in low
        for phrase in [
            "energy and the charge",
            "energy and charge",
            "charge and the energy",
            "charge and energy",
        ]
    )

