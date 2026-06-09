"""RLC Circuit solver for phasor and impedance math.

This module provides data models and solver functions for series RLC
circuits, handling impedance, resonant current, and frequency scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesRLCPhasorIR:
    """Small circuit IR for series-RLC phasor calculations.

    This is the expansion point toward SPICE/MNA-style circuit solving: parsing
    produces labeled circuit quantities, and this module owns impedance math.

    Attributes:
        resistance: Resistance value in ohms.
        inductive_reactance: Inductive reactance in ohms.
        capacitive_reactance: Capacitive reactance in ohms.
        voltage: Voltage in volts.
        frequency_scale: Frequency scale multiplier.
        resonance_current: Resonant current in amperes.
        scaled_current: Scaled current in amperes.
    """

    resistance: float
    inductive_reactance: float | None = None
    capacitive_reactance: float | None = None
    voltage: float | None = None
    frequency_scale: float | None = None
    resonance_current: float | None = None
    scaled_current: float | None = None


@dataclass(frozen=True)
class CircuitResult:
    """Represents the output result of a circuit calculation.

    Attributes:
        value: Numeric result of the computation.
        unit: Unit of the computed value.
        formula_id: ID of the formula used.
        variables: Variables used in the computation.
        explanation: Natural language explanation of the computation.
        cot: Chain-of-thought calculation steps.
    """

    value: float
    unit: str
    formula_id: str
    variables: dict[str, float]
    explanation: str
    cot: list[str]


def series_rlc_current_after_frequency_scale(ir: SeriesRLCPhasorIR) -> CircuitResult | None:
    """Computes the new current in a series RLC circuit after frequency scaling.

    Args:
        ir: The RLC circuit state.

    Returns:
        The CircuitResult if parameters are sufficient, or None.
    """
    if (
        ir.voltage is None
        or ir.inductive_reactance is None
        or ir.capacitive_reactance is None
        or ir.frequency_scale is None
    ):
        return None
    x_l_new = ir.frequency_scale * ir.inductive_reactance
    x_c_new = ir.capacitive_reactance / ir.frequency_scale
    impedance = math.hypot(ir.resistance, x_l_new - x_c_new)
    current = ir.voltage / impedance
    return CircuitResult(
        value=current,
        unit="A",
        formula_id="series_rlc_phasor_current_scaled_frequency",
        variables={
            "V": ir.voltage,
            "R": ir.resistance,
            "X_L": ir.inductive_reactance,
            "X_C": ir.capacitive_reactance,
            "scale": ir.frequency_scale,
            "Z": impedance,
        },
        explanation=(
            "Used series-RLC phasor impedance Z = sqrt(R^2 + (X_L - X_C)^2), "
            "scaling X_L with frequency and X_C inversely."
        ),
        cot=[f"X_L'={x_l_new:.6g} ohm, X_C'={x_c_new:.6g} ohm", f"Z={impedance:.6g} ohm", f"I={current:.6g} A"],
    )


def series_rlc_inductive_reactance_from_scaled_current(ir: SeriesRLCPhasorIR) -> CircuitResult | None:
    """Derives inductive reactance at resonance from scaled current.

    Args:
        ir: The RLC circuit state.

    Returns:
        The CircuitResult containing the resonance reactance, or None.
    """
    if ir.frequency_scale is None or ir.resonance_current is None or ir.scaled_current is None:
        return None
    denominator = abs(ir.frequency_scale - (1.0 / ir.frequency_scale))
    if denominator <= 0:
        return None
    source_voltage = ir.resonance_current * ir.resistance
    scaled_impedance = source_voltage / ir.scaled_current
    reactive_difference = math.sqrt(max(0.0, scaled_impedance * scaled_impedance - ir.resistance * ir.resistance))
    x_l0 = reactive_difference / denominator
    return CircuitResult(
        value=x_l0,
        unit="ohm",
        formula_id="series_rlc_phasor_reactance_from_scaled_current",
        variables={
            "R": ir.resistance,
            "I_res": ir.resonance_current,
            "I_scaled": ir.scaled_current,
            "scale": ir.frequency_scale,
            "Z_scaled": scaled_impedance,
            "X_L0": x_l0,
        },
        explanation=(
            "Used resonance phasor relations: at resonance V=I0*R, and after frequency scaling "
            "|X_L-X_C|=(m-1/m)X_L0."
        ),
        cot=[f"V={source_voltage:.6g} V", f"Z_scaled={scaled_impedance:.6g} ohm", f"X_L0={x_l0:.6g} ohm"],
    )

