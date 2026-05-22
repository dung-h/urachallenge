from __future__ import annotations

from app.physics.formulas import Formula
from app.physics.unit_converter import format_si


def physics_explanation(formula: Formula, variables: dict[str, float], answer: float) -> str:
    parts = [f"Used {formula.expression}."]
    if "resistances" in variables:
        vals = ", ".join(format_si(v, "ohm") for v in variables["resistances"])
        parts.append(f"The resistance values are {vals}.")
    elif "capacitances" in variables:
        vals = ", ".join(format_si(v, "F") for v in variables["capacitances"])
        parts.append(f"The capacitance values are {vals}.")
    elif "A" in variables and "d" in variables and "eps" in variables:
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"A={format_si(variables['A'], 'm^2')}",
                    f"d={format_si(variables['d'], 'm')}",
                    f"eps={variables['eps']:.6g}",
                ]
            )
            + "."
        )
    elif {"V_primary", "N_primary", "N_secondary"}.issubset(variables):
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"V_primary={format_si(variables['V_primary'], 'V')}",
                    f"N_primary={variables['N_primary']:.6g}",
                    f"N_secondary={variables['N_secondary']:.6g}",
                ]
            )
            + "."
        )
    else:
        vals = ", ".join(f"{key}={value:.6g}" for key, value in variables.items())
        parts.append(f"Extracted SI variables: {vals}.")
    parts.append(f"Python computed the result as {format_si(answer, formula.target_unit)}.")
    return " ".join(parts)
