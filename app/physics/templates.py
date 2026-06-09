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
    elif formula.formula_id == "electric_field_semicircular_arc_center":
        parts.append("Geometry: uniformly charged semicircular arc at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_circular_arc_center":
        parts.append("Geometry: uniformly charged circular arc at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                    f"theta={variables['theta_rad']:.6g} rad",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_finite_line_perpendicular_bisector":
        parts.append("Geometry: finite straight rod on the perpendicular bisector.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"d={format_si(variables['d'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_finite_line_axis_outside_center":
        parts.append("Geometry: finite straight rod on its axis, outside the rod and measured from the center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"x={format_si(variables['x'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_finite_line_axis_outside_end":
        parts.append("Geometry: finite straight rod on its axis, measured from one end outside the rod.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"d={format_si(variables['d'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_symmetric_loop_center_zero":
        parts.append("Geometry: symmetric closed loop at its center, so opposite contributions cancel by symmetry.")
    elif formula.formula_id == "electric_potential_uniform_ring_center":
        parts.append("Geometry: uniformly charged ring at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_square_loop_center":
        parts.append("Geometry: uniformly charged square wire loop at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"a={format_si(variables['a'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_uniform_ring_axis":
        parts.append("Geometry: uniformly charged ring on its axis.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                    f"x={format_si(variables['x'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_uniform_disk_axis":
        parts.append("Geometry: uniformly charged disk on its axis.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"z={format_si(variables['z'], 'm')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_finite_line_perpendicular_bisector":
        parts.append("Geometry: finite straight rod potential at a point on the perpendicular bisector.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"d={format_si(variables['d'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_finite_line_axis_outside_center":
        parts.append("Geometry: finite straight rod potential on its axis outside the rod, measured from the center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"x={format_si(variables['x'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_finite_line_axis_outside_end":
        parts.append("Geometry: finite straight rod potential on its axis outside the rod, measured from one end.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"d={format_si(variables['d'], 'm')}",
                    f"L={format_si(variables['L'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_uniform_sphere_inside":
        parts.append("Geometry: uniformly charged solid sphere, inside region.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"r={format_si(variables['r'], 'm')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_field_uniform_sphere_outside":
        parts.append("Geometry: uniformly charged solid sphere, outside region.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"r={format_si(variables['r'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_uniform_sphere_shell_inside":
        parts.append("Geometry: inside a uniformly charged thin spherical shell.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_uniform_sphere_shell_outside":
        parts.append("Geometry: outside a uniformly charged thin spherical shell.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"r={format_si(variables['r'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "magnetic_field_circular_loop_center":
        parts.append("Geometry: circular current loop at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"I={format_si(variables['I'], 'A')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "spherical_capacitor_capacitance":
        parts.append("Geometry: spherical capacitor.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"a={format_si(variables['a'], 'm')}",
                    f"b={format_si(variables['b'], 'm')}",
                ]
            )
            + "."
        )
    elif formula.formula_id == "electric_potential_circular_arc_center":
        parts.append("Geometry: uniformly charged circular arc at its center.")
        parts.append(
            "Extracted SI variables: "
            + ", ".join(
                [
                    f"q={format_si(variables['q'], 'C')}",
                    f"R={format_si(variables['R'], 'm')}",
                ]
            )
            + "."
        )
    else:
        vals = ", ".join(f"{key}={value:.6g}" for key, value in variables.items())
        parts.append(f"Extracted SI variables: {vals}.")
    parts.append(f"Python computed the result as {format_si(answer, formula.target_unit)}.")
    return " ".join(parts)
