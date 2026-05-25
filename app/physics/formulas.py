from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


K_COULOMB = 8.9875517923e9
EPSILON_0 = 8.8541878128e-12


@dataclass(frozen=True)
class Formula:
    formula_id: str
    expression: str
    target_unit: str
    variables: tuple[str, ...]
    compute: Callable[[dict[str, float]], float]


FORMULAS: dict[str, Formula] = {
    "ohms_law_v_ir": Formula("ohms_law_v_ir", "V = I * R", "V", ("I", "R"), lambda v: v["I"] * v["R"]),
    "ohms_law_i_v_over_r": Formula("ohms_law_i_v_over_r", "I = V / R", "A", ("V", "R"), lambda v: v["V"] / v["R"]),
    "ohms_law_r_v_over_i": Formula("ohms_law_r_v_over_i", "R = V / I", "ohm", ("V", "I"), lambda v: v["V"] / v["I"]),
    "power_p_vi": Formula("power_p_vi", "P = V * I", "W", ("V", "I"), lambda v: v["V"] * v["I"]),
    "power_i_p_over_v": Formula("power_i_p_over_v", "I = P / V", "A", ("P", "V"), lambda v: v["P"] / v["V"]),
    "power_v_p_over_i": Formula("power_v_p_over_i", "V = P / I", "V", ("P", "I"), lambda v: v["P"] / v["I"]),
    "power_p_i2r": Formula("power_p_i2r", "P = I^2 * R", "W", ("I", "R"), lambda v: (v["I"] ** 2) * v["R"]),
    "power_i_sqrt_p_over_r": Formula("power_i_sqrt_p_over_r", "I = sqrt(P / R)", "A", ("P", "R"), lambda v: math.sqrt(v["P"] / v["R"])),
    "power_r_p_over_i2": Formula("power_r_p_over_i2", "R = P / I^2", "ohm", ("P", "I"), lambda v: v["P"] / (v["I"] ** 2)),
    "power_p_v2r": Formula("power_p_v2r", "P = V^2 / R", "W", ("V", "R"), lambda v: (v["V"] ** 2) / v["R"]),
    "rlc_power_vrms": Formula("rlc_power_vrms", "P = V_rms^2 / R", "W", ("V", "R"), lambda v: (v["V"] ** 2) / v["R"]),
    "ac_power_vi_cos_phi": Formula("ac_power_vi_cos_phi", "P = V_rms * I_rms * cos(phi)", "W", ("V", "I", "cos_phi"), lambda v: v["V"] * v["I"] * v["cos_phi"]),
    "power_r_v2_over_p": Formula("power_r_v2_over_p", "R = V^2 / P", "ohm", ("V", "P"), lambda v: (v["V"] ** 2) / v["P"]),
    "power_v_sqrt_pr": Formula("power_v_sqrt_pr", "V = sqrt(P * R)", "V", ("P", "R"), lambda v: math.sqrt(v["P"] * v["R"])),
    "direct_voltage_source": Formula("direct_voltage_source", "V = V_source", "V", ("V",), lambda v: v["V"]),
    "series_resistance": Formula("series_resistance", "R_total = R1 + R2 + ...", "ohm", ("resistances",), lambda v: sum(v["resistances"])),
    "parallel_resistance": Formula("parallel_resistance", "1/R_total = 1/R1 + 1/R2 + ...", "ohm", ("resistances",), lambda v: 1.0 / sum(1.0 / r for r in v["resistances"])),
    "composite_resistance": Formula("composite_resistance", "R_total = reduce series and parallel groups", "ohm", ("R_total",), lambda v: v["R_total"]),
    "bridge_symmetric_resistance": Formula(
        "bridge_symmetric_resistance",
        "R_total = R_arm when all four bridge arms are equal",
        "ohm",
        ("R_total",),
        lambda v: v["R_total"],
    ),
    "capacitor_charge_q_cv": Formula("capacitor_charge_q_cv", "Q = C * V", "C", ("C", "V"), lambda v: v["C"] * v["V"]),
    "capacitance_c_q_over_v": Formula("capacitance_c_q_over_v", "C = Q / V", "F", ("q", "V"), lambda v: v["q"] / v["V"]),
    "capacitor_energy_e_half_cv2": Formula("capacitor_energy_e_half_cv2", "E = 0.5 * C * V^2", "J", ("C", "V"), lambda v: 0.5 * v["C"] * (v["V"] ** 2)),
    "capacitor_voltage_from_energy": Formula(
        "capacitor_voltage_from_energy",
        "V = sqrt(2*E/C)",
        "V",
        ("E", "C"),
        lambda v: math.sqrt((2.0 * v["E"]) / v["C"]),
    ),
    "dielectric_capacitance_change": Formula("dielectric_capacitance_change", "C = C0 * eps_r", "F", ("C0", "eps"), lambda v: v["C0"] * v["eps"]),
    "series_capacitance": Formula("series_capacitance", "1/C_total = 1/C1 + 1/C2 + ...", "F", ("capacitances",), lambda v: 1.0 / sum(1.0 / c for c in v["capacitances"])),
    "parallel_capacitance": Formula("parallel_capacitance", "C_total = C1 + C2 + ...", "F", ("capacitances",), lambda v: sum(v["capacitances"])),
    "composite_capacitance": Formula("composite_capacitance", "C_total = reduce series and parallel groups", "F", ("C_total",), lambda v: v["C_total"]),
    "rlc_resonant_frequency": Formula("rlc_resonant_frequency", "f = 1 / (2*pi*sqrt(L*C))", "Hz", ("L", "C"), lambda v: 1.0 / (2.0 * math.pi * math.sqrt(v["L"] * v["C"]))),
    "wave_frequency": Formula("wave_frequency", "f = v / wavelength", "Hz", ("v", "wavelength"), lambda v: v["v"] / v["wavelength"]),
    "rlc_capacitance_from_resonant_frequency": Formula("rlc_capacitance_from_resonant_frequency", "C = 1 / ((2*pi*f)^2 * L)", "F", ("f", "L"), lambda v: 1.0 / (((2.0 * math.pi * v["f"]) ** 2) * v["L"])),
    "rlc_inductance_from_resonant_frequency": Formula("rlc_inductance_from_resonant_frequency", "L = 1 / ((2*pi*f)^2 * C)", "H", ("f", "C"), lambda v: 1.0 / (((2.0 * math.pi * v["f"]) ** 2) * v["C"])),
    "rlc_resonance_impedance": Formula("rlc_resonance_impedance", "Z = R at resonance", "ohm", ("R",), lambda v: v["R"]),
    "transformer_secondary_voltage": Formula(
        "transformer_secondary_voltage",
        "V_secondary = V_primary * N_secondary / N_primary",
        "V",
        ("V_primary", "N_primary", "N_secondary"),
        lambda v: v["V_primary"] * v["N_secondary"] / v["N_primary"],
    ),
    "rlc_angular_resonant_frequency": Formula("rlc_angular_resonant_frequency", "omega = 1 / sqrt(L*C)", "rad/s", ("L", "C"), lambda v: 1.0 / math.sqrt(v["L"] * v["C"])),
    "inductive_reactance": Formula("inductive_reactance", "X_L = 2*pi*f*L", "ohm", ("f", "L"), lambda v: 2.0 * math.pi * v["f"] * v["L"]),
    "capacitive_reactance": Formula("capacitive_reactance", "X_C = 1 / (2*pi*f*C)", "ohm", ("f", "C"), lambda v: 1.0 / (2.0 * math.pi * v["f"] * v["C"])),
    "series_rlc_impedance": Formula(
        "series_rlc_impedance",
        "Z = sqrt(R^2 + (X_L - X_C)^2)",
        "ohm",
        ("R", "X_L", "X_C"),
        lambda v: math.sqrt((v["R"] ** 2) + ((v["X_L"] - v["X_C"]) ** 2)),
    ),
    "coulomb_force": Formula("coulomb_force", "F = k * q1 * q2 / r^2", "N", ("q1", "q2", "r"), lambda v: K_COULOMB * abs(v["q1"] * v["q2"]) / (v["r"] ** 2)),
    "electric_field_f_over_q": Formula("electric_field_f_over_q", "E = F / q", "N/C", ("F", "q"), lambda v: v["F"] / v["q"]),
    "force_from_field_charge": Formula("force_from_field_charge", "F = q * E", "N", ("q", "E"), lambda v: abs(v["q"] * v["E"])),
    "electric_field_kq_r2": Formula("electric_field_kq_r2", "E = k * q / r^2", "N/C", ("q", "r"), lambda v: K_COULOMB * abs(v["q"]) / (v["r"] ** 2)),
    "electric_field_kq_r2_in_dielectric": Formula(
        "electric_field_kq_r2_in_dielectric",
        "E = (k/eps) * q / r^2",
        "N/C",
        ("q", "r", "eps"),
        lambda v: (K_COULOMB / v["eps"]) * abs(v["q"]) / (v["r"] ** 2),
    ),
    "electric_field_uniform_disk_axis": Formula(
        "electric_field_uniform_disk_axis",
        "E = (sigma / (2*eps0)) * (1 - z / sqrt(z^2 + R^2))",
        "N/C",
        ("sigma", "z", "R"),
        lambda v: (v["sigma"] / (2.0 * EPSILON_0)) * (1.0 - (v["z"] / math.sqrt((v["z"] ** 2) + (v["R"] ** 2)))),
    ),
    "electric_potential_uniform_disk_axis": Formula(
        "electric_potential_uniform_disk_axis",
        "V = (Q / (2*pi*eps0*R^2)) * (sqrt(z^2 + R^2) - z)",
        "V",
        ("q", "z", "R"),
        lambda v: (v["q"] / (2.0 * math.pi * EPSILON_0 * (v["R"] ** 2))) * (math.sqrt((v["z"] ** 2) + (v["R"] ** 2)) - v["z"]),
    ),
    "electric_field_infinite_line_charge": Formula(
        "electric_field_infinite_line_charge",
        "E = lam / (2*pi*eps0*r)",
        "N/C",
        ("lam", "r"),
        lambda v: v["lam"] / (2.0 * math.pi * EPSILON_0 * v["r"]),
    ),
    "electric_field_uniform_sphere_inside": Formula(
        "electric_field_uniform_sphere_inside",
        "E = k*Q*r/R^3 (inside uniformly charged sphere)",
        "N/C",
        ("q", "r", "R"),
        lambda v: K_COULOMB * v["q"] * v["r"] / (v["R"] ** 3),
    ),
    "electric_field_uniform_sphere_outside": Formula(
        "electric_field_uniform_sphere_outside",
        "E = k*Q/r^2 (outside uniformly charged sphere)",
        "N/C",
        ("q", "r"),
        lambda v: K_COULOMB * abs(v["q"]) / (v["r"] ** 2),
    ),
    "electric_potential_uniform_sphere_shell_inside": Formula(
        "electric_potential_uniform_sphere_shell_inside",
        "V = k*Q/R (inside uniformly charged thin spherical shell)",
        "V",
        ("q", "R"),
        lambda v: K_COULOMB * v["q"] / v["R"],
    ),
    "electric_potential_uniform_sphere_shell_outside": Formula(
        "electric_potential_uniform_sphere_shell_outside",
        "V = k*Q/r (outside uniformly charged thin spherical shell)",
        "V",
        ("q", "r"),
        lambda v: K_COULOMB * abs(v["q"]) / v["r"],
    ),
    "electric_field_dipole_axial": Formula(
        "electric_field_dipole_axial",
        "E = 2*k*p / r^3",
        "N/C",
        ("p", "r"),
        lambda v: 2.0 * K_COULOMB * v["p"] / (v["r"] ** 3),
    ),
    "electric_field_semicircular_arc_center": Formula(
        "electric_field_semicircular_arc_center",
        "E = 2*k*q / (pi*R^2)",
        "N/C",
        ("q", "R"),
        lambda v: (2.0 * K_COULOMB * v["q"]) / (math.pi * (v["R"] ** 2)),
    ),
    "electric_field_circular_arc_center": Formula(
        "electric_field_circular_arc_center",
        "E = 2*k*q*sin(theta/2) / (theta*R^2)",
        "N/C",
        ("q", "R", "theta_rad"),
        lambda v: (2.0 * K_COULOMB * v["q"] * math.sin(v["theta_rad"] / 2.0)) / (v["theta_rad"] * (v["R"] ** 2)),
    ),
    "electric_field_finite_line_perpendicular_bisector": Formula(
        "electric_field_finite_line_perpendicular_bisector",
        "E = k*q / (d*sqrt(d^2 + (L/2)^2))",
        "N/C",
        ("q", "d", "L"),
        lambda v: K_COULOMB * v["q"] / (v["d"] * math.sqrt((v["d"] ** 2) + ((v["L"] / 2.0) ** 2))),
    ),
    "electric_field_finite_line_axis_outside_center": Formula(
        "electric_field_finite_line_axis_outside_center",
        "E = k*q / (x^2 - (L/2)^2)",
        "N/C",
        ("q", "x", "L"),
        lambda v: K_COULOMB * v["q"] / ((v["x"] ** 2) - ((v["L"] / 2.0) ** 2)),
    ),
    "electric_field_finite_line_axis_outside_end": Formula(
        "electric_field_finite_line_axis_outside_end",
        "E = k*q / (d*(d+L))",
        "N/C",
        ("q", "d", "L"),
        lambda v: K_COULOMB * v["q"] / (v["d"] * (v["d"] + v["L"])),
    ),
    "electric_potential_finite_line_perpendicular_bisector": Formula(
        "electric_potential_finite_line_perpendicular_bisector",
        "V = 2*k*Q/L * asinh(L/(2*d))",
        "V",
        ("q", "d", "L"),
        lambda v: 2.0 * K_COULOMB * v["q"] / v["L"] * math.asinh(v["L"] / (2.0 * v["d"])),
    ),
    "electric_potential_finite_line_axis_outside_center": Formula(
        "electric_potential_finite_line_axis_outside_center",
        "V = k*Q/L * ln((x+L/2)/(x-L/2))",
        "V",
        ("q", "x", "L"),
        lambda v: K_COULOMB * v["q"] / v["L"] * math.log((v["x"] + (v["L"] / 2.0)) / (v["x"] - (v["L"] / 2.0))),
    ),
    "electric_potential_finite_line_axis_outside_end": Formula(
        "electric_potential_finite_line_axis_outside_end",
        "V = k*Q/L * ln((d+L)/d)",
        "V",
        ("q", "d", "L"),
        lambda v: K_COULOMB * v["q"] / v["L"] * math.log((v["d"] + v["L"]) / v["d"]),
    ),
    "electric_field_symmetric_loop_center_zero": Formula(
        "electric_field_symmetric_loop_center_zero",
        "E = 0 (symmetry at the center of a uniformly charged closed symmetric loop)",
        "N/C",
        (),
        lambda v: 0.0,
    ),
    "electric_potential_uniform_ring_center": Formula(
        "electric_potential_uniform_ring_center",
        "V = k*q/R",
        "V",
        ("q", "R"),
        lambda v: K_COULOMB * v["q"] / v["R"],
    ),
    "electric_potential_square_loop_center": Formula(
        "electric_potential_square_loop_center",
        "V = 2*k*Q*asinh(1) / a",
        "V",
        ("q", "a"),
        lambda v: 2.0 * K_COULOMB * v["q"] * math.asinh(1.0) / v["a"],
    ),
    "electric_potential_uniform_ring_axis": Formula(
        "electric_potential_uniform_ring_axis",
        "V = k*q/sqrt(R^2 + x^2)",
        "V",
        ("q", "R", "x"),
        lambda v: K_COULOMB * v["q"] / math.sqrt((v["R"] ** 2) + (v["x"] ** 2)),
    ),
    "magnetic_field_circular_loop_center": Formula(
        "magnetic_field_circular_loop_center",
        "B = mu0 * I / (2*R)",
        "T",
        ("I", "R"),
        lambda v: (4.0 * math.pi * 1e-7) * v["I"] / (2.0 * v["R"]),
    ),
    "spherical_capacitor_capacitance": Formula(
        "spherical_capacitor_capacitance",
        "C = 4*pi*eps0*a*b/(b-a)",
        "F",
        ("a", "b"),
        lambda v: 4.0 * math.pi * EPSILON_0 * v["a"] * v["b"] / (v["b"] - v["a"]),
    ),
    "electric_potential_circular_arc_center": Formula(
        "electric_potential_circular_arc_center",
        "V = k*q/R",
        "V",
        ("q", "R"),
        lambda v: K_COULOMB * v["q"] / v["R"],
    ),
    "coulomb_distance": Formula("coulomb_distance", "r = sqrt(k * |q1*q2| / F)", "m", ("q1", "q2", "F"), lambda v: math.sqrt(K_COULOMB * abs(v["q1"] * v["q2"]) / v["F"])),
    "potential_energy_u_qv": Formula("potential_energy_u_qv", "U = q * V", "J", ("q", "V"), lambda v: v["q"] * v["V"]),
    "potential_voltage_v_u_over_q": Formula("potential_voltage_v_u_over_q", "V = U / q", "V", ("U", "q"), lambda v: v["U"] / v["q"]),
    "direct_capacitance_reported": Formula("direct_capacitance_reported", "C = C_reported", "F", ("C",), lambda v: v["C"]),
    # Resultant force formulas
    "resultant_force_collinear_same": Formula("resultant_force_collinear_same", "R = F1 + F2 (same direction)", "N", ("F1", "F2"), lambda v: v["F1"] + v["F2"]),
    "resultant_force_collinear_opposite": Formula("resultant_force_collinear_opposite", "R = |F1 - F2| (opposite direction)", "N", ("F1", "F2"), lambda v: abs(v["F1"] - v["F2"])),
    "resultant_force_perpendicular": Formula("resultant_force_perpendicular", "R = sqrt(F1^2 + F2^2) (perpendicular)", "N", ("F1", "F2"), lambda v: math.sqrt(v["F1"] ** 2 + v["F2"] ** 2)),
    "resultant_force_angle": Formula("resultant_force_angle", "R = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))", "N", ("F1", "F2", "theta_rad"), lambda v: math.sqrt(v["F1"] ** 2 + v["F2"] ** 2 + 2 * v["F1"] * v["F2"] * math.cos(v["theta_rad"]))),
    # Measurement error formulas
    "measurement_absolute_error": Formula("measurement_absolute_error", "absolute_error = least_count", "A", ("least_count",), lambda v: v["least_count"]),
    "measurement_relative_error": Formula("measurement_relative_error", "relative_error_% = (absolute_error / reading) * 100", "%", ("absolute_error", "reading"), lambda v: (v["absolute_error"] / v["reading"]) * 100),
    "measurement_error_propagation_division": Formula("measurement_error_propagation_division", "relative_error = dU/U + dI/I for R=U/I", "%", ("dU", "U", "dI", "I"), lambda v: (v["dU"] / v["U"] + v["dI"] / v["I"]) * 100),
    # Dielectric capacitor transformations
    "dielectric_voltage_disconnected": Formula("dielectric_voltage_disconnected", "V_new = V_old / eps (disconnected)", "V", ("V_old", "eps"), lambda v: v["V_old"] / v["eps"]),
    "dielectric_voltage_connected": Formula(
        "dielectric_voltage_connected",
        "V_new = V_source (connected)",
        "V",
        ("V_source",),
        lambda v: v["V_source"],
    ),
    "dielectric_energy_disconnected": Formula(
        "dielectric_energy_disconnected",
        "E_new = E_old / eps with E_old = 0.5*C*V^2 (disconnected)",
        "J",
        ("C", "V", "eps"),
        lambda v: (0.5 * v["C"] * (v["V"] ** 2)) / v["eps"],
    ),
    "dielectric_energy_connected": Formula(
        "dielectric_energy_connected",
        "E_new = 0.5*(eps*C)*V^2 (connected)",
        "J",
        ("C", "V", "eps"),
        lambda v: 0.5 * (v["eps"] * v["C"]) * (v["V"] ** 2),
    ),
    "dielectric_energy_from_energy_disconnected": Formula(
        "dielectric_energy_from_energy_disconnected",
        "E_new = E_old / eps (disconnected)",
        "J",
        ("E_old", "eps"),
        lambda v: v["E_old"] / v["eps"],
    ),
    "dielectric_field_disconnected": Formula("dielectric_field_disconnected", "E_new = E_old / eps (disconnected)", "N/C", ("E_old", "eps"), lambda v: v["E_old"] / v["eps"]),
    "dielectric_field_connected": Formula("dielectric_field_connected", "E_new = E_old * eps (connected)", "N/C", ("E_old", "eps"), lambda v: v["E_old"] * v["eps"]),
    # Magnetism basics
    "inductor_energy": Formula("inductor_energy", "E = 0.5 * L * I^2", "J", ("L", "I"), lambda v: 0.5 * v["L"] * (v["I"] ** 2)),
    "inductor_current": Formula("inductor_current", "I = sqrt(2*E / L)", "A", ("E", "L"), lambda v: math.sqrt(2 * v["E"] / v["L"])),
    "inductor_inductance": Formula("inductor_inductance", "L = 2*E / I^2", "H", ("E", "I"), lambda v: 2 * v["E"] / (v["I"] ** 2)),
    "solenoid_B": Formula("solenoid_B", "B = mu0 * (N / l) * I", "T", ("N", "l", "I"), lambda v: (4 * math.pi * 1e-7) * (v["N"] / v["l"]) * v["I"]),
    "solenoid_total_flux": Formula("solenoid_total_flux", "Phi_total = N * B * A", "Wb", ("N", "B", "A"), lambda v: v["N"] * v["B"] * v["A"]),
    # Parallel-plate capacitor in dielectric medium
    "parallel_plate_capacitance_dielectric": Formula(
        "parallel_plate_capacitance_dielectric",
        "C = eps0 * eps_r * A / d",
        "F",
        ("A", "d", "eps"),
        lambda v: EPSILON_0 * v["eps"] * v["A"] / v["d"],
    ),
    "dielectric_constant_from_parallel_plate": Formula(
        "dielectric_constant_from_parallel_plate",
        "eps_r = C * d / (eps0 * A)",
        "dimensionless",
        ("C", "d", "A"),
        lambda v: v["C"] * v["d"] / (EPSILON_0 * v["A"]),
    ),
    # Energy density in a dielectric-filled capacitor
    "energy_density_dielectric": Formula(
        "energy_density_dielectric",
        "u = 0.5 * eps0 * eps_r * E^2 with E = V/d",
        "J/m^3",
        ("eps", "V", "d"),
        lambda v: 0.5 * EPSILON_0 * v["eps"] * ((v["V"] / v["d"]) ** 2),
    ),
}


def get_formula(formula_id: str) -> Formula:
    return FORMULAS[formula_id]
