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
    "power_r_v2_over_p": Formula("power_r_v2_over_p", "R = V^2 / P", "ohm", ("V", "P"), lambda v: (v["V"] ** 2) / v["P"]),
    "power_v_sqrt_pr": Formula("power_v_sqrt_pr", "V = sqrt(P * R)", "V", ("P", "R"), lambda v: math.sqrt(v["P"] * v["R"])),
    "series_resistance": Formula("series_resistance", "R_total = R1 + R2 + ...", "ohm", ("resistances",), lambda v: sum(v["resistances"])),
    "parallel_resistance": Formula("parallel_resistance", "1/R_total = 1/R1 + 1/R2 + ...", "ohm", ("resistances",), lambda v: 1.0 / sum(1.0 / r for r in v["resistances"])),
    "composite_resistance": Formula("composite_resistance", "R_total = reduce series and parallel groups", "ohm", ("R_total",), lambda v: v["R_total"]),
    "capacitor_charge_q_cv": Formula("capacitor_charge_q_cv", "Q = C * V", "C", ("C", "V"), lambda v: v["C"] * v["V"]),
    "capacitance_c_q_over_v": Formula("capacitance_c_q_over_v", "C = Q / V", "F", ("q", "V"), lambda v: v["q"] / v["V"]),
    "capacitor_energy_e_half_cv2": Formula("capacitor_energy_e_half_cv2", "E = 0.5 * C * V^2", "J", ("C", "V"), lambda v: 0.5 * v["C"] * (v["V"] ** 2)),
    "series_capacitance": Formula("series_capacitance", "1/C_total = 1/C1 + 1/C2 + ...", "F", ("capacitances",), lambda v: 1.0 / sum(1.0 / c for c in v["capacitances"])),
    "parallel_capacitance": Formula("parallel_capacitance", "C_total = C1 + C2 + ...", "F", ("capacitances",), lambda v: sum(v["capacitances"])),
    "rlc_resonant_frequency": Formula("rlc_resonant_frequency", "f = 1 / (2*pi*sqrt(L*C))", "Hz", ("L", "C"), lambda v: 1.0 / (2.0 * math.pi * math.sqrt(v["L"] * v["C"]))),
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
    "electric_field_kq_r2": Formula("electric_field_kq_r2", "E = k * q / r^2", "N/C", ("q", "r"), lambda v: K_COULOMB * abs(v["q"]) / (v["r"] ** 2)),
    "electric_field_kq_r2_in_dielectric": Formula(
        "electric_field_kq_r2_in_dielectric",
        "E = (k/eps) * q / r^2",
        "N/C",
        ("q", "r", "eps"),
        lambda v: (K_COULOMB / v["eps"]) * abs(v["q"]) / (v["r"] ** 2),
    ),
    "coulomb_distance": Formula("coulomb_distance", "r = sqrt(k * |q1*q2| / F)", "m", ("q1", "q2", "F"), lambda v: math.sqrt(K_COULOMB * abs(v["q1"] * v["q2"]) / v["F"])),
    "potential_energy_u_qv": Formula("potential_energy_u_qv", "U = q * V", "J", ("q", "V"), lambda v: v["q"] * v["V"]),
    "potential_voltage_v_u_over_q": Formula("potential_voltage_v_u_over_q", "V = U / q", "V", ("U", "q"), lambda v: v["U"] / v["q"]),
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
