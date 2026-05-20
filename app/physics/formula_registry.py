"""Comprehensive physics formula registry for deterministic solver.
Covers: RLC circuits (CH), Solenoid/Magnetic (DD), Measurement (TH), E-field geometry (DT/LD).
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FormulaResult:
    value: float
    unit: str
    explanation: str
    formula_name: str


# Constants
K = 8.9875e9  # Coulomb constant
MU0 = 4 * math.pi * 1e-7  # Permeability of free space
EPSILON0 = 8.854e-12  # Permittivity of free space


# ============ RLC CIRCUITS (CH - 310 questions) ============

def rlc_impedance(R: float, XL: float, XC: float) -> FormulaResult:
    Z = math.sqrt(R**2 + (XL - XC)**2)
    return FormulaResult(Z, "Ω", f"Z=√(R²+(XL-XC)²)=√({R}²+({XL}-{XC})²)={Z:.4g}Ω", "rlc_impedance")

def rlc_current(V: float, Z: float) -> FormulaResult:
    I = V / Z
    return FormulaResult(I, "A", f"I=V/Z={V}/{Z:.4g}={I:.4g}A", "rlc_current")

def rlc_current_rms(V_rms: float, Z: float) -> FormulaResult:
    I = V_rms / Z
    return FormulaResult(I, "A", f"I_rms=V_rms/Z={V_rms}/{Z:.4g}={I:.4g}A", "rlc_current_rms")

def rlc_resonant_frequency(L: float, C: float) -> FormulaResult:
    f = 1 / (2 * math.pi * math.sqrt(L * C))
    return FormulaResult(f, "Hz", f"f=1/(2π√LC)=1/(2π√({L}×{C}))={f:.4g}Hz", "rlc_resonant_freq")

def rlc_resonant_omega(L: float, C: float) -> FormulaResult:
    omega = 1 / math.sqrt(L * C)
    return FormulaResult(omega, "rad/s", f"ω₀=1/√(LC)=1/√({L}×{C})={omega:.4g}rad/s", "rlc_resonant_omega")

def rlc_quality_factor(R: float, L: float, C: float) -> FormulaResult:
    Q = (1/R) * math.sqrt(L/C)
    return FormulaResult(Q, "dimensionless", f"Q=(1/R)√(L/C)=(1/{R})√({L}/{C})={Q:.4g}", "rlc_quality_factor")

def rlc_quality_factor_from_reactance(XL: float, R: float) -> FormulaResult:
    Q = XL / R
    return FormulaResult(Q, "dimensionless", f"Q=XL/R={XL}/{R}={Q:.4g}", "rlc_quality_factor_xl")

def rlc_power(I_rms: float, R: float) -> FormulaResult:
    P = I_rms**2 * R
    return FormulaResult(P, "W", f"P=I²R={I_rms}²×{R}={P:.4g}W", "rlc_power")

def rlc_power_from_v(V_rms: float, R: float, Z: float) -> FormulaResult:
    P = V_rms**2 * R / Z**2
    return FormulaResult(P, "W", f"P=V²R/Z²={V_rms}²×{R}/{Z}²={P:.4g}W", "rlc_power_vr")

def rlc_voltage_across_R(I: float, R: float) -> FormulaResult:
    V = I * R
    return FormulaResult(V, "V", f"V_R=IR={I}×{R}={V:.4g}V", "rlc_vr")

def rlc_voltage_across_L(I: float, XL: float) -> FormulaResult:
    V = I * XL
    return FormulaResult(V, "V", f"V_L=I×XL={I}×{XL}={V:.4g}V", "rlc_vl")

def rlc_voltage_across_C(I: float, XC: float) -> FormulaResult:
    V = I * XC
    return FormulaResult(V, "V", f"V_C=I×XC={I}×{XC}={V:.4g}V", "rlc_vc")

def inductive_reactance(omega: float, L: float) -> FormulaResult:
    XL = omega * L
    return FormulaResult(XL, "Ω", f"XL=ωL={omega}×{L}={XL:.4g}Ω", "inductive_reactance")

def capacitive_reactance(omega: float, C: float) -> FormulaResult:
    XC = 1 / (omega * C)
    return FormulaResult(XC, "Ω", f"XC=1/(ωC)=1/({omega}×{C})={XC:.4g}Ω", "capacitive_reactance")

def rlc_at_resonance_current(V: float, R: float) -> FormulaResult:
    I = V / R
    return FormulaResult(I, "A", f"At resonance Z=R, I=V/R={V}/{R}={I:.4g}A", "rlc_resonance_current")

def rlc_cos_phi(R: float, Z: float) -> FormulaResult:
    cos_phi = R / Z
    return FormulaResult(cos_phi, "dimensionless", f"cos φ=R/Z={R}/{Z}={cos_phi:.4g}", "rlc_cos_phi")

def rlc_tan_phi(XL: float, XC: float, R: float) -> FormulaResult:
    tan_phi = (XL - XC) / R
    return FormulaResult(tan_phi, "dimensionless", f"tan φ=(XL-XC)/R=({XL}-{XC})/{R}={tan_phi:.4g}", "rlc_tan_phi")

def rlc_bandwidth(R: float, L: float) -> FormulaResult:
    bw = R / L
    return FormulaResult(bw, "rad/s", f"Δω=R/L={R}/{L}={bw:.4g}rad/s", "rlc_bandwidth")

def rlc_capacitance_from_resonance(f: float, L: float) -> FormulaResult:
    C = 1 / ((2*math.pi*f)**2 * L)
    return FormulaResult(C, "F", f"C=1/(4π²f²L)=1/(4π²×{f}²×{L})={C:.4g}F", "rlc_c_from_resonance")

def rlc_inductance_from_resonance(f: float, C: float) -> FormulaResult:
    L = 1 / ((2*math.pi*f)**2 * C)
    return FormulaResult(L, "H", f"L=1/(4π²f²C)=1/(4π²×{f}²×{C})={L:.4g}H", "rlc_l_from_resonance")

def rlc_voltage_ratio_resonance(V: float, R: float, XL: float) -> FormulaResult:
    """Voltage across L or C at resonance = V*XL/R = V*Q"""
    VL = V * XL / R
    return FormulaResult(VL, "V", f"V_L=V×XL/R={V}×{XL}/{R}={VL:.4g}V", "rlc_v_resonance")


# ============ SOLENOID / MAGNETIC (DD - 129 questions) ============

def solenoid_magnetic_field(N: float, l: float, I: float) -> FormulaResult:
    n = N / l
    B = MU0 * n * I
    return FormulaResult(B, "T", f"B=μ₀nI=μ₀×(N/l)×I={MU0:.4g}×{n:.4g}×{I}={B:.4g}T", "solenoid_B")

def solenoid_turns_density(N: float, l: float) -> FormulaResult:
    n = N / l
    return FormulaResult(n, "turns/m", f"n=N/l={N}/{l}={n:.4g}turns/m", "solenoid_n")

def magnetic_flux(B: float, A: float, theta: float = 0) -> FormulaResult:
    phi = B * A * math.cos(math.radians(theta))
    return FormulaResult(phi, "Wb", f"Φ=BAcosθ={B}×{A}×cos({theta}°)={phi:.4g}Wb", "magnetic_flux")

def induced_emf(N: float, delta_phi: float, delta_t: float) -> FormulaResult:
    emf = N * delta_phi / delta_t
    return FormulaResult(emf, "V", f"EMF=NΔΦ/Δt={N}×{delta_phi}/{delta_t}={emf:.4g}V", "induced_emf")

def solenoid_inductance(N: float, l: float, A: float) -> FormulaResult:
    L = MU0 * N**2 * A / l
    return FormulaResult(L, "H", f"L=μ₀N²A/l={MU0:.4g}×{N}²×{A}/{l}={L:.4g}H", "solenoid_inductance")

def inductor_energy(L: float, I: float) -> FormulaResult:
    E = 0.5 * L * I**2
    return FormulaResult(E, "J", f"E=½LI²=0.5×{L}×{I}²={E:.4g}J", "inductor_energy")

def magnetic_force_on_wire(B: float, I: float, l: float, theta: float = 90) -> FormulaResult:
    F = B * I * l * math.sin(math.radians(theta))
    return FormulaResult(F, "N", f"F=BIlsinθ={B}×{I}×{l}×sin({theta}°)={F:.4g}N", "magnetic_force_wire")

def magnetic_force_on_charge(q: float, v: float, B: float, theta: float = 90) -> FormulaResult:
    F = abs(q) * v * B * math.sin(math.radians(theta))
    return FormulaResult(F, "N", f"F=qvBsinθ={abs(q)}×{v}×{B}×sin({theta}°)={F:.4g}N", "magnetic_force_charge")

def magnetic_energy_density(B: float) -> FormulaResult:
    u = B**2 / (2 * MU0)
    return FormulaResult(u, "J/m³", f"u=B²/(2μ₀)={B}²/(2×{MU0:.4g})={u:.4g}J/m³", "magnetic_energy_density")

def transformer_voltage(V1: float, N1: float, N2: float) -> FormulaResult:
    V2 = V1 * N2 / N1
    return FormulaResult(V2, "V", f"V2=V1×N2/N1={V1}×{N2}/{N1}={V2:.4g}V", "transformer_voltage")

def transformer_current(I1: float, N1: float, N2: float) -> FormulaResult:
    I2 = I1 * N1 / N2
    return FormulaResult(I2, "A", f"I2=I1×N1/N2={I1}×{N1}/{N2}={I2:.4g}A", "transformer_current")


# ============ MEASUREMENT / ERROR (TH - 80 questions) ============

def absolute_error(measured: float, true_val: float) -> FormulaResult:
    err = abs(measured - true_val)
    return FormulaResult(err, "", f"Δ=|measured-true|=|{measured}-{true_val}|={err:.4g}", "absolute_error")

def relative_error(absolute_err: float, true_val: float) -> FormulaResult:
    rel = absolute_err / abs(true_val) * 100
    return FormulaResult(rel, "%", f"δ=Δ/true×100={absolute_err}/{true_val}×100={rel:.4g}%", "relative_error")

def mean_value(values: list[float]) -> FormulaResult:
    mean = sum(values) / len(values)
    return FormulaResult(mean, "", f"mean=Σx/n={sum(values)}/{len(values)}={mean:.4g}", "mean_value")

def random_error(values: list[float]) -> FormulaResult:
    mean = sum(values) / len(values)
    max_dev = max(abs(v - mean) for v in values)
    return FormulaResult(max_dev, "", f"random_error=max|xi-mean|={max_dev:.4g}", "random_error")

def error_sum(delta_a: float, delta_b: float) -> FormulaResult:
    delta = delta_a + delta_b
    return FormulaResult(delta, "", f"Δ(A+B)=ΔA+ΔB={delta_a}+{delta_b}={delta:.4g}", "error_sum")

def error_product_relative(rel_a: float, rel_b: float) -> FormulaResult:
    rel = rel_a + rel_b
    return FormulaResult(rel, "%", f"δ(A×B)=δA+δB={rel_a}+{rel_b}={rel:.4g}%", "error_product")


# ============ CAPACITOR (TD/NL - 362 questions) ============

def capacitor_energy(C: float, V: float) -> FormulaResult:
    E = 0.5 * C * V**2
    return FormulaResult(E, "J", f"E=½CV²=0.5×{C}×{V}²={E:.4g}J", "capacitor_energy")

def capacitor_charge(C: float, V: float) -> FormulaResult:
    Q = C * V
    return FormulaResult(Q, "C", f"Q=CV={C}×{V}={Q:.4g}C", "capacitor_charge")

def capacitor_voltage(Q: float, C: float) -> FormulaResult:
    V = Q / C
    return FormulaResult(V, "V", f"V=Q/C={Q}/{C}={V:.4g}V", "capacitor_voltage")

def parallel_plate_capacitance(A: float, d: float, epsilon_r: float = 1.0) -> FormulaResult:
    C = EPSILON0 * epsilon_r * A / d
    return FormulaResult(C, "F", f"C=ε₀εᵣA/d={EPSILON0:.4g}×{epsilon_r}×{A}/{d}={C:.4g}F", "parallel_plate_C")

def series_capacitance(capacitors: list[float]) -> FormulaResult:
    inv_sum = sum(1/c for c in capacitors)
    C = 1 / inv_sum
    return FormulaResult(C, "F", f"1/C=Σ(1/Ci), C={C:.4g}F", "series_capacitance")

def parallel_capacitance(capacitors: list[float]) -> FormulaResult:
    C = sum(capacitors)
    return FormulaResult(C, "F", f"C=ΣCi={C:.4g}F", "parallel_capacitance")

def dielectric_capacitance_change(C0: float, epsilon_r: float) -> FormulaResult:
    C = C0 * epsilon_r
    return FormulaResult(C, "F", f"C=C₀×εᵣ={C0}×{epsilon_r}={C:.4g}F", "dielectric_C")

def capacitor_energy_from_charge(Q: float, C: float) -> FormulaResult:
    E = Q**2 / (2 * C)
    return FormulaResult(E, "J", f"E=Q²/(2C)={Q}²/(2×{C})={E:.4g}J", "capacitor_energy_Q")


# ============ ELECTRIC FIELD / FORCE (LD/DT - 464 questions) ============

def coulomb_force(q1: float, q2: float, r: float) -> FormulaResult:
    F = K * abs(q1 * q2) / r**2
    return FormulaResult(F, "N", f"F=k|q1q2|/r²={K:.4g}×{abs(q1*q2):.4g}/{r}²={F:.4g}N", "coulomb_force")

def electric_field_point(q: float, r: float) -> FormulaResult:
    E = K * abs(q) / r**2
    return FormulaResult(E, "V/m", f"E=k|q|/r²={K:.4g}×{abs(q):.4g}/{r}²={E:.4g}V/m", "efield_point")

def electric_potential(q: float, r: float) -> FormulaResult:
    V = K * q / r
    return FormulaResult(V, "V", f"V=kq/r={K:.4g}×{q}/{r}={V:.4g}V", "electric_potential")

def electric_field_uniform(V: float, d: float) -> FormulaResult:
    E = V / d
    return FormulaResult(E, "V/m", f"E=V/d={V}/{d}={E:.4g}V/m", "efield_uniform")

def vector_sum_2forces(F1: float, F2: float, angle_deg: float) -> FormulaResult:
    angle_rad = math.radians(angle_deg)
    F = math.sqrt(F1**2 + F2**2 + 2*F1*F2*math.cos(angle_rad))
    return FormulaResult(F, "N", f"F=√(F1²+F2²+2F1F2cosθ)=√({F1:.4g}²+{F2:.4g}²+2×{F1:.4g}×{F2:.4g}×cos{angle_deg}°)={F:.4g}N", "vector_sum")

def vector_sum_2fields(E1: float, E2: float, angle_deg: float) -> FormulaResult:
    angle_rad = math.radians(angle_deg)
    E = math.sqrt(E1**2 + E2**2 + 2*E1*E2*math.cos(angle_rad))
    return FormulaResult(E, "V/m", f"E=√(E1²+E2²+2E1E2cosθ)={E:.4g}V/m", "vector_sum_field")

def perpendicular_bisector_field_same_sign(q1: float, q2: float, sep: float, d: float) -> FormulaResult:
    """E-field at point on perpendicular bisector, distance d from midpoint. Same sign charges."""
    half_sep = sep / 2
    r = math.sqrt(half_sep**2 + d**2)
    e1 = K * abs(q1) / r**2
    e2 = K * abs(q2) / r**2
    alpha = math.atan2(half_sep, d)
    angle_between = math.pi - 2 * alpha
    e_net = math.sqrt(e1**2 + e2**2 + 2*e1*e2*math.cos(angle_between))
    return FormulaResult(e_net, "V/m", f"r={r:.4g}m, E1={e1:.4g}, E2={e2:.4g}, E_net={e_net:.4g}V/m", "perp_bisector_same")

def perpendicular_bisector_field_opposite_sign(q1: float, q2: float, sep: float, d: float) -> FormulaResult:
    """E-field at point on perpendicular bisector, distance d from midpoint. Opposite sign charges."""
    half_sep = sep / 2
    r = math.sqrt(half_sep**2 + d**2)
    e1 = K * abs(q1) / r**2
    e2 = K * abs(q2) / r**2
    alpha = math.atan2(half_sep, d)
    e_x = e1 * math.sin(alpha) + e2 * math.sin(alpha)
    e_y = e1 * math.cos(alpha) - e2 * math.cos(alpha)
    e_net = math.sqrt(e_x**2 + e_y**2)
    return FormulaResult(e_net, "V/m", f"r={r:.4g}m, E1={e1:.4g}, E2={e2:.4g}, E_net={e_net:.4g}V/m", "perp_bisector_opp")

def triangle_force(q_target: float, q1: float, q2: float, r1: float, r2: float, r_opposite: float) -> FormulaResult:
    """Force on target charge in triangle geometry."""
    cos_angle = (r1**2 + r2**2 - r_opposite**2) / (2 * r1 * r2)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.acos(cos_angle)
    f1 = K * abs(q_target * q1) / r1**2
    f2 = K * abs(q_target * q2) / r2**2
    same_direction = (q_target * q1 > 0) == (q_target * q2 > 0)
    force_angle = angle if same_direction else math.pi - angle
    f_net = math.sqrt(f1**2 + f2**2 + 2*f1*f2*math.cos(force_angle))
    return FormulaResult(f_net, "N", f"F1={f1:.4g}N, F2={f2:.4g}N, angle={math.degrees(force_angle):.1f}°, F_net={f_net:.4g}N", "triangle_force")


# ============ OHM'S LAW / POWER / RESISTANCE ============

def ohms_law_V(I: float, R: float) -> FormulaResult:
    return FormulaResult(I*R, "V", f"V=IR={I}×{R}={I*R:.4g}V", "ohms_law_V")

def ohms_law_I(V: float, R: float) -> FormulaResult:
    return FormulaResult(V/R, "A", f"I=V/R={V}/{R}={V/R:.4g}A", "ohms_law_I")

def ohms_law_R(V: float, I: float) -> FormulaResult:
    return FormulaResult(V/I, "Ω", f"R=V/I={V}/{I}={V/I:.4g}Ω", "ohms_law_R")

def power_VI(V: float, I: float) -> FormulaResult:
    return FormulaResult(V*I, "W", f"P=VI={V}×{I}={V*I:.4g}W", "power_VI")

def power_I2R(I: float, R: float) -> FormulaResult:
    return FormulaResult(I**2*R, "W", f"P=I²R={I}²×{R}={I**2*R:.4g}W", "power_I2R")

def power_V2R(V: float, R: float) -> FormulaResult:
    return FormulaResult(V**2/R, "W", f"P=V²/R={V}²/{R}={V**2/R:.4g}W", "power_V2R")

def series_resistance(resistors: list[float]) -> FormulaResult:
    R = sum(resistors)
    return FormulaResult(R, "Ω", f"R_series=ΣRi={R:.4g}Ω", "series_R")

def parallel_resistance(resistors: list[float]) -> FormulaResult:
    inv_sum = sum(1/r for r in resistors)
    R = 1 / inv_sum
    return FormulaResult(R, "Ω", f"1/R=Σ(1/Ri), R={R:.4g}Ω", "parallel_R")


# ============ QUALITATIVE RULES ============

QUALITATIVE_RULES = {
    "capacitor_series_energy": "When identical capacitors in series: C_total=C/2, E_total=½×C/2×V²=E_single/2. Energy is LESS THAN single capacitor.",
    "capacitor_parallel_energy": "When identical capacitors in parallel: C_total=2C, E_total=½×2C×V²=2×E_single. Energy is GREATER.",
    "energy_vs_capacitance_constant_V": "E=½CV². At constant V, E is proportional to C. Graph is STRAIGHT LINE through origin (upward).",
    "energy_vs_voltage_constant_C": "E=½CV². At constant C, E is proportional to V². Graph is PARABOLA.",
    "solenoid_B_depends_on": "B=μ₀nI=μ₀(N/l)I. Depends on: n (turns/length), I (current), μ₀. Does NOT depend on: cross-sectional area, total length alone, wire diameter.",
    "magnetic_energy_vs_B": "Energy density u=B²/(2μ₀). Energy proportional to B². If B doubles, energy quadruples.",
    "resonance_condition": "At resonance: XL=XC, Z=R (minimum), I=V/R (maximum), ω₀=1/√(LC).",
    "quality_factor_meaning": "Q=ω₀L/R=1/(ω₀CR). Higher Q = sharper resonance peak, narrower bandwidth.",
    "dielectric_disconnected": "Battery disconnected, insert dielectric: Q same, C increases (×εᵣ), V decreases (/εᵣ), E decreases.",
    "dielectric_connected": "Battery connected, insert dielectric: V same, C increases (×εᵣ), Q increases (×εᵣ), E same.",
}


def lookup_qualitative(question: str) -> str | None:
    """Try to answer qualitative question from rules."""
    q_lower = question.lower()
    
    # SI unit questions
    if "si unit" in q_lower or "what is the unit" in q_lower:
        if "energy" in q_lower:
            return "Joule"
        if "force" in q_lower:
            return "Newton"
        if "electric field" in q_lower and "energy" not in q_lower:
            return "V/m"
        if "capacitance" in q_lower:
            return "Farad"
        if "charge" in q_lower:
            return "Coulomb"
        if "resistance" in q_lower:
            return "Ohm"
        if "inductance" in q_lower:
            return "Henry"
        if "magnetic" in q_lower:
            return "Tesla"
        if "power" in q_lower:
            return "Watt"
        if "voltage" in q_lower or "potential" in q_lower:
            return "Volt"
    
    # LC circuit energy questions
    if "lc" in q_lower or ("circuit" in q_lower and "oscillat" in q_lower):
        # Current i = 0 → all energy in capacitor
        if ("i = 0" in q_lower or "i=0" in q_lower or "current is zero" in q_lower or
            "current = 0" in q_lower or ("current" in q_lower and "zero" in q_lower)):
            if "where" in q_lower or "form" in q_lower or "stored" in q_lower or "is present" in q_lower:
                return "all the energy is stored in the electric field of the capacitor"
        # Current is MAXIMUM → all energy in inductor (magnetic field)
        if ("current is maximum" in q_lower or "current is max" in q_lower or
            "i is maximum" in q_lower or "i = i" in q_lower or "i=i" in q_lower or
            "maximum current" in q_lower or "current reaches maximum" in q_lower):
            if "where" in q_lower or "form" in q_lower or "stored" in q_lower or "is present" in q_lower:
                return "all energy is entirely stored in the magnetic field of the inductor"
        # Voltage v = 0 → all energy in inductor
        if ("v = 0" in q_lower or "v=0" in q_lower or "voltage is zero" in q_lower):
            if "where" in q_lower or "form" in q_lower or "stored" in q_lower:
                return "all the energy is stored in the magnetic field of the inductor"
        # WL=0 → WC=max (all energy in capacitor)
        if "wl" in q_lower and ("= 0" in q_lower or "=0" in q_lower or "zero" in q_lower):
            if "wc" in q_lower or "capacitor" in q_lower:
                return "maximum (WC = ½LI₀²)"
        # WC=0 → WL=max (all energy in inductor)
        if "wc" in q_lower and ("= 0" in q_lower or "=0" in q_lower or "zero" in q_lower):
            if "wl" in q_lower or "inductor" in q_lower:
                return "maximum (WL = ½CV₀²)"
        # At t=0 capacitor fully charged → WC=max, WL=0
        if "t = 0" in q_lower or "t=0" in q_lower:
            if "fully charged" in q_lower or "maximum voltage" in q_lower:
                if "wl" in q_lower:
                    return "0"
                if "wc" in q_lower:
                    return "maximum (WC = ½CV₀²)"
        # Graph of energy in LC: sinusoidal with phase shift π/2
        if ("graph" in q_lower or "shape" in q_lower) and "energy" in q_lower:
            if "electric" in q_lower and "magnetic" in q_lower:
                return "Sinusoidal waves with a phase shift of π/2"
    
    # Solenoid: double turns → B doubles (B = μ₀nI = μ₀(N/l)I)
    if "solenoid" in q_lower and "double" in q_lower and "turn" in q_lower:
        if "magnetic field" in q_lower or "field change" in q_lower or "how does" in q_lower:
            return "Doubled"
    
    # Solenoid: triple turns → B triples
    if "solenoid" in q_lower and "triple" in q_lower and "turn" in q_lower:
        if "magnetic field" in q_lower or "field change" in q_lower or "how does" in q_lower:
            return "Tripled"
    
    # Solenoid: current suddenly disconnected → induced EMF
    if "solenoid" in q_lower and ("disconnect" in q_lower or "cut off" in q_lower or "suddenly" in q_lower):
        if "what happens" in q_lower or "what will" in q_lower:
            return "An induced electromotive force (EMF) in the opposite direction appears"
    
    # EMF from changing magnetic flux
    if ("magnetic flux" in q_lower or "flux through" in q_lower) and ("change" in q_lower or "varies" in q_lower):
        if "what appears" in q_lower or "what is induced" in q_lower or "what happens" in q_lower:
            return "Induced electromotive force (EMF)"
    
    # Solenoid/inductor: current increases rapidly → induced EMF opposes
    if ("solenoid" in q_lower or "inductor" in q_lower or "coil" in q_lower) and \
       ("increase" in q_lower and ("current" in q_lower or "i " in q_lower)):
        if "what happens" in q_lower or "induced" in q_lower:
            if "rapidly" in q_lower or "suddenly" in q_lower or "quickly" in q_lower:
                return "Increase and the opposite current direction"
    
    # Self-inductance depends on
    if ("self-inductance" in q_lower or "inductance of a solenoid" in q_lower) and ("depend" in q_lower or "quantities" in q_lower or "what" in q_lower):
        return "Number of turns, length, cross-sectional area, and permeability of the core material"
    
    # Capacitance depends on
    if "capacitance" in q_lower and ("depend" in q_lower or "quantities" in q_lower):
        if "parallel plate" in q_lower or "plate" in q_lower:
            return "plate area, distance between plates, and dielectric constant"
    
    # Electric field at center of symmetric charge configuration
    if ("center" in q_lower or "intersection" in q_lower or "middle" in q_lower) and ("square" in q_lower or "diagonal" in q_lower):
        if ("same" in q_lower and ("magnitude" in q_lower or "charge" in q_lower or "equal" in q_lower)) or \
           ("four" in q_lower and "vertices" in q_lower and "same magnitude" in q_lower):
            if "field" in q_lower:
                return "0"
        # Symmetric arrangement: +q at opposite corners, -q at other opposite corners
        if "positive" in q_lower and "negative" in q_lower and "vertices" in q_lower:
            if "field" in q_lower:
                return "0"
    
    # Resistance depends on
    if "resistance" in q_lower and ("depend" in q_lower or "quantities" in q_lower):
        if "wire" in q_lower or "conductor" in q_lower:
            return "length, cross-sectional area, resistivity (material), and temperature"
    
    if "series" in q_lower and "energy" in q_lower and "compare" in q_lower:
        return "less than"
    if "parallel" in q_lower and "energy" in q_lower and "compare" in q_lower:
        return "greater than"
    if ("graph" in q_lower or "shape" in q_lower) and "capacitance" in q_lower and ("constant" in q_lower or "keeping" in q_lower):
        return "Upward straight line"
    if ("graph" in q_lower or "shape" in q_lower) and "voltage" in q_lower and "capacitance" not in q_lower:
        return "Parabola"
    if "does not depend" in q_lower or "not depend" in q_lower or "doesn't depend" in q_lower:
        if "solenoid" in q_lower or "magnetic field" in q_lower:
            return "cross-sectional area (S)"
    if "magnetic" in q_lower and "energy" in q_lower and ("increase" in q_lower or "change" in q_lower):
        # Only match qualitative wording, not numeric problems
        if "calculate" not in q_lower and "find" not in q_lower and "determine" not in q_lower and \
           ("how does" in q_lower or "what happens" in q_lower or "what is the relationship" in q_lower):
            return "the magnetic field energy increases proportionally to B²"
    
    # Bulbs in parallel: lower R → higher current → brighter
    if "bulb" in q_lower and "parallel" in q_lower and ("lower resistance" in q_lower or "smaller resistance" in q_lower):
        if "bright" in q_lower:
            return "Brighter because the current is higher"
    
    # Parallel circuit: one lamp current increases → total current increases
    if "parallel" in q_lower and "current" in q_lower:
        if ("one lamp" in q_lower or "one bulb" in q_lower or "through one" in q_lower) and "increase" in q_lower:
            if "total" in q_lower or "how will" in q_lower or "how does" in q_lower:
                return "Total current increases."
    
    # Bulbs in series: lower R → lower power → dimmer
    if "bulb" in q_lower and "series" in q_lower and ("lower resistance" in q_lower or "smaller resistance" in q_lower):
        if "bright" in q_lower:
            return "Dimmer because the power is lower"
    
    # Capacitor energy proportional to V²: "voltage doubles → energy ×4"
    if "capacitor" in q_lower and "energy" in q_lower:
        if "proportional" in q_lower or "directly proportional" in q_lower:
            if "which" in q_lower or "what" in q_lower:
                return "The square of the voltage (U²)"
        if "double" in q_lower and "voltage" in q_lower:
            if "how" in q_lower or "change" in q_lower:
                return "Increase by 4 times"
        if "triple" in q_lower and "voltage" in q_lower:
            if "how" in q_lower or "change" in q_lower:
                return "Increase by 9 times"
        if "half" in q_lower and "voltage" in q_lower:
            if "how" in q_lower or "change" in q_lower:
                return "Decrease to 1/4"
    
    # Resonance Yes/No: check if given frequency matches f₀ = 1/(2π√LC)
    if ("resonance" in q_lower or "resonant" in q_lower) and ("does" in q_lower or "is there" in q_lower or "occur" in q_lower):
        # Extract f, L, C and check if f = 1/(2π√LC)
        import math as _math
        m_f = re.search(r'(?:f\s*=\s*|frequency\s+(?:of\s+)?|at\s+(?:a\s+)?frequency\s+(?:of\s+)?)(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)', question, re.I)
        m_L = re.search(r'L\s*=\s*(\d+(?:\.\d+)?)\s*(H|mH|μH|µH)', question, re.I)
        m_C = re.search(r'C\s*=\s*(\d+(?:\.\d+)?)\s*(F|mF|μF|µF|uF|nF|pF)', question, re.I)
        if m_f and m_L and m_C:
            f_val = float(m_f.group(1)) * {"Hz": 1, "kHz": 1e3, "MHz": 1e6}.get(m_f.group(2), 1)
            L_val = float(m_L.group(1)) * {"H": 1, "mH": 1e-3, "μH": 1e-6, "µH": 1e-6}.get(m_L.group(2), 1)
            C_val = float(m_C.group(1)) * {"F": 1, "mF": 1e-3, "μF": 1e-6, "µF": 1e-6, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12}.get(m_C.group(2), 1)
            f0 = 1 / (2 * _math.pi * _math.sqrt(L_val * C_val))
            if abs(f_val - f0) / max(f0, 1) < 0.01:
                return "Yes"
            else:
                return "No"
    
    # Capacitor with dielectric replaced: C ∝ ε, so if ε halved → C halved
    if "dielectric" in q_lower and "capacit" in q_lower and \
       ("replac" in q_lower or "change" in q_lower or "different" in q_lower):
        # Try to extract old and new ε
        m1 = re.search(r'\u03b5\s*=\s*(\d+)', q_lower)
        all_eps = re.findall(r'\u03b5\s*=\s*(\d+)', q_lower)
        if len(all_eps) >= 2:
            try:
                eps_old = float(all_eps[0])
                eps_new = float(all_eps[1])
                ratio = eps_new / eps_old
                if abs(ratio - 0.5) < 0.01:
                    return "decreases by half"
                if abs(ratio - 2.0) < 0.01:
                    return "doubles"
                if abs(ratio - 0.25) < 0.01:
                    return "decreases to 1/4"
                if abs(ratio - 4.0) < 0.01:
                    return "increases 4 times"
                if abs(ratio - 1.0) < 0.01:
                    return "remains the same"
            except:
                pass
    
    # Efficiency from energy: useful/(useful+dissipated)
    if "efficiency" in q_lower and "energy" in q_lower:
        # Match: "dissipated X, magnetic Y" or similar
        m_diss = re.search(r'dissipat\w+\s+(?:electrical\s+)?energy\s+(?:was\s+)?(?:measured\s+to\s+be\s+)?(\d+(?:\.\d+)?)\s*(J|mJ|μJ|µJ|nJ)', q_lower, re.I)
        m_use = re.search(r'(?:maximum\s+)?(?:magnetic|useful|stored)\s+energy\s+(?:in\s+the\s+coil\s+)?(?:was\s+)?(\d+(?:\.\d+)?)\s*(J|mJ|μJ|µJ|nJ)', q_lower, re.I)
        if m_diss and m_use:
            d_val = float(m_diss.group(1))
            u_val = float(m_use.group(1))
            eff = u_val / (u_val + d_val) * 100
            return f"{eff:.4g}"
    
    # LC oscillation: "energy increases from zero to maximum, magnetic decreases" → conservation
    if "energy" in q_lower and "capacitor" in q_lower and "magnetic" in q_lower and \
       ("increase" in q_lower or "decrease" in q_lower):
        if "indicate" in q_lower or "law" in q_lower or "principle" in q_lower or \
           "demonstrate" in q_lower or "follows" in q_lower or "what does" in q_lower:
            return "Conservation of energy"
    if "resonan" in q_lower and "angular" in q_lower and "frequency" in q_lower:
        if "what is" in q_lower or "formula" in q_lower:
            return "ω = 1/√(LC)"
    
    return None
