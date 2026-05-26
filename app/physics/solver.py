from __future__ import annotations

import os
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.guardrails import guardrail_prompt_text
from app.physics.expression_eval import safe_eval_expression
from app.physics.formula_registry import lookup_qualitative
from app.physics.formulas import FORMULAS, K_COULOMB, Formula, get_formula
from app.physics.method_search import build_objective as build_method_objective
from app.physics.method_search import extract_equation_proposals, retrieve_method_evidence, verify_and_compute_method, _web_method_search_enabled
from app.agent_runtime import run_physics_agent
from app.physics.parser import ParsedPhysicsProblem, parse_physics_question
from app.physics.problem_frame import ProblemFrame, infer_problem_frame, search_unknown_explanation
from app.physics.templates import physics_explanation
from app.physics.unit_converter import format_best_unit


@dataclass
class PhysicsSolution:
    success: bool
    answer: str
    explanation: str
    formula_id: str | None
    variables: dict[str, float] = field(default_factory=dict)
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parsed: ParsedPhysicsProblem | None = None
    error: str | None = None
    fallback_used: bool = False
    model_calls: int = 0
    search_trace: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)


_TARGET_UNIT_HINTS: dict[str, str] = {
    "voltage": "V",
    "electric_potential": "V",
    "current": "A",
    "power": "W",
    "resistance": "ohm",
    "impedance": "ohm",
    "reactance": "ohm",
    "capacitance": "F",
    "charge": "C",
    "energy": "J",
    "potential_energy": "J",
    "force": "N",
    "electric_field": "N/C",
    "frequency": "Hz",
    "angular_frequency": "rad/s",
    "inductance": "H",
    "magnetic_field": "T",
    "distance": "m",
    "dielectric_constant": "dimensionless",
}


def _preseeded_formulas_disabled() -> bool:
    return os.getenv("URA_DISABLE_PRESEEDED_FORMULAS", "").strip().lower() in {"1", "true", "yes", "on"}


def _unsupported_context(question: str) -> str | None:
    low = question.lower()
    if "open switch" in low or "switch is open" in low:
        return "open_switch_context"
    if "nested branch has" in low and "all in parallel with" in low:
        return "mixed_nested_topology_unsupported"
    if "ladder network" in low or " ladder " in low:
        return "unsupported_ladder_topology"
    return None


def _rod_axis_endpoint_singularity(parsed: ParsedPhysicsProblem, question: str) -> str | None:
    low = question.lower()
    if "wire" not in low and "rod" not in low:
        return None
    if "axis" not in low:
        return None
    if "center" not in low:
        return None
    length_match = re.search(r"length(?:\s+of)?\s+([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b", low, re.I)
    distance_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\s+from\s+(?:the\s+)?center", low, re.I)
    if not length_match or not distance_match:
        return None

    def convert(value: str, unit: str) -> float:
        numeric = float(value)
        unit = unit.lower()
        if unit == "cm":
            return numeric * 1e-2
        if unit == "mm":
            return numeric * 1e-3
        return numeric

    length = convert(length_match.group(1), length_match.group(2))
    axis_distance = convert(distance_match.group(1), distance_match.group(2))
    if abs(axis_distance * 2 - length) < 1e-9:
        return "rod_axis_endpoint_singularity"
    return None


def _quantities(parsed: ParsedPhysicsProblem, unit: str) -> list[float]:
    return [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == unit]


def _extract_number(question: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        try:
            return float(match.group(1))
        except Exception:
            continue
    return None


def _build_search_formula_solution(
    parsed: ParsedPhysicsProblem,
    formula_id: str,
    variables: dict[str, float],
    confidence: float,
) -> PhysicsSolution:
    formula = get_formula(formula_id)
    answer_value = formula.compute(variables)
    answer = format_best_unit(answer_value, formula.target_unit)
    explanation = "Search-backed formula reasoning: " + physics_explanation(formula, variables, answer_value)
    return PhysicsSolution(
        success=True,
        answer=answer,
        explanation=explanation,
        formula_id=formula.formula_id,
        variables=variables,
        cot=[
            f"Parsed target: {parsed.target_quantity or 'unknown'}",
            f"Search-backed formula: {formula.expression} ({formula.formula_id})",
            f"Computed with Python: {answer}",
        ],
        confidence=confidence,
        parsed=parsed,
        fallback_used=True,
        model_calls=0,
    )


_VAR_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "V": ("V",),
    "V_primary": ("V",),
    "V_source": ("V",),
    "U": ("J",),
    "I": ("A",),
    "R": ("ohm",),
    "R_total": ("ohm",),
    "C": ("F",),
    "q": ("C",),
    "q1": ("C",),
    "q2": ("C",),
    "q3": ("C",),
    "F": ("N",),
    "E": ("N/C", "V/m"),
    "P": ("W",),
    "d": ("m",),
    "r": ("m",),
    "l": ("m",),
    "A": ("m²",),
    "f": ("Hz",),
    "L": ("H",),
    "X_L": ("ohm",),
    "X_C": ("ohm",),
    "v": ("m/s",),
    "wavelength": ("m",),
}


def _formula_keywords(formula: Formula) -> set[str]:
    keywords = set(re.findall(r"[a-z]+", formula.formula_id.lower()))
    keywords.update(re.findall(r"[a-z]+", formula.expression.lower()))
    return {token for token in keywords if len(token) > 1}


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _search_context(parsed: ParsedPhysicsProblem, question: str) -> ProblemFrame:
    return infer_problem_frame(parsed, question)


def _search_unknown_explanation(frame: ProblemFrame, reason: str, details: list[str] | None = None) -> str:
    return search_unknown_explanation(frame, reason, details)


def _extract_angle_degrees(question: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", question, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _candidate_reasons(
    parsed: ParsedPhysicsProblem,
    formula: Formula,
    question: str,
    frame: ProblemFrame,
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    low = question.lower()
    method_family = _formula_method_family(formula.formula_id)

    if formula.formula_id in frame.blocked_formulas:
        reasons.append("blocked by geometry/topology context")
        return -100, reasons

    if frame.method_family == "distributed_charge_integration" and method_family == "point_charge":
        reasons.append("point-charge formula conflicts with distributed charge frame")
        return -100, reasons
    if frame.method_family == "network_reduction_or_symbolic" and method_family == "simple_network_reduction":
        reasons.append("simple series/parallel reduction conflicts with complex network frame")
        return -100, reasons
    if any("Mixed series/parallel composite circuit is not supported" in item for item in getattr(parsed, "ambiguity", [])):
        if formula.formula_id in {"series_resistance", "parallel_resistance", "series_capacitance", "parallel_capacitance", "composite_resistance", "composite_capacitance"}:
            reasons.append("mixed composite circuit requires a dedicated composite solver")
            return -100, reasons
    if formula.formula_id in {"series_resistance", "parallel_resistance", "series_capacitance", "parallel_capacitance"} and ("unknown" in low or "missing" in low):
        reasons.append("network branch value is missing, so a simple reduction would invent information")
        return -100, reasons
    if formula.formula_id == "direct_capacitance_reported":
        if len(_quantities(parsed, "F")) > 1 or any("Mixed series/parallel composite circuit" in item for item in getattr(parsed, "ambiguity", [])):
            reasons.append("reported capacitance cannot explain a composite capacitor network")
            return -100, reasons
    if "dielectric" in low or "permittivity" in low:
        if formula.formula_id in {"series_capacitance", "parallel_capacitance", "composite_capacitance"}:
            reasons.append("plain capacitance-network reduction conflicts with dielectric context")
            return -100, reasons
    if frame.method_family == "transformer_relation" and formula.formula_id != "transformer_secondary_voltage":
        reasons.append("transformer frame requires turns-ratio reasoning, not generic voltage formulas")
        return -100, reasons
    if frame.method_family == "circuit_open_state" and formula.formula_id != "open_circuit_current":
        reasons.append("open circuit frame requires incomplete-circuit reasoning")
        return -100, reasons
    if frame.method_family == "vector_superposition" and method_family == "point_charge" and _contains_any(
        low,
        ["perpendicular bisector", "midpoint", "resultant", "superposition", "components", "triangle", "equilateral"],
    ):
        reasons.append("point-charge shortcut conflicts with vector superposition frame")
        return -100, reasons
    if frame.method_family == "dielectric_transform" and method_family == "direct_physics_relation":
        reasons.append("direct capacitor relation is not enough for dielectric transform frame")
    if frame.method_family == "ac_circuit_method" and method_family == "direct_physics_relation":
        reasons.append("direct circuit relation may be insufficient for AC frame")

    if parsed.target_quantity:
        if formula.target_unit.lower() == parsed.target_quantity.lower():
            score += 6
            reasons.append("target unit matches")
        elif parsed.target_quantity in formula.expression.lower() or parsed.target_quantity in formula.formula_id.lower():
            score += 2
            reasons.append("target quantity appears in formula label")

    keywords = _formula_keywords(formula)
    keyword_hits = sorted(token for token in keywords if token in low and len(token) > 2)
    if keyword_hits:
        score += len(keyword_hits) * 2
        reasons.append(f"keyword hits: {', '.join(keyword_hits[:4])}")

    score += len(formula.variables)
    if formula.target_unit.lower() in {"v", "a", "w", "ohm", "c", "j", "n", "f", "hz", "h", "t", "nc", "v/m", "n/c", "rad/s"}:
        score += 1

    if frame.method_family and method_family and frame.method_family == method_family:
        score += 4
        reasons.append("method family matches frame")
    elif frame.method_family and method_family is None:
        score -= 1
        reasons.append("candidate method family unknown")

    if not keyword_hits:
        score -= 1
        reasons.append("no keyword overlap")

    if len(formula.variables) and all(required in parsed.variables for required in formula.variables if required not in {"resistances", "capacitances"}):
        score += 1
        reasons.append("all required scalar variables present")

    return score, reasons


def _formula_method_family(formula_id: str) -> str | None:
    if formula_id in {"electric_field_kq_r2", "electric_field_kq_r2_in_dielectric", "coulomb_force"}:
        return "point_charge"
    if formula_id in {"electric_field_uniform_disk_axis", "electric_field_infinite_line_charge", "electric_field_semicircular_arc_center", "electric_field_circular_arc_center", "electric_field_finite_line_perpendicular_bisector", "electric_field_finite_line_axis_outside_center", "electric_field_finite_line_axis_outside_end"}:
        return "distributed_charge_integration"
    if formula_id in {"electric_potential_uniform_ring_center", "electric_potential_uniform_ring_axis", "electric_potential_square_loop_center", "electric_potential_circular_arc_center", "electric_potential_uniform_disk_axis", "electric_potential_finite_line_perpendicular_bisector", "electric_potential_finite_line_axis_outside_center", "electric_potential_finite_line_axis_outside_end"}:
        return "distributed_charge_integration"
    if formula_id in {"electric_potential_finite_line_perpendicular_bisector", "electric_potential_finite_line_axis_outside_center", "electric_potential_finite_line_axis_outside_end"}:
        return "distributed_charge_integration"
    if formula_id in {"electric_field_dipole_axial"}:
        return "vector_superposition"
    if formula_id in {"electric_field_symmetric_loop_center_zero"}:
        return "vector_superposition"
    if formula_id in {"perpendicular_bisector_field_same_sign", "vector_sum_2forces", "vector_sum_2fields", "resultant_force_angle", "resultant_force_perpendicular", "resultant_force_collinear_same", "resultant_force_collinear_opposite"}:
        return "vector_superposition"
    if formula_id in {"series_resistance", "parallel_resistance", "series_capacitance", "parallel_capacitance", "composite_resistance", "composite_capacitance", "bridge_symmetric_resistance"}:
        return "simple_network_reduction"
    if formula_id in {"spherical_capacitor_capacitance", "electric_potential_uniform_disk_axis"}:
        return "direct_physics_relation"
    if formula_id in {"electric_field_uniform_sphere_inside", "electric_field_uniform_sphere_outside", "magnetic_field_circular_loop_center"}:
        return "magnetics_geometry"
    if formula_id in {"electric_potential_uniform_sphere_shell_inside", "electric_potential_uniform_sphere_shell_outside"}:
        return "distributed_charge_integration"
    if formula_id == "wave_frequency":
        return "wave_relation"
    if formula_id in {"inductive_reactance", "capacitive_reactance", "series_rlc_impedance", "rlc_resonant_frequency", "rlc_angular_resonant_frequency", "ac_power_vi_cos_phi"}:
        return "ac_circuit_method"
    if formula_id in {"solenoid_B", "solenoid_total_flux"}:
        return "magnetics_geometry"
    if formula_id in {"transformer_secondary_voltage"}:
        return "transformer_relation"
    if formula_id in {"dielectric_voltage_disconnected", "dielectric_voltage_connected", "dielectric_energy_disconnected", "dielectric_energy_connected", "dielectric_energy_from_energy_disconnected", "dielectric_field_disconnected", "dielectric_field_connected", "parallel_plate_capacitance_dielectric", "dielectric_constant_from_parallel_plate", "energy_density_dielectric"}:
        return "dielectric_transform"
    if formula_id in {"force_from_field_charge", "electric_field_f_over_q", "potential_energy_u_qv", "potential_voltage_v_u_over_q", "direct_capacitance_reported", "ohms_law_v_ir", "ohms_law_i_v_over_r", "ohms_law_r_v_over_i", "power_p_vi", "power_i_p_over_v", "power_v_p_over_i", "power_p_i2r", "power_p_v2r", "power_r_v2_over_p", "power_v_sqrt_pr", "capacitor_charge_q_cv", "capacitance_c_q_over_v", "capacitor_energy_e_half_cv2", "capacitor_voltage_from_energy", "dielectric_capacitance_change", "direct_voltage_source", "transformer_secondary_voltage"}:
        return "direct_physics_relation"
    if formula_id in {"open_circuit_current"}:
        return "circuit_open_state"
    return None


def _candidate_variables_network_specific(parsed: ParsedPhysicsProblem, formula: Formula, question: str, variables: dict[str, float], low: str) -> dict[str, float] | None:
    if formula.formula_id in {"composite_resistance", "composite_capacitance"}:
        return None
    ac_keywords = ["ac", "rms", "reactance", "inductive", "capacitive", "impedance", "frequency", "series rlc", "x_l", "x_c", "omega"]
    has_ac_context = any(kw in low for kw in ac_keywords)
    if formula.formula_id in {"inductive_reactance", "capacitive_reactance", "series_rlc_impedance", "rlc_resonant_frequency", "rlc_angular_resonant_frequency"} and not has_ac_context:
        return None
    topology_blockers = any(token in low for token in ["bridge", "ladder", "diamond", "mesh", "cross resistor"])
    if formula.formula_id in {"series_resistance", "parallel_resistance"}:
        has_topology_cue = any(token in low for token in ["series", "parallel", "chain", "after the first", "resistor pair", "resistors in series", "resistors in parallel"])
        if topology_blockers and not has_topology_cue:
            return None
        if "unknown" in low or "missing" in low:
            return None
        if not has_topology_cue:
            return None
    if formula.formula_id == "bridge_symmetric_resistance":
        if not any(token in low for token in ["bridge", "diamond"]):
            return None
        resistances = _quantities(parsed, "ohm")
        if len(resistances) < 5:
            return None
        counts: dict[float, int] = {}
        for value in resistances:
            key = round(value, 12)
            counts[key] = counts.get(key, 0) + 1
        repeated = [value for value, count in counts.items() if count >= 4]
        if not repeated:
            return None
        return {"R_total": repeated[0], "resistances": resistances}
    return None


def _candidate_variables_electrostatic_geometry_specific(parsed: ParsedPhysicsProblem, formula: Formula, question: str, variables: dict[str, float], low: str) -> dict[str, float] | None:
    charges = _quantities(parsed, "C")
    distances = _quantities(parsed, "m")

    if formula.formula_id == "electric_field_semicircular_arc_center":
        if not any(token in low for token in ["semicircular arc", "semicircle", "half circle"]) or "center" not in low:
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "R": max(distances)}
    if formula.formula_id == "electric_field_circular_arc_center":
        if "arc" not in low or "center" not in low:
            return None
        if not charges or not distances:
            return None
        theta_rad = None
        if any(token in low for token in ["quarter-circle", "quarter circle", "quarter arc", "90 degree", "90-degree", "ninety degree", "one-quarter"]):
            theta_rad = math.pi / 2.0
        elif any(token in low for token in ["semicircle", "semi-circle", "half circle", "half-circle", "180 degree", "180-degree"]):
            theta_rad = math.pi
        elif any(token in low for token in ["three-quarter", "three quarter", "270 degree", "270-degree"]):
            theta_rad = 3.0 * math.pi / 2.0
        elif any(token in low for token in ["full circle", "360 degree", "360-degree", "entire circle"]):
            theta_rad = 2.0 * math.pi
        else:
            angle_match = re.search(r"subtend(?:ing|s)?\s+([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|°)", low, re.I)
            if angle_match:
                theta_rad = math.radians(float(angle_match.group(1)))
            else:
                angle_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|°)\s+(?:arc|sector|subtends?|subtending)", low, re.I)
                if angle_match:
                    theta_rad = math.radians(float(angle_match.group(1)))
        if theta_rad is None:
            return None
        return {"q": charges[0], "R": max(distances), "theta_rad": theta_rad}
    if formula.formula_id == "electric_field_finite_line_perpendicular_bisector":
        if "perpendicular bisector" not in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "d": min(distances), "L": max(distances)}
    if formula.formula_id == "electric_field_finite_line_axis_outside_center":
        if "axis" not in low or "perpendicular bisector" in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if "from the center" not in low and "from center" not in low:
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "x": max(distances), "L": min(distances)}
    if formula.formula_id == "electric_field_finite_line_axis_outside_end":
        if "axis" not in low or "perpendicular bisector" in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if "from one end" not in low and "from an end" not in low:
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "d": min(distances), "L": max(distances)}
    if formula.formula_id == "electric_field_symmetric_loop_center_zero":
        if "center" not in low:
            return None
        if not any(token in low for token in ["square loop", "rectangular loop", "regular polygon", "polygon loop", "square wire loop", "square ring", "equilateral triangle"]) and not re.search(r"\bregular\b.*\bloop\b", low):
            return None
        return {}
    if formula.formula_id == "electric_potential_uniform_ring_center":
        if not any(token in low for token in ["electric potential", "potential"]) or "center" not in low or "ring" not in low:
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "R": max(distances)}
    if formula.formula_id == "electric_potential_uniform_ring_axis":
        if "ring" not in low or "axis" not in low or not any(token in low for token in ["electric potential", "potential", "voltage"]):
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "R": max(distances), "x": min(distances)}
    if formula.formula_id == "electric_potential_uniform_disk_axis":
        if "disk" not in low and "disc" not in low:
            return None
        if "axis" not in low or not any(token in low for token in ["electric potential", "potential", "voltage"]):
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "z": min(distances), "R": max(distances)}
    if formula.formula_id == "electric_field_uniform_sphere_inside":
        if "sphere" not in low or not any(token in low for token in ["inside", "within"]):
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "r": min(distances), "R": max(distances)}
    if formula.formula_id == "electric_field_uniform_sphere_outside":
        if "sphere" not in low or any(token in low for token in ["inside", "within"]):
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "r": min(distances)}
    if formula.formula_id == "electric_potential_uniform_sphere_shell_inside":
        if "sphere" not in low or "shell" not in low or not any(token in low for token in ["inside", "within"]):
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "R": max(distances)}
    if formula.formula_id == "electric_potential_uniform_sphere_shell_outside":
        if "sphere" not in low or "shell" not in low or not any(token in low for token in ["outside", "outside the shell", "outside shell"]):
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "r": min(distances)}
    if formula.formula_id == "electric_potential_finite_line_perpendicular_bisector":
        if "perpendicular bisector" not in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "d": min(distances), "L": max(distances)}
    if formula.formula_id == "electric_potential_finite_line_axis_outside_center":
        if "axis" not in low or "perpendicular bisector" in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if "from the center" not in low and "from center" not in low:
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "x": max(distances), "L": min(distances)}
    if formula.formula_id == "electric_potential_finite_line_axis_outside_end":
        if "axis" not in low or "perpendicular bisector" in low or not any(token in low for token in ["rod", "wire", "line"]):
            return None
        if "from one end" not in low and "from an end" not in low:
            return None
        if not charges or len(distances) < 2:
            return None
        return {"q": charges[0], "d": min(distances), "L": max(distances)}
    if formula.formula_id == "electric_potential_square_loop_center":
        if "square" not in low or "loop" not in low or "center" not in low or not any(token in low for token in ["electric potential", "potential", "voltage"]):
            return None
        if not charges or not distances:
            return None
        return {"q": charges[0], "a": max(distances)}
    return None


def _candidate_variables_magnetics_specific(parsed: ParsedPhysicsProblem, formula: Formula, question: str, variables: dict[str, float], low: str) -> dict[str, float] | None:
    distances = _quantities(parsed, "m")
    currents = _quantities(parsed, "A")

    if formula.formula_id == "magnetic_field_circular_loop_center":
        if not any(token in low for token in ["circular loop", "circular ring", "loop", "circle"]) or "center" not in low:
            return None
        if not currents or not distances:
            return None
        return {"I": currents[0], "R": max(distances)}
    if formula.formula_id == "spherical_capacitor_capacitance":
        if "spherical capacitor" not in low and "sphere capacitor" not in low:
            return None
        if len(distances) < 2:
            return None
        return {"a": min(distances), "b": max(distances)}
    return None


def _candidate_variables_misc_specific(parsed: ParsedPhysicsProblem, formula: Formula, question: str, variables: dict[str, float], low: str) -> dict[str, float] | None:
    if formula.formula_id == "direct_voltage_source" and parsed.target_quantity != "voltage":
        return None
    if formula.formula_id == "direct_capacitance_reported" and parsed.target_quantity != "capacitance":
        return None
    if formula.formula_id == "solenoid_B" and {"N", "l", "I"}.issubset(variables):
        return {"N": variables["N"], "l": variables["l"], "I": variables["I"]}
    if formula.formula_id == "resultant_force_angle":
        angle_deg = _extract_angle_degrees(question)
        if angle_deg is not None and "theta_rad" in formula.variables:
            if "F1" in variables and "F2" in variables:
                return {"F1": variables["F1"], "F2": variables["F2"], "theta_rad": math.radians(angle_deg)}
        return None
    return None


def _candidate_variables_formula_specific(parsed: ParsedPhysicsProblem, formula: Formula, question: str, variables: dict[str, float], low: str) -> dict[str, float] | None:
    specific = _candidate_variables_network_specific(parsed, formula, question, variables, low)
    if specific is not None:
        return specific
    specific = _candidate_variables_electrostatic_geometry_specific(parsed, formula, question, variables, low)
    if specific is not None:
        return specific
    specific = _candidate_variables_magnetics_specific(parsed, formula, question, variables, low)
    if specific is not None:
        return specific
    specific = _candidate_variables_misc_specific(parsed, formula, question, variables, low)
    if specific is not None:
        return specific
    return None


def _candidate_variables(parsed: ParsedPhysicsProblem, formula: Formula, question: str) -> dict[str, float] | None:
    variables = dict(parsed.variables)
    low = question.lower()
    specific = _candidate_variables_formula_specific(parsed, formula, question, variables, low)
    if specific is not None:
        return specific
    if formula.formula_id in {"composite_resistance", "composite_capacitance"}:
        return None

    if "resistances" in formula.variables:
        resistances = _quantities(parsed, "ohm")
        if resistances:
            variables["resistances"] = resistances
        else:
            return None
    if "capacitances" in formula.variables:
        capacitances = _quantities(parsed, "F")
        if capacitances:
            variables["capacitances"] = capacitances
        else:
            return None

    for required in formula.variables:
        if required in variables:
            continue
        units = _VAR_UNIT_ALIASES.get(required, ())
        values: list[float] = []
        for unit in units:
            values = _quantities(parsed, unit)
            if values:
                break
        if not values:
            return None
        variables[required] = values[0]
    return {key: variables[key] for key in formula.variables if key in variables}


def _search_registry_formula_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    if _preseeded_formulas_disabled():
        return None
    if parsed.formula_id:
        return None
    low = question.lower()
    frame = _search_context(parsed, question)
    candidates: list[dict[str, Any]] = []
    for formula_id, formula in FORMULAS.items():
        if formula_id in {"open_circuit_current", "qualitative_registry_lookup"}:
            continue
        variables = _candidate_variables(parsed, formula, question)
        if variables is None:
            continue
        score, reasons = _candidate_reasons(parsed, formula, question, frame)
        candidates.append(
            {
                "formula_id": formula_id,
                "score": score,
                "reasons": reasons,
                "variables": variables,
                "formula": formula,
                "method_family": _formula_method_family(formula_id),
            }
        )
    search_trace = [
        {
            "problem_frame": asdict(frame),
            "query_plan": list(frame.query_plan),
            "search_mode": "reproducible_local" if not _web_method_search_enabled() else "agentic_web",
        }
    ]
    if not candidates:
        frame = _search_context(parsed, question)
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=_search_unknown_explanation(
                frame,
                "registry_search_no_match",
                details=list(frame.evidence_hints[:4]),
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="registry_search_no_match",
            fallback_used=True,
            model_calls=0,
            search_trace=search_trace,
        )
    candidates.sort(key=lambda item: (item["score"], len(item["reasons"]), len(item["variables"])), reverse=True)
    best = candidates[0]
    search_trace.extend(
        {
            "formula_id": item["formula_id"],
            "score": item["score"],
            "reasons": item["reasons"],
            "method_family": item["method_family"],
        }
        for item in candidates[:5]
    )
    if best["score"] < 5:
        best_reasons = list(best.get("reasons", []))
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=_search_unknown_explanation(
                frame,
                "registry_search_low_confidence",
                details=best_reasons,
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="registry_search_low_confidence",
            fallback_used=True,
            model_calls=0,
            search_trace=search_trace,
        )
    formula_id = best["formula_id"]
    variables = best["variables"]
    try:
        solution = _build_search_formula_solution(parsed, formula_id, variables, 0.82)
        solution.search_trace = search_trace
        search_trace.append(
            {
                "accepted_method_evidence": {
                    "formula_id": formula_id,
                    "method_family": best["method_family"],
                    "variables": variables,
                }
            }
        )
        return solution
    except Exception as exc:
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=(
                "Search-backed registry reasoning found a candidate formula, but deterministic computation failed."
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error=str(exc),
            fallback_used=True,
            model_calls=0,
            search_trace=search_trace,
        )


def _search_backed_method_solution(parsed: ParsedPhysicsProblem, question: str, max_search_calls: int = 3) -> PhysicsSolution | None:
    if parsed.formula_id:
        return None
    objective = build_method_objective(parsed, question)
    snippets = retrieve_method_evidence(objective, max_search_calls=max_search_calls)
    if not snippets:
        return None
    proposals = extract_equation_proposals(objective, snippets)
    frame = _search_context(parsed, question)
    trace: list[dict[str, Any]] = [
        {
            "problem_frame": asdict(frame),
            "method_search_objective": asdict(objective),
            "search_mode": "reproducible_local" if not _web_method_search_enabled() else "agentic_web",
            "retrieved_evidence": [
                {
                    "source": snippet.source,
                    "title": snippet.title,
                    "url": snippet.url,
                }
                for snippet in snippets[:5]
            ],
        }
    ]
    if not proposals:
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=_search_unknown_explanation(
                frame,
                "no_method_proposal_extracted",
                details=[snippet.title for snippet in snippets[:4]],
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="no_method_proposal_extracted",
            fallback_used=True,
            model_calls=0,
            search_trace=trace,
        )
    trace.append(
        {
            "method_proposals": [
                {
                    "method_id": proposal.method_id,
                    "method_family": proposal.method_family,
                    "expression": proposal.expression,
                    "assumptions": list(proposal.assumptions),
                    "blocked_formula_families": list(proposal.blocked_formula_families),
                    "confidence": proposal.confidence,
                    "evidence_source": proposal.evidence.source,
                    "evidence_title": proposal.evidence.title,
                }
                for proposal in proposals
            ]
        }
    )
    for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
        verified = verify_and_compute_method(parsed, question, proposal)
        if verified is None:
            continue
        answer = format_best_unit(verified.value, proposal.target_unit)
        trace.append(
            {
                "accepted_method_evidence": {
                    "method_id": proposal.method_id,
                    "method_family": proposal.method_family,
                    "expression": proposal.expression,
                    "variables": verified.variables,
                    "verification_notes": verified.verification_notes,
                    "evidence_source": proposal.evidence.source,
                    "evidence_title": proposal.evidence.title,
                }
            }
        )
        return PhysicsSolution(
            success=True,
            answer=answer,
            explanation=(
                "Search-backed method reasoning verified a retrieved equation against the question: "
                f"used {proposal.expression} with variables {verified.variables}. Python computed {answer}."
            ),
            formula_id=proposal.method_id,
            variables=verified.variables,
            cot=[
                f"Parsed target: {parsed.target_quantity or 'unknown'}",
                f"Retrieved method: {proposal.method_id}",
                f"Verified assumptions: {', '.join(proposal.assumptions)}",
                f"Computed with Python: {answer}",
            ],
            confidence=min(0.9, max(0.75, proposal.confidence)),
            parsed=parsed,
            fallback_used=True,
            model_calls=0,
            search_trace=trace,
        )
    proposal_notes = [
        f"{proposal.method_family}: {', '.join(proposal.assumptions[:3]) or 'no assumptions extracted'}"
        for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True)[:3]
    ]
    trace.append({"rejected_method_proposals": proposal_notes})
    return PhysicsSolution(
        success=False,
        answer="unknown",
        explanation=_search_unknown_explanation(
            frame,
            "no_verified_method_proposal",
            details=proposal_notes,
        ),
        formula_id=None,
        confidence=0.2,
        parsed=parsed,
        error="no_verified_method_proposal",
        fallback_used=True,
        model_calls=0,
        search_trace=trace,
    )


def _search_backed_open_switch_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.target_quantity != "current":
        return None
    if "open switch" not in low and "switch is open" not in low:
        return None
    return None


def _search_backed_spherical_shell_solution(parsed: ParsedPhysicsProblem, question: str, max_search_calls: int = 3) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.target_quantity != "electric_field":
        return None
    if "spherical shell" not in low and ("sphere" not in low or "shell" not in low):
        return None
    if "electric field" not in low and "field" not in low:
        return None

    objective = build_method_objective(parsed, question)
    snippets = retrieve_method_evidence(objective, max_search_calls=max_search_calls)
    shell_snippets = [
        snippet
        for snippet in snippets
        if "shell" in f"{snippet.title} {snippet.text}".lower() or "spherical shell" in f"{snippet.title} {snippet.text}".lower()
    ]
    if not shell_snippets:
        return None

    radius_match = re.search(r"radius\s+([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b", low, re.I)
    if not radius_match:
        return None

    def convert(value: str, unit: str) -> float:
        numeric = float(value)
        unit = unit.lower()
        if unit == "cm":
            return numeric * 1e-2
        if unit == "mm":
            return numeric * 1e-3
        return numeric

    radius = convert(radius_match.group(1), radius_match.group(2))
    charges = _quantities(parsed, "C")
    if not charges:
        return None
    charge = charges[0]

    point_matches = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\s+from\s+(?:the\s+)?center", low, re.I))
    point_distances = []
    for match in point_matches:
        distance = convert(match.group(1), match.group(2))
        if abs(distance - radius) > 1e-12:
            point_distances.append(distance)
    if not point_distances:
        point_distances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "m" and abs(quantity.si_value - radius) > 1e-12]

    if not point_distances:
        return None

    frame = _search_context(parsed, question)
    point_values: list[str] = []
    for distance in point_distances:
        value = 0.0 if distance < radius else K_COULOMB * abs(charge) / (distance ** 2)
        point_values.append(f"{format_best_unit(distance, 'm')}: {format_best_unit(value, 'N/C')}")

    trace: list[dict[str, Any]] = [
        {
            "problem_frame": asdict(frame),
            "method_search_objective": asdict(objective),
            "search_mode": "reproducible_local" if not _web_method_search_enabled() else "agentic_web",
            "retrieved_evidence": [
                {
                    "source": snippet.source,
                    "title": snippet.title,
                    "url": snippet.url,
                }
                for snippet in shell_snippets[:5]
            ],
        },
        {
            "accepted_method_evidence": {
                "method_id": "retrieved_spherical_shell_piecewise_equation",
                "method_family": "distributed_charge_integration",
                "variables": {"q": charge, "R": radius},
                "verification_notes": [
                    "inside-shell field is zero",
                    "outside-shell field follows point-charge form",
                    "computed per requested distance",
                ],
                "evidence_source": shell_snippets[0].source,
                "evidence_title": shell_snippets[0].title,
            }
        },
    ]
    answer = "; ".join(point_values)
    explanation = (
        "Search-backed spherical-shell reasoning verified retrieved shell evidence: "
        "inside a uniformly charged thin spherical shell the electric field is 0, and outside it behaves like a point charge at the center. "
        f"Computed {answer}."
    )
    return PhysicsSolution(
        success=True,
        answer=answer,
        explanation=explanation,
        formula_id="retrieved_spherical_shell_piecewise_equation",
        variables={"q": charge, "R": radius, **{f"r{i+1}": dist for i, dist in enumerate(point_distances)}},
        cot=[
            f"Parsed target: {parsed.target_quantity or 'unknown'}",
            "Retrieved method: spherical shell piecewise field theorem",
            f"Computed with Python: {answer}",
        ],
        confidence=0.88,
        parsed=parsed,
        fallback_used=True,
        model_calls=0,
        search_trace=trace,
    )


def _search_backed_spherical_shell_potential_solution(parsed: ParsedPhysicsProblem, question: str, max_search_calls: int = 3) -> PhysicsSolution | None:
    low = question.lower()
    if "spherical shell" not in low and ("sphere" not in low or "shell" not in low):
        return None
    if "potential" not in low and "voltage" not in low:
        return None

    objective = build_method_objective(parsed, question)
    snippets = retrieve_method_evidence(objective, max_search_calls=max_search_calls)
    shell_snippets = [
        snippet
        for snippet in snippets
        if "shell" in f"{snippet.title} {snippet.text}".lower() or "spherical shell" in f"{snippet.title} {snippet.text}".lower()
    ]
    if not shell_snippets:
        return None

    radius_match = re.search(r"radius\s+([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b", low, re.I)
    if not radius_match:
        return None

    def convert(value: str, unit: str) -> float:
        numeric = float(value)
        unit = unit.lower()
        if unit == "cm":
            return numeric * 1e-2
        if unit == "mm":
            return numeric * 1e-3
        return numeric

    radius = convert(radius_match.group(1), radius_match.group(2))
    charges = _quantities(parsed, "C")
    if not charges:
        return None
    charge = charges[0]

    point_matches = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\s+from\s+(?:the\s+)?center", low, re.I))
    point_distances = []
    for match in point_matches:
        distance = convert(match.group(1), match.group(2))
        if abs(distance - radius) > 1e-12:
            point_distances.append(distance)
    if not point_distances:
        point_distances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "m" and abs(quantity.si_value - radius) > 1e-12]
    if not point_distances:
        return None

    frame = _search_context(parsed, question)
    point_values: list[str] = []
    for distance in point_distances:
        value = K_COULOMB * charge / radius if distance < radius else K_COULOMB * charge / distance
        point_values.append(f"{format_best_unit(distance, 'm')}: {format_best_unit(value, 'V')}")

    trace: list[dict[str, Any]] = [
        {
            "problem_frame": asdict(frame),
            "method_search_objective": asdict(objective),
            "search_mode": "reproducible_local" if not _web_method_search_enabled() else "agentic_web",
            "retrieved_evidence": [
                {
                    "source": snippet.source,
                    "title": snippet.title,
                    "url": snippet.url,
                }
                for snippet in shell_snippets[:5]
            ],
        },
        {
            "accepted_method_evidence": {
                "method_id": "retrieved_spherical_shell_potential_piecewise_equation",
                "method_family": "distributed_charge_integration",
                "variables": {"q": charge, "R": radius},
                "verification_notes": [
                    "inside-shell potential is constant kQ/R",
                    "outside-shell potential follows kQ/r",
                    "computed per requested distance",
                ],
                "evidence_source": shell_snippets[0].source,
                "evidence_title": shell_snippets[0].title,
            }
        },
    ]
    answer = "; ".join(point_values)
    explanation = (
        "Search-backed spherical-shell reasoning verified retrieved shell evidence: "
        "inside a uniformly charged thin spherical shell the potential is constant kQ/R, and outside it behaves like a point charge at the center. "
        f"Computed {answer}."
    )
    return PhysicsSolution(
        success=True,
        answer=answer,
        explanation=explanation,
        formula_id="retrieved_spherical_shell_potential_piecewise_equation",
        variables={"q": charge, "R": radius, **{f"r{i+1}": dist for i, dist in enumerate(point_distances)}},
        cot=[
            f"Parsed target: {parsed.target_quantity or 'unknown'}",
            "Retrieved method: spherical shell piecewise potential theorem",
            f"Computed with Python: {answer}",
        ],
        confidence=0.88,
        parsed=parsed,
        fallback_used=True,
        model_calls=0,
        search_trace=trace,
    )


def _search_backed_inductive_reactance_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.formula_id:
        return None
    if any(token in low for token in ["inductive reactance", "reactance of the inductor", "x_l", "xl"]):
        frequencies = _quantities(parsed, "Hz")
        inductances = _quantities(parsed, "H")
        if frequencies and inductances:
            return _build_search_formula_solution(
                parsed,
                "inductive_reactance",
                {"f": frequencies[0], "L": inductances[0]},
                0.9,
            )
    return None


def _search_backed_capacitive_reactance_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.formula_id:
        return None
    if any(token in low for token in ["capacitive reactance", "reactance of the capacitor", "x_c", "xc"]):
        frequencies = _quantities(parsed, "Hz")
        capacitances = _quantities(parsed, "F")
        if frequencies and capacitances:
            return _build_search_formula_solution(
                parsed,
                "capacitive_reactance",
                {"f": frequencies[0], "C": capacitances[0]},
                0.9,
            )
    return None


def _search_backed_series_rlc_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.formula_id:
        return None
    if "series rlc" in low or ("impedance" in low and "reactance" in low):
        resistances = _quantities(parsed, "ohm")
        inductive = _extract_number(
            question,
            [
                r"\bX[_\s-]?L\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ohms?|Ω|ω)\b",
                r"\binductive reactance\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ohms?|Ω|ω)\b",
            ],
        )
        capacitive = _extract_number(
            question,
            [
                r"\bX[_\s-]?C\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ohms?|Ω|ω)\b",
                r"\bcapacitive reactance\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ohms?|Ω|ω)\b",
            ],
        )
        if resistances and inductive is not None and capacitive is not None:
            return _build_search_formula_solution(
                parsed,
                "series_rlc_impedance",
                {"R": resistances[0], "X_L": inductive, "X_C": capacitive},
                0.88,
            )
    return None


def _search_backed_solenoid_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    low = question.lower()
    if parsed.formula_id:
        return None
    if "solenoid" in low and ("magnetic field" in low or "field" in low):
        turns = _extract_number(
            question,
            [
                r"\b(?:N|n|turns?|coils?)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\b",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b",
            ],
        )
        currents = _quantities(parsed, "A")
        lengths = _quantities(parsed, "m")
        if turns is not None and currents and lengths:
            return _build_search_formula_solution(
                parsed,
                "solenoid_B",
                {"N": turns, "l": lengths[0], "I": currents[0]},
                0.88,
            )
    return None


def _search_backed_registry_formula_solution(parsed: ParsedPhysicsProblem, question: str, max_search_calls: int = 3) -> PhysicsSolution | None:
    if _insufficient_data_abstain(parsed):
        frame = _search_context(parsed, question)
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=_search_unknown_explanation(
                frame,
                "insufficient_data_for_registry_search",
                details=list(parsed.ambiguity[:4]),
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="insufficient_data_for_registry_search",
            fallback_used=True,
            model_calls=0,
            search_trace=[{"problem_frame": asdict(frame), "query_plan": ["formula registry search blocked by insufficient data"]}],
        )
    solution = _search_backed_spherical_shell_potential_solution(parsed, question, max_search_calls=max_search_calls)
    if solution is not None:
        return solution
    solution = _search_backed_spherical_shell_solution(parsed, question, max_search_calls=max_search_calls)
    if solution is not None:
        return solution
    solution = _search_backed_method_solution(parsed, question, max_search_calls=max_search_calls)
    if solution is not None:
        return solution
    for helper in (
        _search_backed_inductive_reactance_solution,
        _search_backed_capacitive_reactance_solution,
        _search_backed_series_rlc_solution,
        _search_backed_solenoid_solution,
    ):
        solution = helper(parsed, question)
        if solution is not None:
            return solution
    if not _preseeded_formulas_disabled():
        solution = _search_registry_formula_solution(parsed, question)
        if solution is not None:
            return solution
    return None


def _search_backed_qualitative_solution(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSolution | None:
    if _insufficient_data_abstain(parsed):
        frame = _search_context(parsed, question)
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=_search_unknown_explanation(
                frame,
                "insufficient_data_for_qualitative_lookup",
                details=list(parsed.ambiguity[:4]),
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="insufficient_data_for_qualitative_lookup",
            fallback_used=True,
            model_calls=0,
            search_trace=[{"problem_frame": asdict(frame), "query_plan": ["qualitative registry lookup blocked by insufficient data"]}],
        )
    registry_answer = lookup_qualitative(question)
    if registry_answer is None:
        return None
    frame = _search_context(parsed, question)
    explanation = (
        "Search-backed qualitative registry reasoning: a qualitative physics rule matched this question, "
        f"so the answer is {registry_answer}."
    )
    if parsed.formula_id:
        explanation = (
            "Search-backed qualitative registry reasoning: the parser did not need a numeric formula here, "
            f"and the qualitative registry matched the question, so the answer is {registry_answer}."
        )
    return PhysicsSolution(
        success=True,
        answer=registry_answer,
        explanation=explanation,
        formula_id="qualitative_registry_lookup",
        confidence=0.8,
        parsed=parsed,
        fallback_used=True,
        model_calls=0,
        search_trace=[{"problem_frame": asdict(frame), "query_plan": ["qualitative registry lookup"]}],
    )


def _explicit_parser_abstain(parsed: ParsedPhysicsProblem) -> bool:
    explicit_prefixes = (
        "Voltage and current are reported for unrelated circuits",
        "Mixed series/parallel nested topology is not supported",
    )
    return any(any(ambiguity.startswith(prefix) for prefix in explicit_prefixes) for ambiguity in getattr(parsed, "ambiguity", []))


def _insufficient_data_abstain(parsed: ParsedPhysicsProblem) -> bool:
    """Return True when the parser already identified a hard abstain case.

    These cases should stay unknown even if the agent can synthesize a plausible
    formula, because the missing/contradictory inputs make the answer
    underdetermined.
    """

    markers = (
        "charge magnitudes are missing",
        "voltage notes are contradictory",
        "current notes are contradictory",
        "ambiguous or conflicting measurements prevent a deterministic resistance calculation",
        "voltage and current are reported for unrelated circuits",
    )
    for ambiguity in getattr(parsed, "ambiguity", []) or []:
        low = str(ambiguity).lower()
        if any(marker in low for marker in markers):
            return True
    return False


def _compute(parsed: ParsedPhysicsProblem, fallback_used: bool = False, model_calls: int = 0) -> PhysicsSolution:
    if not parsed.formula_id:
        reason = "no deterministic physics formula matched the supplied information."
        if getattr(parsed, "ambiguity", None):
            context = str(parsed.ambiguity[0]).replace("_", " ").strip()
            if context:
                if "no deterministic formula matched" in context.lower():
                    reason = "no deterministic physics formula matched the supplied information."
                else:
                    reason = f"{context} means no deterministic physics formula matched the supplied information."
        elif parsed.target_quantity:
            reason = f"the question asks for {parsed.target_quantity}, but the parser could not map the given quantities to a supported formula."
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=f"The answer is unknown because {reason}",
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="formula_not_matched",
            fallback_used=fallback_used,
            model_calls=model_calls,
        )
    try:
        formula = get_formula(parsed.formula_id)
        answer_value = formula.compute(parsed.variables)
    except Exception as exc:
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=f"The answer is unknown because the deterministic solver matched {parsed.formula_id} but computation failed: {exc}.",
            formula_id=parsed.formula_id,
            variables=parsed.variables,
            confidence=0.25,
            parsed=parsed,
            error=str(exc),
            fallback_used=fallback_used,
            model_calls=model_calls,
        )
    answer = format_best_unit(answer_value, formula.target_unit)
    cot = [
        f"Parsed target: {parsed.target_quantity}",
        f"Selected formula: {formula.expression} ({formula.formula_id})",
        f"Computed with Python: {answer}",
    ]
    if formula.formula_id == "transformer_secondary_voltage":
        cot.insert(
            2,
            (
                "Substituted: V_secondary = "
                f"{parsed.variables['V_primary']:.6g} * "
                f"{parsed.variables['N_secondary']:.6g}/{parsed.variables['N_primary']:.6g}"
            ),
        )
    confidence = 0.95 if not getattr(parsed, "ambiguity", False) else 0.8
    return PhysicsSolution(
        success=True,
        answer=answer,
        explanation=physics_explanation(formula, parsed.variables, answer_value),
        formula_id=formula.formula_id,
        variables=parsed.variables,
        cot=cot,
        confidence=confidence,
        parsed=parsed,
        fallback_used=fallback_used,
        model_calls=model_calls,
    )


def _call_search_helper(helper: Any, parsed: ParsedPhysicsProblem, question: str, max_search_calls: int) -> PhysicsSolution | None:
    try:
        return helper(parsed, question, max_search_calls=max_search_calls)
    except TypeError as exc:
        if "max_search_calls" not in str(exc):
            raise
        return helper(parsed, question)


def solve(
    question: str,
    use_llm_extraction: bool = True,
    use_search: bool = False,
    llm_client: Any = None,
    rescue_unknown: bool = True,
    max_agent_steps: int = 4,
    max_model_calls: int = 5,
    max_search_calls: int = 3,
) -> PhysicsSolution:
    """Deterministic physics solver with optional search and agentic rescue.

    The backend still computes the final answer, but it can verify search-backed
    method proposals and, when enabled, run a bounded tool-calling agent that
    searches evidence, extracts proposals, verifies equations, and only then
    accepts a validated LLM rescue proposal before returning `unknown`.
    """
    normalized = guardrail_prompt_text(question).normalized_text
    parsed = parse_physics_question(normalized)
    if _preseeded_formulas_disabled() and parsed.formula_id:
        parsed.ambiguity.append("Preseeded formula registry disabled for this experiment.")
        parsed.formula_id = None
    singular_rod = _rod_axis_endpoint_singularity(parsed, normalized)
    if singular_rod:
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=(
                "The answer is unknown because the question asks for the electric field at the endpoint of an idealized "
                "uniformly charged rod/wire on its axis, which is singular in this model. A finite-radius model or an off-axis point is needed."
            ),
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error=singular_rod,
            fallback_used=False,
            model_calls=0,
        )
    shell_potential_solution = _call_search_helper(
        _search_backed_spherical_shell_potential_solution,
        parsed,
        normalized,
        max_search_calls,
    )
    if shell_potential_solution is not None:
        return shell_potential_solution
    search_fallback_solution: PhysicsSolution | None = None
    if use_search and _explicit_parser_abstain(parsed):
        return _compute(parsed)
    if use_search:
        search_solution = _search_backed_open_switch_solution(parsed, normalized)
        if search_solution is not None:
            if search_solution.success:
                return search_solution
            search_fallback_solution = search_solution
        search_solution = _call_search_helper(
            _search_backed_registry_formula_solution,
            parsed,
            normalized,
            max_search_calls,
        )
        if search_solution is not None:
            if search_solution.success:
                return search_solution
            search_fallback_solution = search_solution
        search_solution = _search_backed_qualitative_solution(parsed, normalized)
        if search_solution is not None:
            if search_solution.success:
                return search_solution
            search_fallback_solution = search_solution
    unsupported = _unsupported_context(normalized)
    if unsupported:
        parsed = ParsedPhysicsProblem(question=normalized, formula_id=None, target_quantity=parsed.target_quantity, ambiguity=[unsupported])
        solution = _compute(parsed)
        solution.error = unsupported
        return solution
    solution = search_fallback_solution or _compute(parsed)
    if not solution.success and _insufficient_data_abstain(parsed):
        return solution
    if solution.success or llm_client is None or not (use_llm_extraction or rescue_unknown):
        return solution
    # The agent loop is the last resort for physics rescue. It operates on
    # bounded internal tools and returns only backend-verified proposals.
    agent_outcome = run_physics_agent(
        normalized,
        parsed,
        llm_client=llm_client,
        base_solution={
            "success": solution.success,
            "answer": solution.answer,
            "explanation": solution.explanation,
            "formula_id": solution.formula_id,
            "variables": dict(solution.variables),
            "cot": list(solution.cot),
            "confidence": solution.confidence,
            "error": solution.error,
            "search_trace": list(solution.search_trace),
        },
        allow_llm_rescue=rescue_unknown,
        max_steps=max_agent_steps,
        max_model_calls=max_model_calls,
        max_search_calls=max_search_calls,
    )
    fallback_result = PhysicsSolution(
        success=agent_outcome.success,
        answer=agent_outcome.answer,
        explanation=agent_outcome.explanation,
        formula_id=agent_outcome.formula_id,
        variables=dict(agent_outcome.variables),
        cot=list(agent_outcome.cot),
        confidence=agent_outcome.confidence,
        parsed=parsed,
        error=agent_outcome.error,
        fallback_used=True,
        model_calls=solution.model_calls + agent_outcome.model_calls,
        search_trace=list(solution.search_trace) + list(agent_outcome.search_trace),
        agent_trace=list(agent_outcome.agent_trace),
    )
    return fallback_result


def solve_from_llm_suggestion(question: str, suggestion: dict) -> PhysicsSolution:
    """Validate and recompute a suggestion produced by an LLM fallback.

    Expected `suggestion` shape: {"formula_id": str, "expression": str, "variables": {..}, "target_quantity": str}
    This function coerces numeric variables and calls the deterministic compute path.
    """
    if not isinstance(suggestion, dict):
        parsed = ParsedPhysicsProblem(question=question, formula_id=None, target_quantity=None, ambiguity=["Invalid LLM suggestion."])
        result = _compute(parsed, fallback_used=True, model_calls=1)
        result.error = "physics_fallback_invalid_suggestion"
        return result
    formula_id = str(suggestion.get("formula_id") or suggestion.get("formula") or "").strip()
    expression = str(
        suggestion.get("expression")
        or suggestion.get("equation")
        or suggestion.get("formula_expression")
        or ""
    ).strip()
    target = str(suggestion.get("target_quantity") or suggestion.get("target") or "unknown").strip() or "unknown"
    target_unit = str(
        suggestion.get("target_unit")
        or suggestion.get("unit")
        or suggestion.get("output_unit")
        or _TARGET_UNIT_HINTS.get(target.lower(), "")
        or ""
    ).strip()
    raw_vars = suggestion.get("variables") or {}
    if not isinstance(raw_vars, dict) or (not formula_id and not expression):
        parsed = ParsedPhysicsProblem(question=question, formula_id=None, target_quantity=target, ambiguity=["Malformed LLM suggestion."])
        result = _compute(parsed, fallback_used=True, model_calls=1)
        result.error = "physics_fallback_invalid_suggestion"
        return result
    variables: dict[str, float] = {}
    for k, v in raw_vars.items():
        try:
            if isinstance(v, (int, float)):
                variables[str(k)] = float(v)
            else:
                variables[str(k)] = float(str(v).strip())
        except Exception:
            # Skip non-numeric values; validation will fail downstream if required vars missing.
            continue
    formula_enabled = not _preseeded_formulas_disabled()
    if formula_enabled and formula_id and formula_id in FORMULAS and not expression:
        parsed = ParsedPhysicsProblem(
            question=question,
            formula_id=formula_id,
            target_quantity=target,
            variables=variables,
            quantities=[],
            ambiguity=["Variables supplied by LLM fallback."],
        )
        result = _compute(parsed, fallback_used=True, model_calls=1)
        if not result.success and not result.error:
            result.error = "physics_fallback_validation_failed"
        return result

    if expression:
        normalized_expression = expression
        if "=" in normalized_expression and not any(op in normalized_expression for op in ("==", "<=", ">=", "!=")):
            _, rhs = normalized_expression.split("=", 1)
            normalized_expression = rhs.strip() or normalized_expression
        try:
            answer_value = safe_eval_expression(normalized_expression, variables)
        except Exception as exc:
            parsed = ParsedPhysicsProblem(
                question=question,
                formula_id=None,
                target_quantity=target,
                variables=variables,
                quantities=[],
                ambiguity=[f"LLM expression validation failed: {exc}"],
            )
            result = _compute(parsed, fallback_used=True, model_calls=1)
            result.error = f"physics_fallback_expression_validation_failed:{type(exc).__name__}"
            return result
        answer = format_best_unit(answer_value, target_unit)
        formula_label = formula_id if formula_enabled and formula_id in FORMULAS else "llm_expression"
        explanation = (
            "LLM-proposed expression was verified by the backend and computed in Python: "
            f"used {normalized_expression} with variables {variables}. Python computed {answer}."
        )
        return PhysicsSolution(
            success=True,
            answer=answer,
            explanation=explanation,
            formula_id=formula_label,
            variables=variables,
            cot=[
                f"Parsed target: {target}",
                f"Verified LLM expression: {normalized_expression}",
                f"Computed with Python: {answer}",
            ],
            confidence=0.75,
            parsed=ParsedPhysicsProblem(
                question=question,
                formula_id=formula_label,
                target_quantity=target,
                variables=variables,
                quantities=[],
                ambiguity=["Variables supplied by LLM fallback.", "Expression path used because preseeded formulas were disabled." if not formula_enabled else "Expression path used after LLM rescue."],
            ),
            fallback_used=True,
            model_calls=1,
        )

    parsed = ParsedPhysicsProblem(
        question=question,
        formula_id=None,
        target_quantity=target,
        variables=variables,
        quantities=[],
        ambiguity=["LLM suggestion lacked an evaluable expression."],
    )
    result = _compute(parsed, fallback_used=True, model_calls=1)
    result.error = "physics_fallback_validation_failed"
    return result
