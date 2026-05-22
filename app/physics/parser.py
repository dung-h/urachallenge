from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.physics.unit_converter import Quantity, extract_quantities, normalize_number_words


@dataclass
class ParsedPhysicsProblem:
    question: str
    formula_id: str | None
    target_quantity: str | None
    variables: dict[str, float] = field(default_factory=dict)
    quantities: list[Quantity] = field(default_factory=list)
    ambiguity: list[str] = field(default_factory=list)


def _values(quantities: list[Quantity], si_unit: str) -> list[float]:
    return [q.si_value for q in quantities if q.si_unit == si_unit]


def _parallel_resistance(values: list[float]) -> float:
    return 1.0 / sum(1.0 / value for value in values)


def _composite_resistance(question: str) -> float | None:
    text = normalize_number_words(question.lower()).replace(",", " ")
    text = re.sub(r"\s+", " ", text)
    num = r"([0-9]+(?:\.[0-9]+)?)"

    # Pattern: "4 ohm resistor is in series with a parallel combination of 6 ohm and 3 ohm resistors"
    match = re.search(rf"{num}\s*ohms?\s+resistor\s+is\s+in\s+series\s+with\s+a\s+parallel\s+combination\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?", text)
    if match:
        series_branch, first, second = (float(value) for value in match.groups())
        return series_branch + _parallel_resistance([first, second])

    match = re.search(rf"{num}\s*ohms?\s+and\s+{num}\s*ohms?\s+series\s+pair\s+is\s+in\s+parallel\s+with\s+{num}\s*ohms?", text)
    if match:
        first, second, parallel_branch = (float(value) for value in match.groups())
        return _parallel_resistance([first + second, parallel_branch])

    match = re.search(rf"{num}\s*ohms?\s+resistor\s+is\s+in\s+series\s+with\s+a\s+parallel\s+pair\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?", text)
    if match:
        series_branch, first, second = (float(value) for value in match.groups())
        return series_branch + _parallel_resistance([first, second])

    match = re.search(rf"{num}\s*ohms?\s+resistor\s+is\s+in\s+parallel\s+with\s+a\s+series\s+branch\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?", text)
    if match:
        parallel_branch, first, second = (float(value) for value in match.groups())
        return _parallel_resistance([parallel_branch, first + second])

    match = re.search(rf"{num}\s*ohms?\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?\s+chain\s+has\s+the\s+last\s+(?:two|2)\s+in\s+parallel\s+after\s+the\s+first", text)
    if match:
        first, second, third = (float(value) for value in match.groups())
        return first + _parallel_resistance([second, third])

    match = re.search(rf"{num}\s+{num}\s*ohms?\s+resistors\s+in\s+series\s+are\s+in\s+parallel\s+with\s+{num}\s*ohms?", text)
    if match:
        count, value, parallel_branch = (float(value) for value in match.groups())
        return _parallel_resistance([count * value, parallel_branch])

    match = re.search(rf"{num}\s*ohms?\s+resistor\s+is\s+in\s+series\s+with\s+{num}\s+{num}\s*ohms?\s+resistors\s+in\s+parallel", text)
    if match:
        series_branch, count, value = (float(value) for value in match.groups())
        return series_branch + _parallel_resistance([value] * int(count))

    return None


def _target(text: str) -> str | None:
    low = text.lower()
    resonance_context = "resonance" in low or "resonant" in low or "resonate" in low
    if resonance_context and (
        "angular frequency" in low
        or "angular resonant frequency" in low
        or "angular resonance frequency" in low
        or "omega" in low
        or "ω" in text
        or "rad/s" in low
    ):
        return "angular_frequency"
    if any(phrase in low for phrase in ["resonant frequency", "resonance frequency", "natural frequency", "oscillation frequency"]):
        return "frequency"
    if "energy density" in low:
        return "energy_density"
    # Dielectric constant / relative permittivity is dimensionless. Only treat as target
    # when the question explicitly asks for it (otherwise it's often just given data).
    if any(
        phrase in low
        for phrase in [
            "what is the dielectric constant",
            "find the dielectric constant",
            "calculate the dielectric constant",
            "determine the dielectric constant",
            "what is the relative permittivity",
            "find the relative permittivity",
            "calculate the relative permittivity",
            "determine the relative permittivity",
        ]
    ):
        return "dielectric_constant"
    # Distinguish electric field from magnetic field and electric field energy
    if "electric field energy" in low:
        return "energy"
    if "electric field" in low or "electric intensity" in low:
        return "electric_field"
    # Do not use generic "field" token - too ambiguous (magnetic field, etc)
    if "separation distance" in low or re.search(r"\bdistance\b", low):
        return "distance"
    if "force" in low:
        return "force"
    if any(token in low for token in ["return v", "what is v", "find v", "potential drop", "emf"]):
        return "voltage"
    if "current" in low and (
        re.search(r"\bwhat\s+is\s+(?:the\s+)?current\b", low)
        or re.search(r"\bfind\s+(?:the\s+)?current\b", low)
        or any(token in low for token in ["what current", "find the current"])
    ):
        return "current"
    if re.search(r"\bwhat is i\b", low):
        return "current"
    if "impedance" in low and "resonance" in low:
        return "resistance"
    if "resistance" in low and any(token in low for token in ["what is the resistance", "what resistance", "has what resistance", "find the resistance"]):
        return "resistance"
    if "compute vi" in low or "rate of energy use" in low or re.search(r"\b(?:what is|find|return)\s+p\b", low):
        return "power"
    # Distinguish "potential energy" (U = qV) from "energy stored" (capacitor/inductor)
    # Check "potential energy" first before generic "energy"
    if "potential energy" in low and "capacitor" not in low:
        return "potential_energy"
    if "energy" in low and ("stored" in low or "stores" in low):
        return "energy"
    if "capacitor" in low and "work" in low:
        return "energy"
    if "power" in low:
        return "power"
    if "voltage" in low or re.search(r"\bpotential difference\b", low):
        return "voltage"
    if "total resistance" in low or "equivalent resistance" in low or "effective resistance" in low or "parallel resistance" in low or "series resistance" in low:
        return "resistance"
    # Fix charge target: capacitor problems often say "is charged to U=..." which should not block charge detection
    if "charge" in low or "stored charge" in low or "charge accumulated" in low:
        # Only block if "charged" appears in the actual question part (after "what")
        question_part = low.split("what")[-1] if "what" in low else low
        if "charged to" not in question_part and "charged" not in question_part:
            return "charge"
        # Allow charge target even if "charged to" appears in setup, as long as question asks for charge
        if any(phrase in low for phrase in ["what is the charge", "find the charge", "calculate the charge", "determine the charge"]):
            return "charge"
    if ("capacitance" in low or re.search(r"\bwhat\s+value\s+of\s+c\b", low) or re.search(r"\bfind\s+c\b", low)) and resonance_context:
        return "capacitance"
    if "inductance" in low or (re.search(r"\bwhat\s+value\s+of\s+l\b", low) and resonance_context):
        return "inductance"
    if "capacitance" in low:
        return "capacitance"
    if "conductance puzzle" in low and "current" in low and "resistance" in low:
        return "voltage"
    return None


def parse_physics_question(question: str) -> ParsedPhysicsProblem:
    quantities = extract_quantities(question)
    low = question.lower()
    target = _target(question)
    variables: dict[str, float] = {}
    formula_id: str | None = None
    ambiguity: list[str] = []

    currents = _values(quantities, "A")
    voltages = _values(quantities, "V")
    resistances = _values(quantities, "ohm")
    capacitances = _values(quantities, "F")
    charges = _values(quantities, "C")
    distances = _values(quantities, "m")
    forces = _values(quantities, "N")
    powers = _values(quantities, "W")
    energies = _values(quantities, "J")
    inductances = _values(quantities, "H")
    magnetic_fields = _values(quantities, "T")
    fluxes = _values(quantities, "Wb")
    areas = _values(quantities, "m²")

    mixed_composite = ("series" in low and "parallel" in low) or ("parallel" in low and any(token in low for token in ["chain", "after the first", "branch"]))
    composite_resistance = _composite_resistance(question) if target == "resistance" and mixed_composite else None

    resonance_context = "resonance" in low or "resonant" in low or "resonate" in low
    if target == "angular_frequency" and resonance_context and inductances and capacitances:
        variables.update(L=inductances[0], C=capacitances[0])
        formula_id = "rlc_angular_resonant_frequency"
    elif target == "frequency" and resonance_context and inductances and capacitances:
        variables.update(L=inductances[0], C=capacitances[0])
        formula_id = "rlc_resonant_frequency"
    elif target == "capacitance" and resonance_context and inductances:
        frequencies = _values(quantities, "Hz")
        if frequencies:
            variables.update(f=frequencies[0], L=inductances[0])
            formula_id = "rlc_capacitance_from_resonant_frequency"
    elif target == "inductance" and resonance_context and capacitances:
        frequencies = _values(quantities, "Hz")
        if frequencies:
            variables.update(f=frequencies[0], C=capacitances[0])
            formula_id = "rlc_inductance_from_resonant_frequency"
    elif target == "resistance" and resonance_context and resistances and "impedance" in low:
        variables.update(R=resistances[0])
        formula_id = "rlc_resonance_resistance"

    # Resultant force detection
    if target == "force" and len(forces) == 2:
        # Check for angle information
        angle_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", low)
        if angle_match:
            theta_deg = float(angle_match.group(1))
            theta_rad = math.radians(theta_deg)
            variables.update(F1=forces[0], F2=forces[1], theta_rad=theta_rad)
            formula_id = "resultant_force_angle"
        elif "perpendicular" in low or "right angle" in low or "90" in low:
            variables.update(F1=forces[0], F2=forces[1])
            formula_id = "resultant_force_perpendicular"
        elif "opposite direction" in low or "opposite" in low:
            # Fix: opposite direction should use opposite formula, not same
            variables.update(F1=forces[0], F2=forces[1])
            formula_id = "resultant_force_collinear_opposite"
        elif "same direction" in low or "collinear" in low:
            variables.update(F1=forces[0], F2=forces[1])
            formula_id = "resultant_force_collinear_same"

    # Measurement error detection
    if "error" in low or "least count" in low or "uncertainty" in low:
        if "absolute error" in low and currents:
            # Absolute error from least count
            if len(currents) >= 2:
                # Assume smallest value is the least count
                least_count = min(currents)
                variables.update(least_count=least_count)
                formula_id = "measurement_absolute_error"
        elif "relative error" in low and len(currents) >= 2:
            # Relative error calculation
            # Pattern: reading and error value
            reading = max(currents)
            absolute_error = min(currents)
            variables.update(absolute_error=absolute_error, reading=reading)
            formula_id = "measurement_relative_error"
        elif "error" in low and "resistance" in low and len(voltages) >= 2 and len(currents) >= 2:
            # Error propagation for R = U/I
            # Assume format: U ± dU, I ± dI
            U = max(voltages)
            dU = min(voltages)
            I = max(currents)
            dI = min(currents)
            variables.update(dU=dU, U=U, dI=dI, I=I)
            formula_id = "measurement_error_propagation_division"

    # Dielectric handling
    # 1) Capacitor transforms when a dielectric is inserted/immersed/removed, with "connected" vs "disconnected".
    # 2) Electric field scaling in a dielectric medium: E = (k/eps_r) * q / r^2.
    # Keep this deterministic and conservative: only match when wording is explicit.

    # Extract relative permittivity / dielectric constant eps_r.
    # Avoid treating Coulomb's constant k (9e9) or vacuum permittivity epsilon_0 as eps_r.
    eps: float | None = None
    # Unambiguous variable forms: eps_r / epsilon_r / ε_r
    eps_match = re.search(r"\b(?:eps_r|epsilon_r|ε_r)\b\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if not eps_match:
        # Phrase-driven forms: dielectric constant / relative permittivity
        eps_match = re.search(r"(?:dielectric constant|relative permittivity)\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if not eps_match and "dielectric" in low:
        # Dielectric context + epsilon/eps/kappa assignment
        eps_match = re.search(r"(?:eps|epsilon|ε|κ)\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if eps_match:
        try:
            eps = float(eps_match.group(1))
        except Exception:
            eps = None

    dielectric_context = ("dielectric" in low) or ("relative permittivity" in low) or ("dielectric constant" in low)
    dielectric_event = dielectric_context and any(
        token in low for token in ["inserted", "introduced", "removed", "immersed", "immersed in", "filled", "replaced"]
    )

    # Parallel-plate capacitor in a dielectric medium (no insertion/removal event required).
    # Capacitance: C = eps0 * eps_r * A / d
    if target == "capacitance" and eps is not None and areas and distances and ("capacitor" in low and "plate" in low):
        variables.update(A=areas[0], d=distances[0], eps=eps)
        formula_id = "parallel_plate_capacitance_dielectric"

    # Dielectric constant inference from parallel-plate capacitance.
    if target == "dielectric_constant" and capacitances and areas and distances and ("capacitor" in low and "plate" in low):
        variables.update(C=capacitances[0], d=distances[0], A=areas[0])
        formula_id = "dielectric_constant_from_parallel_plate"

    # Electric field energy density in a dielectric-filled parallel-plate capacitor.
    if target == "energy_density" and eps is not None and voltages and distances and ("capacitor" in low and "plate" in low):
        variables.update(eps=eps, V=voltages[0], d=distances[0])
        formula_id = "energy_density_dielectric"

    if eps is not None and dielectric_event:
        disconnected = any(token in low for token in ["disconnected", "isolated", "battery removed", "source removed", "disconnected from the source"])
        connected = any(
            token in low
            for token in [
                "still connected",
                "remains connected",
                "kept connected",
                "connected to the source",
                "connected to a voltage source",
                "connected to the voltage source",
            ]
        )

        # Potential difference after dielectric insertion
        if target == "voltage" and voltages and disconnected:
            variables.update(V_old=voltages[0], eps=eps)
            formula_id = "dielectric_voltage_disconnected"
        elif target == "voltage" and voltages and connected:
            variables.update(V_source=voltages[0])
            formula_id = "dielectric_voltage_connected"

        # Electric field energy after dielectric insertion
        if target == "energy" and capacitances and voltages and "capacitor" in low and disconnected:
            variables.update(C=capacitances[0], V=voltages[0], eps=eps)
            formula_id = "dielectric_energy_disconnected"
        elif target == "energy" and capacitances and voltages and "capacitor" in low and connected:
            variables.update(C=capacitances[0], V=voltages[0], eps=eps)
            formula_id = "dielectric_energy_connected"

    # Electric field in a dielectric medium: scale k by 1/eps.
    if target == "electric_field" and eps is not None and dielectric_context:
        # Only apply if the problem states a charge and a distance, and doesn't look like multi-charge geometry.
        if charges and distances:
            multi_charge_keywords = ["q1", "q2", "q3", "two charges", "three charges", "multiple charges", "midpoint", "perpendicular bisector", "triangle"]
            has_multi_charge = any(kw in low for kw in multi_charge_keywords) or len(charges) > 1
            if not has_multi_charge and len(charges) == 1 and len(distances) >= 1:
                variables.update(q=charges[0], r=distances[0], eps=eps)
                formula_id = "electric_field_kq_r2_in_dielectric"

    # Energy scaling when the capacitor is disconnected and permittivity changes by a factor.
    # Example: "permittivity increases by a factor of 3" with an initial energy given.
    if target == "energy" and eps is None and energies and any(token in low for token in ["dielectric", "permittivity"]):
        factor_match = re.search(r"(?:increases|decreases)\s+by\s+(?:a\s+factor\s+of\s+)?(\d+(?:\.\d+)?)", low)
        if factor_match and any(token in low for token in ["disconnected", "isolated", "battery removed", "source removed"]):
            try:
                eps_factor = float(factor_match.group(1))
            except Exception:
                eps_factor = None
            if eps_factor and eps_factor > 0:
                variables.update(E_old=energies[0], eps=eps_factor)
                formula_id = "dielectric_energy_from_energy_disconnected"

    # Magnetism: inductor energy
    if "inductor" in low and target == "energy" and inductances and currents:
        variables.update(L=inductances[0], I=currents[0])
        formula_id = "inductor_energy"
    elif "inductor" in low and target == "current" and inductances and energies:
        variables.update(E=energies[0], L=inductances[0])
        formula_id = "inductor_current"
    elif "inductor" in low and "inductance" in low and energies and currents:
        variables.update(E=energies[0], I=currents[0])
        formula_id = "inductor_inductance"

    # Magnetism: solenoid flux
    if "solenoid" in low and ("flux" in low or "magnetic flux" in low):
        # Extract number of turns N
        n_match = re.search(r"(\d+)\s*(?:turns?|coils?)", low)
        if n_match and magnetic_fields and areas:
            N = float(n_match.group(1))
            variables.update(N=N, B=magnetic_fields[0], A=areas[0])
            formula_id = "solenoid_total_flux"

    # Skip to existing logic if already matched
    if formula_id:
        pass
    elif target == "resistance" and composite_resistance is not None:
        variables.update(R_total=composite_resistance, resistances=resistances)
        formula_id = "composite_resistance"
    elif target in {"resistance", "capacitance"} and mixed_composite:
        ambiguity.append("Mixed series/parallel composite circuit is not supported by deterministic parser.")
    elif target == "resistance" and resistances and ("parallel" in low or "series" in low):
        variables["resistances"] = resistances
        formula_id = "parallel_resistance" if "parallel" in low else "series_resistance" if "series" in low else None
    elif target == "resistance" and "heat" in low:
        ambiguity.append("Question includes an unsupported heat target/distractor.")
    elif target == "resistance":
        if voltages and currents:
            variables.update(V=voltages[0], I=currents[0])
            formula_id = "ohms_law_r_v_over_i"
        elif powers and currents:
            variables.update(P=powers[0], I=currents[0])
            formula_id = "power_r_p_over_i2"
        elif voltages and powers:
            variables.update(V=voltages[0], P=powers[0])
            formula_id = "power_r_v2_over_p"
    elif target == "capacitance" and capacitances and ("parallel" in low or "series" in low):
        variables["capacitances"] = capacitances
        formula_id = "series_capacitance" if "series" in low else "parallel_capacitance" if "parallel" in low else None
    elif target == "capacitance" and charges and voltages:
        variables.update(q=charges[0], V=voltages[0])
        formula_id = "capacitance_c_q_over_v"
    elif target == "voltage" and currents and resistances:
        variables.update(I=currents[0], R=resistances[0])
        formula_id = "ohms_law_v_ir"
    elif target == "voltage" and powers and currents:
        variables.update(P=powers[0], I=currents[0])
        formula_id = "power_v_p_over_i"
    elif target == "voltage" and powers and resistances:
        variables.update(P=powers[0], R=resistances[0])
        formula_id = "power_v_sqrt_pr"
    elif target == "voltage" and energies and charges:
        variables.update(U=energies[0], q=charges[0])
        formula_id = "potential_voltage_v_u_over_q"
    elif target == "current" and voltages and resistances:
        variables.update(V=voltages[0], R=resistances[0])
        formula_id = "ohms_law_i_v_over_r"
    elif target == "current" and powers and voltages:
        variables.update(P=powers[0], V=voltages[0])
        formula_id = "power_i_p_over_v"
    elif target == "current" and powers and resistances:
        variables.update(P=powers[0], R=resistances[0])
        formula_id = "power_i_sqrt_p_over_r"
    elif target == "power":
        # Guard for power_p_v2r: abstain on AC/RMS/reactance/network problems
        ac_keywords = ["ac", "rms", "reactance", "inductive", "capacitive", "impedance", "frequency", "network", "equivalent circuit"]
        has_ac_or_network = any(kw in low for kw in ac_keywords)
        if voltages and currents:
            variables.update(V=voltages[0], I=currents[0])
            formula_id = "power_p_vi"
        elif currents and resistances:
            variables.update(I=currents[0], R=resistances[0])
            formula_id = "power_p_i2r"
        elif voltages and resistances and has_ac_or_network and "rms" in low:
            variables.update(V=voltages[0], R=resistances[0])
            formula_id = "rlc_power_vrms"
        elif voltages and resistances and not has_ac_or_network:
            variables.update(V=voltages[0], R=resistances[0])
            formula_id = "power_p_v2r"
        elif has_ac_or_network:
            ambiguity.append("AC/reactance/network power problem requires extended formula registry (not yet implemented).")
    elif target == "charge" and capacitances and voltages:
        variables.update(C=capacitances[0], V=voltages[0])
        formula_id = "capacitor_charge_q_cv"
    elif target == "energy" and capacitances and voltages and "capacitor" in low:
        variables.update(C=capacitances[0], V=voltages[0])
        formula_id = "capacitor_energy_e_half_cv2"
    elif target == "force" and len(charges) >= 2 and distances:
        geometry_keywords = ["q3", "third charge", "ca", "cb", "ac", "bc", "ma", "mb", "midpoint", "perpendicular bisector", "triangle", "equilateral", "geometry", "placed at", "located at"]
        has_geometry = any(kw in low for kw in geometry_keywords)
        multi_distance_pattern = r"\b(ca|cb|ab|ac|bc|ma|mb|r1|r2|d1|d2)\b"
        has_multi_distance = bool(re.search(multi_distance_pattern, low))
        simple_two_charge = "two charges" in low and len(charges) == 2 and len(distances) == 1
        if (simple_two_charge or (not has_geometry and not has_multi_distance)) and len(charges) == 2 and len(distances) == 1:
            variables.update(q1=charges[0], q2=charges[1], r=distances[0])
            formula_id = "coulomb_force"
        else:
            ambiguity.append("Multi-charge geometry or complex force problem requires vector/net-force solver (not yet implemented).")
    elif target == "distance" and len(charges) >= 2 and forces:
        variables.update(q1=charges[0], q2=charges[1], F=forces[0])
        formula_id = "coulomb_distance"
    elif target == "electric_field" and forces and charges:
        variables.update(F=forces[0], q=charges[0])
        formula_id = "electric_field_f_over_q"
    elif target == "electric_field" and charges and distances:
        # Guard: only match single-charge-at-distance problems
        # Abstain on multi-charge (q1/q2, "two charges", "three charges") or geometry
        # Abstain on distributed charge geometry (ring, wire, plate, infinite, uniformly distributed)
        multi_charge_keywords = ["q1", "q2", "q3", "two charges", "three charges", "multiple charges", "charges are", "midpoint", "perpendicular bisector", "triangle"]
        distributed_charge_keywords = ["ring", "wire", "plate", "infinite", "infinitely", "uniformly distributed", "uniformly charged", "linear charge density", "surface charge density", "semicircle"]
        has_multi_charge = any(kw in low for kw in multi_charge_keywords) or len(charges) > 1
        has_distributed_charge = any(kw in low for kw in distributed_charge_keywords)
        if not has_multi_charge and not has_distributed_charge and len(charges) == 1 and len(distances) >= 1:
            variables.update(q=charges[0], r=distances[0])
            formula_id = "electric_field_kq_r2"
        else:
            if has_distributed_charge:
                ambiguity.append("Distributed charge geometry (ring/wire/plate) requires integration or specialized formula (not yet implemented).")
            else:
                ambiguity.append("Multi-charge electric field problem requires superposition/vector solver (not yet implemented).")
    elif target == "potential_energy" and charges and voltages:
        variables.update(q=charges[0], V=voltages[0])
        formula_id = "potential_energy_u_qv"
    elif target == "energy" and charges and voltages:
        # If target is generic "energy" with charge+voltage, prefer potential_energy
        variables.update(q=charges[0], V=voltages[0])
        formula_id = "potential_energy_u_qv"

    if not formula_id:
        ambiguity.append("No deterministic formula matched the question.")
    if formula_id and not variables:
        ambiguity.append("Formula matched but variables were not extracted.")
    return ParsedPhysicsProblem(question=question, formula_id=formula_id, target_quantity=target, variables=variables, quantities=quantities, ambiguity=ambiguity)
