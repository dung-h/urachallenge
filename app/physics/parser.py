from __future__ import annotations

import os
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


@dataclass(frozen=True)
class _QuantityBuckets:
    currents: list[float]
    voltages: list[float]
    resistances: list[float]
    capacitances: list[float]
    charges: list[float]
    charge_density_line: list[float]
    charge_density_area: list[float]
    dipole_moments: list[float]
    distances: list[float]
    forces: list[float]
    electric_fields: list[float]
    powers: list[float]
    energies: list[float]
    inductances: list[float]
    magnetic_fields: list[float]
    fluxes: list[float]
    areas: list[float]
    frequencies: list[float]
    speeds: list[float]


@dataclass(frozen=True)
class _FormulaCandidate:
    formula_id: str
    variables: dict[str, float]


def _values(quantities: list[Quantity], si_unit: str) -> list[float]:
    return [q.si_value for q in quantities if q.si_unit == si_unit]


def _bucket_quantities(quantities: list[Quantity]) -> _QuantityBuckets:
    return _QuantityBuckets(
        currents=_values(quantities, "A"),
        voltages=_values(quantities, "V"),
        resistances=_values(quantities, "ohm"),
        capacitances=_values(quantities, "F"),
        charges=_values(quantities, "C"),
        charge_density_line=_values(quantities, "C/m"),
        charge_density_area=_values(quantities, "C/m²"),
        dipole_moments=_values(quantities, "C·m"),
        distances=_values(quantities, "m"),
        forces=_values(quantities, "N"),
        electric_fields=_values(quantities, "N/C"),
        powers=_values(quantities, "W"),
        energies=_values(quantities, "J"),
        inductances=_values(quantities, "H"),
        magnetic_fields=_values(quantities, "T"),
        fluxes=_values(quantities, "Wb"),
        areas=_values(quantities, "m²"),
        frequencies=_values(quantities, "Hz"),
        speeds=_values(quantities, "m/s"),
    )


def _count_word_value(question: str) -> int | None:
    low = question.lower()
    for word, value in {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }.items():
        if re.search(rf"\b{word}\b", low):
            return value
    return None


def _parallel_resistance(values: list[float]) -> float:
    return 1.0 / sum(1.0 / value for value in values)


def _series_capacitance(values: list[float]) -> float:
    return 1.0 / sum(1.0 / value for value in values)


def _parallel_capacitance(values: list[float]) -> float:
    return sum(values)


def _arc_angle_radians(question: str) -> float | None:
    low = normalize_number_words(question.lower())
    if any(token in low for token in ["quarter-circle", "quarter circle", "quarter arc", "90 degree", "90-degree", "ninety degree", "one-quarter"]):
        return math.pi / 2.0
    if any(token in low for token in ["semicircle", "semi-circle", "semicircular", "half circle", "half-circle", "180 degree", "180-degree"]):
        return math.pi
    if any(token in low for token in ["three-quarter", "three quarter", "270 degree", "270-degree"]):
        return 3.0 * math.pi / 2.0
    if any(token in low for token in ["full circle", "360 degree", "360-degree", "entire circle"]):
        return 2.0 * math.pi
    match = re.search(r"subtend(?:ing|s)?\s+([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|°)", low, re.I)
    if match:
        return math.radians(float(match.group(1)))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|°)\s+(?:arc|sector|subtends?|subtending)", low, re.I)
    if match:
        return math.radians(float(match.group(1)))
    return None


def _composite_resistance(question: str, resistances: list[float] | None = None) -> float | None:
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

    match = re.search(rf"{num}\s*ohms?\s+resistor\s+is\s+in\s+series\s+with\s+a\s+parallel\s+pair\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?,?\s+all\s+in\s+parallel\s+with\s+{num}\s*ohms?", text)
    if match:
        series_branch, first, second, outer_branch = (float(value) for value in match.groups())
        nested = series_branch + _parallel_resistance([first, second])
        return _parallel_resistance([nested, outer_branch])

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

    match = re.search(
        rf"{num}\s*ohms?\s+branch\s+is\s+in\s+parallel\s+with\s+a\s+nested\s+branch\s+containing\s+{num}\s*ohms?\s+in\s+series\s+with\s+a\s+parallel\s+pair\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?",
        text,
    )
    if match:
        outer_branch, series_branch, first, second = (float(value) for value in match.groups())
        return _parallel_resistance([outer_branch, series_branch + _parallel_resistance([first, second])])

    match = re.search(
        rf"{num}\s*ohms?\s+in\s+series\s+with\s+a\s+parallel\s+pair\s+of\s+{num}\s*ohms?\s+and\s+{num}\s*ohms?,?\s+all\s+in\s+parallel\s+with\s+{num}\s*ohms?",
        text,
    )
    if match:
        series_branch, first, second, outer_branch = (float(value) for value in match.groups())
        nested = series_branch + _parallel_resistance([first, second])
        return _parallel_resistance([nested, outer_branch])

    match = re.search(
        rf"(?:2|two)\s+{num}\s*ohms?\s+resistors?\s+are\s+in\s+parallel\s+and\s+then\s+in\s+series\s+with\s+(?:2|two)\s+{num}\s*ohms?\s+resistors?\s+in\s+parallel",
        text,
    )
    if match:
        first, second = (float(value) for value in match.groups())
        return _parallel_resistance([first, first]) + _parallel_resistance([second, second])

    return None


def _composite_capacitance(question: str, capacitances: list[float] | None = None) -> float | None:
    text = normalize_number_words(question.lower()).replace(",", " ")
    text = re.sub(r"\s+", " ", text)
    num = r"([0-9]+(?:\.[0-9]+)?)"
    cap = r"(?:microfarads?|u?f|μf|µf)"

    def to_farads(value: float) -> float:
        return value * 1e-6

    match = re.search(
        rf"(?:2|two)\s+{num}\s*{cap}\s+capacitors?\s+in\s+parallel\s+are\s+then\s+in\s+series\s+with\s+{num}\s*{cap}",
        text,
    )
    if match:
        if capacitances and len(capacitances) >= 2:
            return _series_capacitance([_parallel_capacitance([capacitances[0], capacitances[0]]), capacitances[1]])
        parallel_value, series_branch = (to_farads(float(value)) for value in match.groups())
        return _series_capacitance([_parallel_capacitance([parallel_value, parallel_value]), series_branch])

    match = re.search(
        rf"(?:a\s+)?{num}\s*{cap}\s+and\s+{num}\s*{cap}\s+capacitor\s+pair\s+is\s+in\s+series,?\s+then\s+that\s+pair\s+is\s+in\s+parallel\s+with\s+{num}\s*{cap}",
        text,
    )
    if match:
        first, second, outer_branch = (to_farads(float(value)) for value in match.groups())
        nested = _series_capacitance([first, second])
        return _parallel_capacitance([nested, outer_branch])

    match = re.search(
        rf"a\s+{num}\s*{cap}\s+and\s+{num}\s*{cap}\s+capacitor\s+pair\s+is\s+in\s+series,\s+then\s+that\s+pair\s+is\s+in\s+parallel\s+with\s+{num}\s*{cap}",
        text,
    )
    if match:
        if capacitances and len(capacitances) >= 3:
            return _parallel_capacitance([_series_capacitance([capacitances[0], capacitances[1]]), capacitances[2]])
        first, second, parallel_branch = (to_farads(float(value)) for value in match.groups())
        return _parallel_capacitance([_series_capacitance([first, second]), parallel_branch])

    match = re.search(
        rf"(?:2|two)\s+{num}\s*{cap}\s+capacitors?\s+are\s+in\s+parallel\s+and\s+then\s+in\s+series\s+with\s+{num}\s*{cap}",
        text,
    )
    if match:
        if capacitances and len(capacitances) >= 2:
            return _series_capacitance([_parallel_capacitance([capacitances[0], capacitances[0]]), capacitances[1]])
        parallel_value, series_branch = (to_farads(float(value)) for value in match.groups())
        return _series_capacitance([_parallel_capacitance([parallel_value, parallel_value]), series_branch])

    match = re.search(
        rf"{num}\s*{cap}\s+capacitor\s+is\s+in\s+series\s+with\s+a\s+parallel\s+pair\s+of\s+{num}\s*{cap}\s+and\s+{num}\s*{cap}",
        text,
    )
    if match:
        series_branch, first, second = (to_farads(float(value)) for value in match.groups())
        return _series_capacitance([series_branch, _parallel_capacitance([first, second])])

    match = re.search(
        rf"{num}\s*{cap}\s+capacitor\s+is\s+in\s+parallel\s+with\s+a\s+series\s+branch\s+of\s+{num}\s*{cap}\s+and\s+{num}\s*{cap}",
        text,
    )
    if match:
        parallel_branch, first, second = (to_farads(float(value)) for value in match.groups())
        return _parallel_capacitance([parallel_branch, _series_capacitance([first, second])])

    return None


def _target(text: str) -> str | None:
    low = text.lower()
    resonance_context = "resonance" in low or "resonant" in low or "resonate" in low
    explicit = _target_from_explicit_phrases(low, text)
    if explicit:
        return explicit
    if resonance_context and (
        "angular frequency" in low
        or "angular resonant frequency" in low
        or "angular resonance frequency" in low
        or "omega" in low
        or "ω" in text
        or "rad/s" in low
    ):
        return "angular_frequency"
    circuit = _target_from_circuit_context(low, resonance_context)
    if circuit:
        return circuit
    return None


def _preseeded_formulas_disabled() -> bool:
    return os.getenv("URA_DISABLE_PRESEEDED_FORMULAS", "").strip().lower() in {"1", "true", "yes", "on"}


def _resonance_or_wave_candidate(target: str | None, low: str, buckets: _QuantityBuckets) -> _FormulaCandidate | None:
    resonance_context = "resonance" in low or "resonant" in low or "resonate" in low
    if target == "angular_frequency" and resonance_context and buckets.inductances and buckets.capacitances:
        return _FormulaCandidate("rlc_angular_resonant_frequency", {"L": buckets.inductances[0], "C": buckets.capacitances[0]})
    if target == "frequency" and resonance_context and buckets.inductances and buckets.capacitances:
        return _FormulaCandidate("rlc_resonant_frequency", {"L": buckets.inductances[0], "C": buckets.capacitances[0]})
    if target == "frequency" and buckets.speeds and buckets.distances and any(term in low for term in ["wave", "wavelength", "lambda", "λ"]):
        return _FormulaCandidate("wave_frequency", {"v": buckets.speeds[0], "wavelength": buckets.distances[0]})
    if target == "capacitance" and resonance_context and buckets.inductances and buckets.frequencies:
        return _FormulaCandidate("rlc_capacitance_from_resonant_frequency", {"f": buckets.frequencies[0], "L": buckets.inductances[0]})
    if target == "inductance" and resonance_context and buckets.capacitances and buckets.frequencies:
        return _FormulaCandidate("rlc_inductance_from_resonant_frequency", {"f": buckets.frequencies[0], "C": buckets.capacitances[0]})
    if target == "resistance" and resonance_context and buckets.resistances and "impedance" in low:
        return _FormulaCandidate("rlc_resonance_impedance", {"R": buckets.resistances[0]})
    return None


def _transformer_candidate(target: str | None, low: str, buckets: _QuantityBuckets) -> _FormulaCandidate | None:
    if "transformer" not in low or target != "voltage" or not buckets.voltages:
        return None
    primary_patterns = [
        r"\b(?:n1|n_primary|primary(?:\s+coil)?(?:\s+turns?)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        r"\b(\d+(?:\.\d+)?)\s*(?:turns?|coils?)\s+on\s+the\s+primary\b",
        r"\bprimary\s*(?:has|with|of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:turns?|coils?)\b",
    ]
    secondary_patterns = [
        r"\b(?:n2|n_secondary|secondary(?:\s+coil)?(?:\s+turns?)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        r"\b(\d+(?:\.\d+)?)\s*(?:turns?|coils?)\s+on\s+the\s+secondary\b",
        r"\bsecondary\s*(?:has|with|of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:turns?|coils?)\b",
    ]
    primary_turns = None
    for pattern in primary_patterns:
        primary_turns = re.search(pattern, low)
        if primary_turns:
            break
    secondary_turns = None
    for pattern in secondary_patterns:
        secondary_turns = re.search(pattern, low)
        if secondary_turns:
            break
    if not primary_turns or not secondary_turns:
        return None
    return _FormulaCandidate(
        "transformer_secondary_voltage",
        {
            "V_primary": buckets.voltages[0],
            "N_primary": float(primary_turns.group(1)),
            "N_secondary": float(secondary_turns.group(1)),
        },
    )


def _solenoid_field_candidate(target: str | None, low: str) -> _FormulaCandidate | None:
    if target != "magnetic_field" or "solenoid" not in low:
        return None
    turns_match = re.search(r"\b(?:N|n)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", low) or re.search(r"\b(\d+(?:\.\d+)?)\s*(?:turns?|coils?)\b", low)
    length_match = re.search(r"\b(?:l|length)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*m\b", low)
    current_match = re.search(r"\b(?:I|current)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*a\b", low)
    if not turns_match or not length_match or not current_match:
        return None
    return _FormulaCandidate(
        "solenoid_B",
        {"N": float(turns_match.group(1)), "l": float(length_match.group(1)), "I": float(current_match.group(1))},
    )


def _force_candidate(target: str | None, low: str, buckets: _QuantityBuckets) -> _FormulaCandidate | None:
    if target == "force":
        if len(buckets.electric_fields) == 1 and len(buckets.charges) == 1:
            return _FormulaCandidate("force_from_field_charge", {"q": buckets.charges[0], "E": buckets.electric_fields[0]})
    if target != "force" or len(buckets.forces) != 2:
        return None
    angle_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", low)
    if angle_match:
        return _FormulaCandidate(
            "resultant_force_angle",
            {"F1": buckets.forces[0], "F2": buckets.forces[1], "theta_rad": math.radians(float(angle_match.group(1)))},
        )
    if "perpendicular" in low or "right angle" in low or "90" in low:
        return _FormulaCandidate("resultant_force_perpendicular", {"F1": buckets.forces[0], "F2": buckets.forces[1]})
    if "opposite direction" in low or "opposite" in low:
        return _FormulaCandidate("resultant_force_collinear_opposite", {"F1": buckets.forces[0], "F2": buckets.forces[1]})
    if "same direction" in low or "collinear" in low:
        return _FormulaCandidate("resultant_force_collinear_same", {"F1": buckets.forces[0], "F2": buckets.forces[1]})
    return None


def _measurement_error_candidate(low: str, buckets: _QuantityBuckets) -> _FormulaCandidate | None:
    if "error" not in low and "least count" not in low and "uncertainty" not in low:
        return None
    if "absolute error" in low and buckets.currents and len(buckets.currents) >= 2:
        return _FormulaCandidate("measurement_absolute_error", {"least_count": min(buckets.currents)})
    if "relative error" in low and len(buckets.currents) >= 2:
        return _FormulaCandidate(
            "measurement_relative_error",
            {"absolute_error": min(buckets.currents), "reading": max(buckets.currents)},
        )
    if "error" in low and "resistance" in low and len(buckets.voltages) >= 2 and len(buckets.currents) >= 2:
        return _FormulaCandidate(
            "measurement_error_propagation_division",
            {
                "dU": min(buckets.voltages),
                "U": max(buckets.voltages),
                "dI": min(buckets.currents),
                "I": max(buckets.currents),
            },
        )
    return None


def _extract_relative_permittivity(low: str) -> float | None:
    eps_match = re.search(r"\b(?:eps_r|epsilon_r|ε_r)\b\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if not eps_match:
        eps_match = re.search(r"(?:dielectric constant|relative permittivity)\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if not eps_match:
        eps_match = re.search(r"(?:dielectric constant|relative permittivity)\s+(\d+(?:\.\d+)?)", low)
    if not eps_match and "dielectric" in low:
        eps_match = re.search(r"(?:eps|epsilon|ε|κ)\s*(?:=|is|of)\s*(\d+(?:\.\d+)?)", low)
    if not eps_match:
        return None
    try:
        return float(eps_match.group(1))
    except Exception:
        return None


def _extract_power_factor_or_phase(low: str) -> float | None:
    factor_match = re.search(r"\b(?:power factor|pf|cos\s*phi|cos\s*ϕ|cos\s*theta|cos\s*θ)\b\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)", low, re.I)
    if factor_match:
        try:
            return float(factor_match.group(1))
        except Exception:
            return None
    angle_match = re.search(r"\b(?:phase angle|phase difference|phi|ϕ|theta|θ)\b\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:degrees?|°)\b", low, re.I)
    if angle_match:
        try:
            return math.cos(math.radians(float(angle_match.group(1))))
        except Exception:
            return None
    return None


def _dielectric_candidate(target: str | None, low: str, buckets: _QuantityBuckets, eps: float | None) -> _FormulaCandidate | None:
    dielectric_context = ("dielectric" in low) or ("relative permittivity" in low) or ("dielectric constant" in low)
    dielectric_event = dielectric_context and any(
        token in low for token in ["inserted", "introduced", "removed", "immersed", "immersed in", "filled", "replaced"]
    )
    candidate: _FormulaCandidate | None = None
    if target == "capacitance" and eps is not None and buckets.areas and buckets.distances and ("capacitor" in low and "plate" in low):
        candidate = _FormulaCandidate("parallel_plate_capacitance_dielectric", {"A": buckets.areas[0], "d": buckets.distances[0], "eps": eps})
    if target == "capacitance" and eps is not None and buckets.capacitances and any(token in low for token in ["fills", "filled", "with a dielectric", "dielectric fills", "dielectric fills the capacitor", "new capacitance"]):
        candidate = _FormulaCandidate("dielectric_capacitance_change", {"C0": buckets.capacitances[0], "eps": eps})
    if target == "dielectric_constant" and buckets.capacitances and buckets.areas and buckets.distances and ("capacitor" in low and "plate" in low):
        candidate = _FormulaCandidate("dielectric_constant_from_parallel_plate", {"C": buckets.capacitances[0], "d": buckets.distances[0], "A": buckets.areas[0]})
    if target == "energy_density" and eps is not None and buckets.voltages and buckets.distances and ("capacitor" in low and "plate" in low):
        candidate = _FormulaCandidate("energy_density_dielectric", {"eps": eps, "V": buckets.voltages[0], "d": buckets.distances[0]})
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
        if target == "voltage" and buckets.voltages and disconnected:
            candidate = _FormulaCandidate("dielectric_voltage_disconnected", {"V_old": buckets.voltages[0], "eps": eps})
        elif target == "voltage" and buckets.voltages and connected:
            candidate = _FormulaCandidate("dielectric_voltage_connected", {"V_source": buckets.voltages[0]})
        if target == "energy" and buckets.capacitances and buckets.voltages and "capacitor" in low and disconnected:
            candidate = _FormulaCandidate("dielectric_energy_disconnected", {"C": buckets.capacitances[0], "V": buckets.voltages[0], "eps": eps})
        elif target == "energy" and buckets.capacitances and buckets.voltages and "capacitor" in low and connected:
            candidate = _FormulaCandidate("dielectric_energy_connected", {"C": buckets.capacitances[0], "V": buckets.voltages[0], "eps": eps})
    if target == "electric_field" and eps is not None and dielectric_context and buckets.charges and buckets.distances:
        multi_charge_keywords = ["q1", "q2", "q3", "two charges", "three charges", "multiple charges", "midpoint", "perpendicular bisector", "triangle"]
        has_multi_charge = any(kw in low for kw in multi_charge_keywords) or len(buckets.charges) > 1
        if not has_multi_charge and len(buckets.charges) == 1:
            candidate = _FormulaCandidate("electric_field_kq_r2_in_dielectric", {"q": buckets.charges[0], "r": buckets.distances[0], "eps": eps})
    if target == "energy" and eps is None and buckets.energies and any(token in low for token in ["dielectric", "permittivity"]):
        factor_match = re.search(r"(?:increases|decreases)\s+by\s+(?:a\s+factor\s+of\s+)?(\d+(?:\.\d+)?)", low)
        if factor_match and any(token in low for token in ["disconnected", "isolated", "battery removed", "source removed"]):
            try:
                eps_factor = float(factor_match.group(1))
            except Exception:
                eps_factor = None
            if eps_factor and eps_factor > 0:
                candidate = _FormulaCandidate("dielectric_energy_from_energy_disconnected", {"E_old": buckets.energies[0], "eps": eps_factor})
    return candidate


def _magnetism_candidate(target: str | None, low: str, buckets: _QuantityBuckets) -> _FormulaCandidate | None:
    if "inductor" in low and target == "energy" and buckets.inductances and buckets.currents:
        return _FormulaCandidate("inductor_energy", {"L": buckets.inductances[0], "I": buckets.currents[0]})
    if "inductor" in low and target == "current" and buckets.inductances and buckets.energies:
        return _FormulaCandidate("inductor_current", {"E": buckets.energies[0], "L": buckets.inductances[0]})
    if "inductor" in low and "inductance" in low and buckets.energies and buckets.currents:
        return _FormulaCandidate("inductor_inductance", {"E": buckets.energies[0], "I": buckets.currents[0]})
    if "solenoid" in low and ("flux" in low or "magnetic flux" in low):
        n_match = re.search(r"(\d+)\s*(?:turns?|coils?)", low)
        if n_match and buckets.magnetic_fields and buckets.areas:
            return _FormulaCandidate("solenoid_total_flux", {"N": float(n_match.group(1)), "B": buckets.magnetic_fields[0], "A": buckets.areas[0]})
    if target == "magnetic_field" and buckets.currents and buckets.distances and any(token in low for token in ["circular loop", "circular ring", "loop", "circle"]) and "center" in low:
        return _FormulaCandidate("magnetic_field_circular_loop_center", {"I": buckets.currents[0], "R": max(buckets.distances)})
    return None


def _target_from_explicit_phrases(low: str, text: str) -> str | None:
    if re.search(r"\bwhat\s+is\s+(?:the\s+)?resistance\b", low) or re.search(r"\bwhat\s+resistance\b", low) or re.search(r"\bfind\s+(?:the\s+)?resistance\b", low) or re.search(r"\bcalculate\s+(?:the\s+)?resistance\b", low) or re.search(r"\bdetermine\s+(?:the\s+)?resistance\b", low):
        return "resistance"
    if any(phrase in low for phrase in ["resonant frequency", "resonance frequency", "natural frequency", "oscillation frequency"]):
        return "frequency"
    if "frequency" in low and any(term in low for term in ["wave", "wavelength", "lambda", "λ"]):
        return "frequency"
    if "energy density" in low:
        return "energy_density"
    if any(phrase in low for phrase in ["what is capacitance", "find capacitance", "calculate capacitance", "determine capacitance"]):
        return "capacitance"
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
    if (
        "force" in low
        and "electric field" in low
        and any(phrase in low for phrase in ["what force", "find force", "find the force", "force acts", "what is the force"])
        and not any(phrase in low for phrase in ["what is the electric field", "find the electric field", "calculate the electric field", "determine the electric field", "what is electric field", "find electric field"])
    ):
        return "force"
    if "electric field energy" in low:
        return "energy"
    if "electric field" in low or "electric intensity" in low:
        return "electric_field"
    if any(phrase in low for phrase in ["capacitive reactance", "reactance of the capacitor", "inductive reactance", "reactance of the inductor"]):
        return "reactance"
    if "impedance" in low:
        return "impedance"
    if "separation distance" in low or re.search(r"\bdistance\b", low):
        return "distance"
    return None


def _target_from_circuit_context(low: str, resonance_context: bool) -> str | None:
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
    if "impedance" in low and resonance_context:
        return "resistance"
    if "resistance" in low and any(token in low for token in ["what is the resistance", "what resistance", "has what resistance", "find the resistance"]):
        return "resistance"
    if "compute vi" in low or "rate of energy use" in low or re.search(r"\b(?:what is|find|return)\s+p\b", low):
        return "power"
    if "potential energy" in low and "capacitor" not in low:
        return "potential_energy"
    if "energy" in low and "charge" in low and "voltage" not in low and any(token in low for token in ["potential difference", "through a potential", "through potential"]):
        return "potential_energy"
    if "energy" in low and ("stored" in low or "stores" in low):
        return "energy"
    if "capacitor" in low and "work" in low:
        return "energy"
    if "energy" in low and "capacitor" in low:
        return "energy"
    if "power" in low:
        return "power"
    if "capacitance" in low:
        return "capacitance"
    if "electric potential" in low:
        return "voltage"
    if "voltage" in low or re.search(r"\bpotential difference\b", low):
        return "voltage"
    if "total resistance" in low or "equivalent resistance" in low or "effective resistance" in low or "parallel resistance" in low or "series resistance" in low:
        return "resistance"
    if ("capacitance" in low or re.search(r"\bwhat\s+value\s+of\s+c\b", low) or re.search(r"\bfind\s+c\b", low)) and resonance_context:
        return "capacitance"
    if "inductance" in low or (re.search(r"\bwhat\s+value\s+of\s+l\b", low) and resonance_context):
        return "inductance"
    if "magnetic field" in low or re.search(r"\bwhat\s+is\s+the\s+magnetic\s+field\b", low):
        return "magnetic_field"
    if any(phrase in low for phrase in ["what is the charge", "find the charge", "calculate the charge", "determine the charge", "how much charge", "stored charge", "charge on the capacitor"]):
        return "charge"
    if "charge" in low and "charged to" not in low and "charged" not in low:
        return "charge"
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

    buckets = _bucket_quantities(quantities)
    currents = buckets.currents
    voltages = buckets.voltages
    resistances = buckets.resistances
    capacitances = buckets.capacitances
    charges = buckets.charges
    charge_density_line = buckets.charge_density_line
    charge_density_area = buckets.charge_density_area
    dipole_moments = buckets.dipole_moments
    distances = buckets.distances
    forces = buckets.forces
    powers = buckets.powers
    energies = buckets.energies
    inductances = buckets.inductances
    magnetic_fields = buckets.magnetic_fields
    fluxes = buckets.fluxes
    areas = buckets.areas
    frequencies = buckets.frequencies
    speeds = buckets.speeds

    mixed_composite = ("series" in low and "parallel" in low) or ("parallel" in low and any(token in low for token in ["chain", "after the first", "branch"]))
    composite_resistance = _composite_resistance(question, resistances) if target == "resistance" and mixed_composite else None
    composite_capacitance = _composite_capacitance(question, capacitances) if target == "capacitance" and mixed_composite else None

    resonance_context = "resonance" in low or "resonant" in low or "resonate" in low
    candidate = _resonance_or_wave_candidate(target, low, buckets)
    if candidate:
        variables.update(candidate.variables)
        formula_id = candidate.formula_id

    transformer_context = "transformer" in low and target == "voltage"
    for candidate in (
        _transformer_candidate(target, low, buckets),
        _solenoid_field_candidate(target, low),
        _force_candidate(target, low, buckets) if not formula_id or target == "force" else None,
    ):
        if candidate:
            variables.update(candidate.variables)
            formula_id = candidate.formula_id

    candidate = _measurement_error_candidate(low, buckets)
    if candidate:
        variables.update(candidate.variables)
        formula_id = candidate.formula_id

    # Dielectric handling
    # 1) Capacitor transforms when a dielectric is inserted/immersed/removed, with "connected" vs "disconnected".
    # 2) Electric field scaling in a dielectric medium: E = (k/eps_r) * q / r^2.
    # Keep this deterministic and conservative: only match when wording is explicit.

    # Extract relative permittivity / dielectric constant eps_r.
    # Avoid treating Coulomb's constant k (9e9) or vacuum permittivity epsilon_0 as eps_r.
    eps = _extract_relative_permittivity(low)

    dielectric_context = ("dielectric" in low) or ("relative permittivity" in low) or ("dielectric constant" in low)

    for candidate in (
        _dielectric_candidate(target, low, buckets, eps),
        _magnetism_candidate(target, low, buckets),
    ):
        if candidate:
            variables.update(candidate.variables)
            formula_id = candidate.formula_id

    # Skip to existing logic if already matched
    if formula_id:
        pass
    elif target == "resistance" and composite_resistance is not None:
        variables.update(R_total=composite_resistance, resistances=resistances)
        formula_id = "composite_resistance"
    elif target == "resistance" and ("bridge" in low or "diamond" in low):
        count = _count_word_value(question)
        if count == 4 and len(resistances) >= 2:
            arm = min(resistances)
            bridge = max(resistances)
            variables.update(R_total=arm, resistances=[arm, bridge])
            formula_id = "bridge_symmetric_resistance"
    elif target == "capacitance" and composite_capacitance is not None:
        variables.update(C_total=composite_capacitance, capacitances=capacitances)
        formula_id = "composite_capacitance"
    elif target in {"resistance", "capacitance"} and mixed_composite:
        ambiguity.append("Mixed series/parallel nested topology is not supported by deterministic parser.")
    elif target == "resistance" and resistances and ("parallel" in low or "series" in low):
        repeated_count = _count_word_value(question)
        if len(resistances) == 1 and repeated_count and repeated_count >= 2:
            variables["resistances"] = [resistances[0]] * repeated_count
        else:
            variables["resistances"] = resistances
        formula_id = "parallel_resistance" if "parallel" in low else "series_resistance" if "series" in low else None
    elif target == "resistance" and "heat" in low:
        ambiguity.append("Question includes an unsupported heat target/distractor.")
    elif target == "resistance":
        unrelated_measurements = any(token in low for token in ["unrelated circuits", "separate unrelated", "separate circuits", "different circuits", "unrelated quantities"])
        if len(voltages) > 1 and len(currents) == 1:
            ambiguity.append("Voltage notes are contradictory, so resistance cannot be determined deterministically.")
        elif len(currents) > 1 and len(voltages) == 1:
            ambiguity.append("Current notes are contradictory, so resistance cannot be determined deterministically.")
        elif len(voltages) == 1 and len(currents) == 1 and unrelated_measurements:
            ambiguity.append("Voltage and current are reported for unrelated circuits, so resistance cannot be determined deterministically.")
        elif len(voltages) == 1 and len(currents) == 1:
            variables.update(V=voltages[0], I=currents[0])
            formula_id = "ohms_law_r_v_over_i"
        elif len(powers) == 1 and len(currents) == 1:
            variables.update(P=powers[0], I=currents[0])
            formula_id = "power_r_p_over_i2"
        elif len(voltages) == 1 and len(powers) == 1:
            variables.update(V=voltages[0], P=powers[0])
            formula_id = "power_r_v2_over_p"
        elif voltages or currents or powers:
            ambiguity.append("Ambiguous or conflicting measurements prevent a deterministic resistance calculation.")
    elif target == "capacitance" and capacitances and ("parallel" in low or "series" in low):
        if dielectric_context and eps is not None:
            variables.update(C0=capacitances[0], eps=eps)
            formula_id = "dielectric_capacitance_change"
        else:
            variables["capacitances"] = capacitances
            formula_id = "series_capacitance" if "series" in low else "parallel_capacitance" if "parallel" in low else None
    elif target == "capacitance" and len(capacitances) == 1 and any(token in low for token in ["what is capacitance", "find capacitance", "calculate capacitance", "determine capacitance"]):
        variables.update(C=capacitances[0])
        formula_id = "direct_capacitance_reported"
    elif target == "capacitance" and charges and voltages:
        variables.update(q=charges[0], V=voltages[0])
        formula_id = "capacitance_c_q_over_v"
    elif target == "capacitance" and any(token in low for token in ["spherical capacitor", "sphere capacitor"]) and len(distances) >= 2:
        variables.update(a=min(distances), b=max(distances))
        formula_id = "spherical_capacitor_capacitance"
    elif target == "reactance" and frequencies and capacitances and "capacitive" in low:
        variables.update(f=frequencies[0], C=capacitances[0])
        formula_id = "capacitive_reactance"
    elif target == "reactance" and frequencies and inductances and "inductive" in low:
        variables.update(f=frequencies[0], L=inductances[0])
        formula_id = "inductive_reactance"
    elif target == "impedance" and frequencies and inductances and capacitances and resistances and ("series rlc" in low or ("rlc" in low and "impedance" in low)):
        variables.update(R=resistances[0], L=inductances[0], C=capacitances[0], f=frequencies[0])
        formula_id = "series_rlc_impedance"
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
        cos_phi = _extract_power_factor_or_phase(low)
        rms_or_average_context = any(
            token in low
            for token in [
                "rms",
                "average power",
                "average power dissipated",
                "effective voltage",
                "effective current",
                "rms voltage",
                "rms current",
            ]
        )
        if has_ac_or_network and voltages and currents and cos_phi is not None:
            variables.update(V=voltages[0], I=currents[0], cos_phi=cos_phi)
            formula_id = "ac_power_vi_cos_phi"
        elif has_ac_or_network and voltages and resistances and rms_or_average_context:
            variables.update(V=voltages[0], R=resistances[0])
            formula_id = "rlc_power_vrms"
        elif has_ac_or_network and currents and resistances and rms_or_average_context:
            variables.update(I=currents[0], R=resistances[0])
            formula_id = "power_p_i2r"
        elif has_ac_or_network and voltages and currents:
            ambiguity.append("AC power with voltage/current needs RMS or phase information (not yet implemented).")
        elif voltages and currents:
            variables.update(V=voltages[0], I=currents[0])
            formula_id = "power_p_vi"
        elif currents and resistances and not has_ac_or_network:
            variables.update(I=currents[0], R=resistances[0])
            formula_id = "power_p_i2r"
        elif voltages and resistances and has_ac_or_network and any(token in low for token in ["rms", "average power", "effective voltage", "rms voltage"]):
            variables.update(V=voltages[0], R=resistances[0])
            formula_id = "rlc_power_vrms"
        elif voltages and resistances and not has_ac_or_network:
            variables.update(V=voltages[0], R=resistances[0])
            formula_id = "power_p_v2r"
        elif has_ac_or_network and currents and resistances:
            ambiguity.append("AC power with current/resistance needs RMS or phase information (not yet implemented).")
        elif has_ac_or_network:
            ambiguity.append("AC/reactance/network power problem requires extended formula registry (not yet implemented).")
    elif target == "energy" and capacitances and voltages and "capacitor" in low:
        variables.update(C=capacitances[0], V=voltages[0])
        formula_id = "capacitor_energy_e_half_cv2"
    elif target == "voltage" and capacitances and energies and "capacitor" in low:
        variables.update(E=energies[0], C=capacitances[0])
        formula_id = "capacitor_voltage_from_energy"
    elif target == "voltage" and charges and distances and "ring" in low and "axis" in low and any(token in low for token in ["potential", "voltage"]):
        variables.update(q=charges[0], R=max(distances), x=min(distances) if len(distances) > 1 else distances[0])
        formula_id = "electric_potential_uniform_ring_axis"
    elif target == "voltage" and charges and distances and "square" in low and "loop" in low and "center" in low and any(token in low for token in ["potential", "voltage"]):
        variables.update(q=charges[0], a=max(distances))
        formula_id = "electric_potential_square_loop_center"
    elif target == "voltage" and charges and distances and "arc" in low and "center" in low and "potential" in low:
        variables.update(q=charges[0], R=max(distances))
        formula_id = "electric_potential_circular_arc_center"
    elif target == "voltage" and charges and distances and "electric potential" in low and "ring" in low and "center" in low:
        variables.update(q=charges[0], R=max(distances))
        formula_id = "electric_potential_uniform_ring_center"
    elif target == "charge" and capacitances and voltages:
        variables.update(C=capacitances[0], V=voltages[0])
        formula_id = "capacitor_charge_q_cv"
    elif target == "voltage" and voltages and "source" in low and not transformer_context and not currents and not powers:
        variables.update(V=voltages[0])
        formula_id = "direct_voltage_source"
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
    elif target == "force":
        if any(token in low for token in ["charge", "coulomb", "electric force"]):
            ambiguity.append("Charge magnitudes are missing, so Coulomb's law cannot be evaluated deterministically.")
    elif target == "distance" and len(charges) >= 2 and forces:
        variables.update(q1=charges[0], q2=charges[1], F=forces[0])
        formula_id = "coulomb_distance"
    elif target == "electric_field" and forces and charges:
        variables.update(F=forces[0], q=charges[0])
        formula_id = "electric_field_f_over_q"
    elif target == "electric_field" and charge_density_area and distances and ("disk" in low or "disc" in low) and "axis" in low:
        radius = max(distances)
        axis_distance = min(distances)
        if radius is not None and axis_distance is not None:
            variables.update(sigma=charge_density_area[0], z=axis_distance, R=radius)
            formula_id = "electric_field_uniform_disk_axis"
    elif target == "voltage" and charges and distances and ("disk" in low or "disc" in low) and "axis" in low:
        radius = max(distances)
        axis_distance = min(distances)
        if radius is not None and axis_distance is not None:
            variables.update(q=charges[0], z=axis_distance, R=radius)
            formula_id = "electric_potential_uniform_disk_axis"
    elif target == "voltage" and charges and distances and "sphere" in low and "shell" in low:
        if any(token in low for token in ["inside", "within"]):
            variables.update(q=charges[0], R=max(distances))
            formula_id = "electric_potential_uniform_sphere_shell_inside"
        elif any(token in low for token in ["outside", "outside the shell", "outside shell"]):
            variables.update(q=charges[0], r=min(distances))
            formula_id = "electric_potential_uniform_sphere_shell_outside"
    elif target == "electric_field" and charge_density_line and distances and ("infinite" in low or "line charge" in low):
        variables.update(lam=charge_density_line[0], r=distances[0])
        formula_id = "electric_field_infinite_line_charge"
    elif target == "electric_field" and charges and distances and "sphere" in low and "shell" not in low:
        if any(token in low for token in ["inside", "within"]):
            variables.update(q=charges[0], r=min(distances), R=max(distances))
            formula_id = "electric_field_uniform_sphere_inside"
        else:
            variables.update(q=charges[0], r=min(distances))
            formula_id = "electric_field_uniform_sphere_outside"
    elif target == "electric_field" and dipole_moments and distances and ("dipole" in low and ("axial" in low or "axis" in low)):
        variables.update(p=dipole_moments[0], r=distances[0])
        formula_id = "electric_field_dipole_axial"
    elif target == "electric_field" and charges and distances and "arc" in low and "center" in low:
        theta_rad = _arc_angle_radians(question)
        if theta_rad is not None:
            variables.update(q=charges[0], R=max(distances), theta_rad=theta_rad)
            formula_id = "electric_field_circular_arc_center"
    elif target == "electric_field" and charges and distances and ("semicircular arc" in low or "semicircle" in low or "half circle" in low) and "center" in low:
        variables.update(q=charges[0], R=max(distances))
        formula_id = "electric_field_semicircular_arc_center"
    elif target == "electric_field" and charges and distances and "perpendicular bisector" in low and any(token in low for token in ["rod", "wire", "line"]) and any(token in low for token in ["finite", "length"]):
        variables.update(q=charges[0], d=min(distances), L=max(distances))
        formula_id = "electric_field_finite_line_perpendicular_bisector"
    elif target == "electric_field" and charges and distances and any(token in low for token in ["rod", "wire", "line"]) and any(token in low for token in ["finite", "length"]) and "axis" in low and "perpendicular bisector" not in low:
        if any(token in low for token in ["from one end", "from an end"]):
            variables.update(q=charges[0], d=min(distances), L=max(distances))
            formula_id = "electric_field_finite_line_axis_outside_end"
        elif "from the center" in low or "from center" in low:
            variables.update(q=charges[0], x=max(distances), L=min(distances))
            formula_id = "electric_field_finite_line_axis_outside_center"
    elif target == "voltage" and charges and distances and any(token in low for token in ["rod", "wire", "line"]) and any(token in low for token in ["finite", "length"]) and "perpendicular bisector" in low:
        variables.update(q=charges[0], d=min(distances), L=max(distances))
        formula_id = "electric_potential_finite_line_perpendicular_bisector"
    elif target == "voltage" and charges and distances and any(token in low for token in ["rod", "wire", "line"]) and any(token in low for token in ["finite", "length"]) and "axis" in low:
        if any(token in low for token in ["from one end", "from an end"]):
            variables.update(q=charges[0], d=min(distances), L=max(distances))
            formula_id = "electric_potential_finite_line_axis_outside_end"
        elif "from the center" in low or "from center" in low:
            variables.update(q=charges[0], x=max(distances), L=min(distances))
            formula_id = "electric_potential_finite_line_axis_outside_center"
    elif target == "voltage" and charges and distances and "square" in low and "loop" in low and "center" in low and any(token in low for token in ["potential", "voltage"]):
        variables.update(q=charges[0], a=max(distances))
        formula_id = "electric_potential_square_loop_center"
    elif target == "electric_field" and "center" in low and (
        re.search(r"\bsquare\b.*\bloop\b", low)
        or re.search(r"\brectangular\b.*\bloop\b", low)
        or "regular polygon" in low
        or re.search(r"\bregular\b.*\bloop\b", low)
        or "polygon loop" in low
        or "equilateral triangle" in low
    ):
        formula_id = "electric_field_symmetric_loop_center_zero"
    elif target == "electric_field" and charges and distances:
        # Guard: only match single-charge-at-distance problems
        # Abstain on multi-charge (q1/q2, "two charges", "three charges") or geometry
        # Abstain on distributed charge geometry (ring, wire, plate, infinite, uniformly distributed)
        multi_charge_keywords = ["q1", "q2", "q3", "two charges", "three charges", "multiple charges", "charges are", "midpoint", "perpendicular bisector", "triangle"]
        distributed_charge_keywords = ["ring", "wire", "plate", "infinite", "infinitely", "uniformly distributed", "uniformly charged", "linear charge density", "surface charge density", "semicircle", "arc"]
        has_multi_charge = any(kw in low for kw in multi_charge_keywords) or len(charges) > 1
        has_distributed_charge = any(kw in low for kw in distributed_charge_keywords)
        if not has_multi_charge and not has_distributed_charge and len(charges) == 1 and len(distances) >= 1:
            variables.update(q=charges[0], r=distances[0])
            formula_id = "electric_field_kq_r2"
        else:
            if has_distributed_charge and "shell" in low and "sphere" in low:
                # Allow the runtime search layer to handle spherical shell piecewise questions.
                pass
            elif has_distributed_charge:
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

    if _preseeded_formulas_disabled() and formula_id:
        ambiguity.append("Preseeded formula registry disabled for this experiment.")
        formula_id = None

    if not formula_id:
        ambiguity.append("No deterministic formula matched the question.")
    if formula_id and not variables:
        ambiguity.append("Formula matched but variables were not extracted.")
    return ParsedPhysicsProblem(question=question, formula_id=formula_id, target_quantity=target, variables=variables, quantities=quantities, ambiguity=ambiguity)
