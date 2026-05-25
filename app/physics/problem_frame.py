from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.physics.parser import ParsedPhysicsProblem


@dataclass(frozen=True)
class ProblemFrame:
    target_quantity: str | None
    geometry: str | None
    source_type: str | None
    observation_point: str | None
    method_family: str | None
    query_plan: list[str] = field(default_factory=list)
    blocked_formulas: list[str] = field(default_factory=list)
    evidence_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PhysicsSearchObjective:
    target_quantity: str | None
    quantity_units: list[str]
    structural_terms: list[str]
    query_plan: list[str]


@dataclass(frozen=True)
class MethodEvidence:
    method_family: str
    evidence_terms: list[str]
    required_terms: list[str] = field(default_factory=list)
    blocked_formula_families: list[str] = field(default_factory=list)
    query_templates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MethodProposal:
    method_family: str
    confidence: float
    matched_terms: list[str]
    blocked_formula_families: list[str]
    query_plan: list[str]
    evidence_source: str = "local_method_evidence"


_STOP_TERMS = {
    "what", "from", "with", "that", "this", "then", "than", "into", "onto", "over", "under",
    "find", "used", "total", "value", "center", "carries", "applied", "question", "calculate",
}


_METHOD_EVIDENCE: tuple[MethodEvidence, ...] = (
    MethodEvidence(
        method_family="circuit_open_state",
        evidence_terms=["open switch", "switch is open", "incomplete circuit", "no current"],
        required_terms=["switch"],
        query_templates=["{target} open switch incomplete circuit reasoning", "{target} no current open circuit method"],
    ),
    MethodEvidence(
        method_family="distributed_charge_integration",
        evidence_terms=[
            "uniformly charged", "uniformly distributed", "linear charge density", "surface charge density",
            "ring", "wire", "rod", "arc", "plate", "semicircle", "axis", "infinite", "infinitely",
            "disk", "disc", "sheet", "line charge",
        ],
        blocked_formula_families=["point_charge"],
        query_templates=["{target} distributed charge derivation", "{terms} symmetry integration method"],
    ),
    MethodEvidence(
        method_family="network_reduction_or_symbolic",
        evidence_terms=["bridge", "ladder", "diamond", "mesh", "cross resistor", "resistor network", "network of resistors"],
        required_terms=["network"],
        blocked_formula_families=["simple_network_reduction"],
        query_templates=["{terms} equivalent resistance method", "{target} network nodal analysis derivation"],
    ),
    MethodEvidence(
        method_family="vector_superposition",
        evidence_terms=[
            "perpendicular bisector", "midpoint", "equidistant", "triangle", "equilateral", "resultant",
            "components", "vector sum", "superposition", "dipole", "axial line", "equatorial line",
        ],
        blocked_formula_families=["point_charge"],
        query_templates=["{target} vector superposition method", "{terms} component symmetry derivation"],
    ),
    MethodEvidence(
        method_family="dielectric_transform",
        evidence_terms=["dielectric", "relative permittivity", "epsilon_r", "inserted dielectric", "disconnected", "connected"],
        query_templates=["{target} dielectric capacitor state change method", "{terms} connected disconnected dielectric derivation"],
    ),
    MethodEvidence(
        method_family="ac_circuit_method",
        evidence_terms=["resonance", "resonant", "impedance", "reactance", "xl", "xc", "series rlc", "ac circuit", "alternating current"],
        query_templates=["{target} ac circuit reactance impedance method", "{terms} resonance derivation"],
    ),
    MethodEvidence(
        method_family="transformer_relation",
        evidence_terms=["transformer", "primary", "secondary", "turns ratio"],
        query_templates=["{target} transformer turns ratio method", "{terms} ideal transformer relation"],
    ),
    MethodEvidence(
        method_family="magnetics_geometry",
        evidence_terms=["solenoid", "inductor", "magnetic flux", "magnetic field", "coil"],
        query_templates=["{target} magnetic geometry method", "{terms} solenoid inductor derivation"],
    ),
)


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _structural_terms(question: str) -> list[str]:
    low = question.lower().replace("ε", "epsilon")
    terms: set[str] = set()
    for match in re.finditer(r"[a-zA-Z_][a-zA-Z_/-]{2,}", low):
        token = match.group(0).strip("-_/")
        if token and token not in _STOP_TERMS and not token.isdigit():
            terms.add(token)
    for phrase in (
        "open switch", "switch is open", "uniformly charged", "uniformly distributed", "linear charge density",
        "surface charge density", "perpendicular bisector", "cross resistor", "resistor network", "network of resistors",
        "relative permittivity", "series rlc", "ac circuit", "alternating current", "turns ratio", "magnetic field",
        "magnetic flux",
    ):
        if phrase in low:
            terms.add(phrase)
    return sorted(terms)


def _build_search_objective(parsed: ParsedPhysicsProblem, question: str) -> PhysicsSearchObjective:
    quantity_units = sorted({quantity.si_unit for quantity in parsed.quantities})
    terms = _structural_terms(question)
    compact_terms = " ".join(terms[:6]) or "physics"
    target = parsed.target_quantity or "physics"
    return PhysicsSearchObjective(
        target_quantity=parsed.target_quantity,
        quantity_units=quantity_units,
        structural_terms=terms,
        query_plan=[
            f"{target} method selection with quantities {' '.join(quantity_units) or 'unknown units'}",
            f"{target} derivation for {compact_terms}",
        ],
    )


def _format_query(template: str, objective: PhysicsSearchObjective, evidence: MethodEvidence) -> str:
    terms = " ".join(objective.structural_terms[:6]) or "physics"
    target = objective.target_quantity or "physics"
    return template.format(target=target, terms=terms, method=evidence.method_family)


def _propose_method(objective: PhysicsSearchObjective) -> MethodProposal | None:
    terms_set = set(objective.structural_terms)
    best: tuple[float, MethodEvidence, list[str]] | None = None
    for evidence in _METHOD_EVIDENCE:
        matched = [term for term in evidence.evidence_terms if term in terms_set]
        required_ok = not evidence.required_terms or any(term in terms_set for term in evidence.required_terms)
        if not matched or not required_ok:
            continue
        score = len(matched) / max(1, len(evidence.evidence_terms))
        score += 0.2 if required_ok else 0.0
        if best is None or score > best[0]:
            best = (score, evidence, matched)
    if best is None:
        return None
    score, evidence, matched = best
    query_plan = list(objective.query_plan)
    query_plan.extend(_format_query(template, objective, evidence) for template in evidence.query_templates)
    return MethodProposal(
        method_family=evidence.method_family,
        confidence=min(0.95, score),
        matched_terms=sorted(set(matched)),
        blocked_formula_families=list(evidence.blocked_formula_families),
        query_plan=query_plan,
    )


def _fallback_method_family(parsed: ParsedPhysicsProblem, question: str) -> str | None:
    low = question.lower()
    if parsed.target_quantity == "resistance" and _contains_any(low, ["series", "parallel"]):
        return "network_reduction"
    if parsed.target_quantity == "capacitance" and _contains_any(low, ["series", "parallel"]):
        return "capacitor_network_reduction"
    if parsed.target_quantity == "electric_field":
        return "point_charge_formula_lookup"
    if parsed.target_quantity in {"current", "voltage", "power", "charge", "force", "energy", "potential_energy"}:
        return "direct_physics_relation"
    return None


def infer_problem_frame(parsed: ParsedPhysicsProblem, question: str) -> ProblemFrame:
    objective = _build_search_objective(parsed, question)
    proposal = _propose_method(objective)
    method_family = proposal.method_family if proposal else _fallback_method_family(parsed, question)
    blockers: set[str] = set()
    if proposal and "point_charge" in proposal.blocked_formula_families:
        blockers.update({"electric_field_kq_r2", "electric_field_kq_r2_in_dielectric"})
    if proposal and "simple_network_reduction" in proposal.blocked_formula_families:
        blockers.update({"series_resistance", "parallel_resistance", "series_capacitance", "parallel_capacitance"})
    query_plan = proposal.query_plan if proposal else objective.query_plan
    evidence_hints = proposal.matched_terms if proposal else objective.structural_terms[:8]
    geometry = method_family or "unknown"
    source_type = "method_evidence" if proposal else "objective_only"
    observation_point = "derived_from_question_terms" if objective.structural_terms else None

    return ProblemFrame(
        target_quantity=parsed.target_quantity,
        geometry=geometry,
        source_type=source_type,
        observation_point=observation_point,
        method_family=method_family,
        query_plan=query_plan,
        blocked_formulas=sorted(blockers),
        evidence_hints=sorted(set(evidence_hints)),
    )


def search_unknown_explanation(frame: ProblemFrame, reason: str, details: list[str] | None = None) -> str:
    details = [item for item in (details or []) if item]

    def _detail_clause() -> str | None:
        joined = " | ".join(details).lower()
        if "charge magnitudes are missing" in joined or "missing charge values" in joined:
            return "the magnitudes of the charges are missing, so Coulomb's law cannot be evaluated deterministically"
        if "voltage notes are contradictory" in joined or "contradiction in the given voltage values" in joined:
            return "the voltage values are contradictory, so resistance cannot be determined deterministically"
        if "current notes are contradictory" in joined:
            return "the current values are contradictory, so resistance cannot be determined deterministically"
        if "voltage and current are reported for unrelated circuits" in joined:
            return "the voltage and current belong to unrelated circuits, so resistance cannot be determined deterministically"
        if "ambiguous or conflicting measurements prevent a deterministic resistance calculation" in joined:
            return "the measurements are conflicting, so resistance cannot be determined deterministically"
        if "no deterministic formula matched the question" in joined:
            return "no deterministic physics formula matched the supplied information"
        return None

    base = "The answer is unknown because "
    detail_clause = _detail_clause()
    if detail_clause:
        clause = detail_clause
    elif frame.method_family == "distributed_charge_integration":
        clause = "this is a distributed-charge geometry, so a point-charge shortcut is not safe"
        if frame.blocked_formulas:
            clause += f" and {', '.join(frame.blocked_formulas[:2])} is blocked by the geometry"
    elif frame.method_family == "network_reduction_or_symbolic":
        clause = "this is a bridge/ladder-style resistor network, so a simple series/parallel reduction is not safe"
        if frame.blocked_formulas:
            clause += f" and {', '.join(frame.blocked_formulas[:3])} is blocked"
    elif frame.method_family == "vector_superposition":
        clause = "the geometry needs a vector-superposition setup that is not pinned down enough for a safe shortcut"
    elif frame.method_family == "dielectric_transform":
        clause = "the dielectric state change is underspecified, so the connected/disconnected capacitance relation is not safe to assume"
    elif frame.method_family == "ac_circuit_method":
        clause = "the problem needs AC frequency/reactance context that is not fully established"
    elif frame.method_family == "transformer_relation":
        clause = "the transformer turns-ratio relation is not fully specified"
    elif frame.method_family == "magnetics_geometry":
        clause = "the magnetic geometry is not fully specified for a safe deterministic formula"
    elif frame.method_family == "direct_physics_relation":
        clause = "the given quantities do not determine a supported direct physics relation"
    elif reason:
        clause = reason.replace("_", " ")
    else:
        clause = "no deterministic physics formula matched the supplied information"
    if details:
        clause += f"; evidence/rejection notes: {'; '.join(details[:3])}"
    return base + clause + "."
