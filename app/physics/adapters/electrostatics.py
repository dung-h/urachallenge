"""Electrostatics vector adapter.

This module provides the ElectrostaticsVectorAdapter class, which delegates
geometric point-charge vector questions to the scene parser solver.
"""

from __future__ import annotations

import math

from app.physics.adapters.base import AdapterSolution
from app.physics.equation_graph import EquationGraph
from app.physics.formulas import K_COULOMB
from app.physics.ir import PhysicsProblemIR
from app.physics.scene_parser import solve_physics_scene
from app.physics.unit_converter import format_best_unit


class ElectrostaticsVectorAdapter:
    """Adapter for electrostatics questions involving vector forces or electric fields.

    This adapter delegates charge geometry questions (e.g. collinear, triangle, square)
    to the specialized coordinate-based scene parser solver.
    """
    name = "electrostatics_vector_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        low = ir.question.lower()
        if any(token in low for token in ["charge", "electric field", "electric force", "coulomb", "point charges"]):
            return 0.8
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph for the problem.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            None, as this adapter does not use equation graphs.
        """
        return None

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the electrostatics problem using the scene parser.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            An AdapterSolution, or None if the problem cannot be solved.
        """
        result = solve_physics_scene(ir.question)

        if result is not None:
            # Attach the parsed scene to the IR so the trace can reference the
            # coordinated geometry (Req 5.5).
            ir.scene = result.scene
            if result.success:
                target = result.scene.target
                return AdapterSolution(
                    answer=result.answer,
                    explanation=result.explanation,
                    formula_id=result.formula_id or "coulomb_vector_scene",
                    variables=dict(result.variables),
                    cot=list(result.cot),
                    confidence=result.confidence,
                    trace={
                        "scene_geometry": result.scene.geometry,
                        "target": target.__dict__ if target else None,
                        "points": dict(result.scene.points),
                        "charges": {
                            label: {"value": charge.value, "point": charge.point}
                            for label, charge in result.scene.charges.items()
                        },
                        # Per-charge contribution magnitude/direction and the resultant
                        # Ex/Ey (or Fx/Fy) vector sum that produced the answer (Req 5.5).
                        "contributions": [dict(contribution) for contribution in result.contributions],
                        "resultant": dict(result.resultant),
                    },
                )

        # Fallback: when the scene parser cannot build a coordinated scene
        # (because the question lacks explicit geometry but still names two
        # point charges and a separation, or three identical charges at the
        # vertices of an equilateral triangle), apply the structural
        # Coulomb-magnitude formula directly. This is a *structural* path
        # (variable shapes only), not a question-text override (AGENTS.md §20).
        coulomb = _coulomb_pair_force(ir)
        if coulomb is not None:
            return coulomb
        triangle = _equilateral_triangle_net_force(ir)
        if triangle is not None:
            return triangle
        return None


def _is_force_target(ir: PhysicsProblemIR) -> bool:
    """Whether the parsed problem asks for a force-magnitude answer."""
    target = ir.target.quantity if ir.target else None
    if target is None:
        return False
    return target == "force"


def _coulomb_pair_force(ir: PhysicsProblemIR) -> AdapterSolution | None:
    """Compute |F| = k|q1 q2|/r^2 for an unambiguous two-charge, one-distance scene.

    Only fires when the IR carries exactly two charge values and one distance,
    the question targets a force, and no geometry signal (triangle, square,
    midpoint, third charge, etc.) is present that would require the vector
    scene solver. This complements the scene parser, which abstains when the
    question text lacks explicit coordinates.
    """
    if not _is_force_target(ir):
        return None
    charges = [q.value for q in ir.quantities if q.si_unit == "C"]
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    if len(charges) != 2 or len(distances) != 1:
        return None
    low = ir.question.lower()
    # Reject geometric arrangements that need the vector scene solver. We use
    # word-boundary or explicit phrase tokens to avoid the substring false
    # positives that plague short keywords like "ac"/"ma".
    geometric_phrases = [
        "triangle", "equilateral", "square", "rectangle", "polygon",
        "midpoint", "perpendicular bisector", "vertices", "vertex",
        "third charge", "three charges", "three identical", "third point",
    ]
    if any(phrase in low for phrase in geometric_phrases):
        return None
    q1, q2 = charges[0], charges[1]
    r = distances[0]
    if r <= 0:
        return None
    force = K_COULOMB * abs(q1 * q2) / (r ** 2)
    answer = format_best_unit(force, "N")
    return AdapterSolution(
        answer=answer,
        explanation=(
            "Applied Coulomb's law for two point charges separated by a single distance: "
            "F = k * |q1 * q2| / r^2."
        ),
        formula_id="coulomb_force",
        variables={"q1": q1, "q2": q2, "r": r},
        cot=[
            f"q1 = {q1:.3g} C, q2 = {q2:.3g} C, r = {r:.3g} m",
            f"|F| = k * |q1*q2| / r^2 = {force:.6g} N",
            f"Answer: {answer}",
        ],
        confidence=0.9,
        trace={"formula": "coulomb_force", "k": K_COULOMB},
    )


def _equilateral_triangle_net_force(ir: PhysicsProblemIR) -> AdapterSolution | None:
    """Compute |F_net| for a charge at one vertex of an equilateral triangle.

    Requires three identical charges (or one charge magnitude with a side
    length explicitly described as an equilateral triangle). Two pairwise
    Coulomb forces of equal magnitude meet at 60°, so the resultant is
    sqrt(3) * F0 where F0 = k * q^2 / r^2.
    """
    if not _is_force_target(ir):
        return None
    low = ir.question.lower()
    if "equilateral" not in low or "triangle" not in low:
        return None
    charges = [q.value for q in ir.quantities if q.si_unit == "C"]
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    if not charges or not distances:
        return None
    # Identical-charge case: every charge value matches up to sign. The "net
    # force on one charge" formula uses |q|, so we collapse to a single
    # magnitude. When the question only emits one charge value (the parser
    # de-duplicates "three identical +2 μC charges" to a single number), reuse it.
    magnitudes = sorted({abs(value) for value in charges})
    if len(magnitudes) > 1:
        # Not strictly identical; abstain rather than guess which pair to use.
        return None
    q = magnitudes[0]
    r = distances[0]
    if q <= 0 or r <= 0:
        return None
    f0 = K_COULOMB * (q ** 2) / (r ** 2)
    force = math.sqrt(3.0) * f0
    answer = format_best_unit(force, "N")
    return AdapterSolution(
        answer=answer,
        explanation=(
            "Three identical charges at the vertices of an equilateral triangle exert two "
            "pairwise Coulomb forces of equal magnitude F0 on any one charge, separated by 60°. "
            "Their vector sum has magnitude sqrt(3) * F0 with F0 = k * q^2 / r^2."
        ),
        formula_id="coulomb_equilateral_triangle_net_force",
        variables={"q": q, "r": r, "F0": f0},
        cot=[
            f"q = {q:.3g} C, side r = {r:.3g} m",
            f"F0 = k * q^2 / r^2 = {f0:.6g} N",
            f"|F_net| = sqrt(3) * F0 = {force:.6g} N",
            f"Answer: {answer}",
        ],
        confidence=0.9,
        trace={"formula": "coulomb_equilateral_triangle_net_force", "k": K_COULOMB},
    )
