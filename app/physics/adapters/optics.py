"""Optics problem adapter.

This module provides the OpticsAdapter class, which solves small geometric-optics
problems deterministically (AGENTS.md §13.2):

  * Snell's law refraction:  n1 * sin(theta1) = n2 * sin(theta2)
      -> theta2 = asin(n1 * sin(theta1) / n2)
  * Thin lens / mirror equation:  1/f = 1/do + 1/di
      -> di = 1 / (1/f - 1/do)
      Convex lens and concave mirror use focal length f > 0; a mirror's focal
      length is derived from its radius of curvature R via f = R / 2.

Refractive indices ("n = 1.5") and bare angles ("30 degrees") are not standard
SI quantities, so this adapter extracts them with conservative, general regex
patterns rather than relying on the unit table. All arithmetic is deterministic
Python; no LLM is involved.
"""

from __future__ import annotations

import math
import re

from app.physics.adapters.base import AdapterSolution
from app.physics.ir import PhysicsProblemIR
from app.physics.unit_converter import format_best_unit


class OpticsAdapter:
    """Deterministic adapter for small geometric-optics relations."""

    name = "optics_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Positive score for refraction (Snell) and thin lens/mirror image
        problems. Structural signal based on optics tokens.
        """
        low = ir.question.lower()
        if _asks_refraction_angle(low) and _extract_refraction_inputs(ir) is not None:
            return 0.8
        if _asks_image_distance(low) and _extract_lens_mirror_inputs(ir) is not None:
            return 0.8
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR):  # noqa: D401 - graph unused
        """Optics relations are solved directly in :meth:`solve`.

        The lens/mirror and Snell relations are single closed-form expressions,
        so a full equation graph is unnecessary. Returns None.
        """
        return None

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the optics problem deterministically."""
        low = ir.question.lower()
        if _asks_refraction_angle(low):
            return _solve_refraction(ir)
        if _asks_image_distance(low):
            return _solve_lens_mirror(ir)
        return None


# ---------------------------------------------------------------------------
# Question-shape detectors (general phrasing, never per-question text).
# ---------------------------------------------------------------------------


def _asks_refraction_angle(low: str) -> bool:
    """Checks if the question asks for a refraction (or refracted) angle."""
    if "refraction angle" in low or "angle of refraction" in low or "refracted angle" in low:
        return True
    if "refract" in low and "angle" in low:
        return True
    return False


def _asks_image_distance(low: str) -> bool:
    """Checks if the question asks where the image forms for a lens or mirror."""
    if not any(token in low for token in ["lens", "mirror"]):
        return False
    return any(
        token in low
        for token in [
            "image", "where is the image", "image distance", "image formed",
            "image form", "image position", "image located",
        ]
    )


# ---------------------------------------------------------------------------
# Snell's law refraction.
# ---------------------------------------------------------------------------


def _extract_refractive_indices(question: str) -> list[float]:
    """Extract refractive index values from phrasings like "n = 1.5" or "n1=1".

    General pattern: the symbol n (optionally subscripted) followed by a number.
    Refractive indices are dimensionless and >= 1 for ordinary media.
    """
    indices: list[float] = []
    for m in re.finditer(r"\bn\s*_?\d?\s*=\s*([0-9]+(?:\.[0-9]+)?)", question, re.I):
        try:
            indices.append(float(m.group(1)))
        except ValueError:
            continue
    return indices


def _extract_bare_angle_deg(question: str) -> float | None:
    """Extract an angle in degrees from "30 degrees", "30°", or "30 deg"."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|deg|°)", question, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_refraction_inputs(ir: PhysicsProblemIR) -> tuple[float, float, float] | None:
    """Resolve (n1, n2, incident_angle_deg) for a Snell's-law refraction.

    Requires two refractive indices and an incident angle. The first index is
    the source medium (where light starts), the second is the destination.
    """
    indices = _extract_refractive_indices(ir.question)
    angle_deg = _extract_bare_angle_deg(ir.question)
    if len(indices) < 2 or angle_deg is None:
        return None
    return indices[0], indices[1], angle_deg


def _solve_refraction(ir: PhysicsProblemIR) -> AdapterSolution | None:
    """Solves for the refraction angle using Snell's law."""
    inputs = _extract_refraction_inputs(ir)
    if inputs is None:
        return None
    n1, n2, theta1_deg = inputs
    if n2 == 0:
        return None
    sin_theta2 = n1 * math.sin(math.radians(theta1_deg)) / n2
    # Total internal reflection: no real refraction angle.
    if not (-1.0 <= sin_theta2 <= 1.0):
        return None
    theta2_deg = math.degrees(math.asin(sin_theta2))
    answer = f"{theta2_deg:.6g}°"
    return AdapterSolution(
        answer=answer,
        explanation=(
            f"Applied Snell's law n1·sin(θ1) = n2·sin(θ2) with n1={n1}, n2={n2}, "
            f"θ1={theta1_deg}°: θ2 = asin(n1·sin(θ1)/n2) = {theta2_deg:.4g}°. "
            "Computed with backend arithmetic."
        ),
        formula_id="optics_snell_refraction",
        variables={"n1": n1, "n2": n2, "incident_angle_deg": theta1_deg, "refraction_angle_deg": theta2_deg},
        cot=[
            "Snell's law: n1·sin(θ1) = n2·sin(θ2)",
            f"sin(θ2) = {n1}·sin({theta1_deg}°)/{n2} = {sin_theta2:.6g}",
            f"θ2 = {theta2_deg:.6g}°",
        ],
        confidence=0.9,
        trace={"method": "snell_law"},
    )


# ---------------------------------------------------------------------------
# Thin lens / spherical mirror equation.
# ---------------------------------------------------------------------------


def _extract_focal_length_m(question: str) -> float | None:
    """Extract a focal length in metres from "focal length 20 cm"."""
    m = re.search(
        r"focal\s+length\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b",
        question,
        re.I,
    )
    if not m:
        return None
    return _length_to_m(m.group(1), m.group(2))


def _extract_radius_of_curvature_m(question: str) -> float | None:
    """Extract a radius of curvature in metres from "radius of curvature 40 cm"."""
    m = re.search(
        r"radius\s+of\s+curvature\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b",
        question,
        re.I,
    )
    if not m:
        return None
    return _length_to_m(m.group(1), m.group(2))


def _extract_object_distance_m(question: str) -> float | None:
    """Extract the object distance in metres.

    Matches "object is placed 30 cm from", "object at 30 cm", "object distance
    of 30 cm", "placed 30 cm from the lens/mirror".
    """
    patterns = [
        r"object\s+(?:is\s+)?(?:placed\s+)?(?:at\s+)?(?:a\s+distance\s+of\s+)?([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b",
        r"object\s+distance\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\b",
        r"placed\s+([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\s+from",
        r"(?:is\s+)?(?:at\s+)?([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)\s+from\s+the\s+(?:lens|mirror)",
    ]
    for pattern in patterns:
        m = re.search(pattern, question, re.I)
        if m:
            return _length_to_m(m.group(1), m.group(2))
    return None


def _length_to_m(value: str, unit: str) -> float | None:
    """Convert a numeric string with a length unit to metres."""
    try:
        numeric = float(value)
    except ValueError:
        return None
    unit = unit.lower()
    if unit == "cm":
        return numeric * 1e-2
    if unit == "mm":
        return numeric * 1e-3
    return numeric


def _extract_lens_mirror_inputs(ir: PhysicsProblemIR) -> tuple[float, float, str] | None:
    """Resolve (focal_length_m, object_distance_m, kind) for the lens/mirror eq.

    ``kind`` is "lens" or "mirror". The focal length comes directly from a
    "focal length" mention, or from a mirror's radius of curvature via f = R/2.
    """
    low = ir.question.lower()
    object_distance = _extract_object_distance_m(ir.question)
    if object_distance is None or object_distance <= 0:
        return None
    focal_length = _extract_focal_length_m(ir.question)
    kind = "lens" if "lens" in low else ("mirror" if "mirror" in low else "lens")
    if focal_length is None:
        radius = _extract_radius_of_curvature_m(ir.question)
        if radius is not None:
            focal_length = radius / 2.0
    if focal_length is None or focal_length <= 0:
        return None
    return focal_length, object_distance, kind


def _solve_lens_mirror(ir: PhysicsProblemIR) -> AdapterSolution | None:
    """Solves for the image distance using the thin lens / mirror equation."""
    inputs = _extract_lens_mirror_inputs(ir)
    if inputs is None:
        return None
    focal_length, object_distance, kind = inputs
    denom = (1.0 / focal_length) - (1.0 / object_distance)
    if abs(denom) < 1e-12:
        # Object at focal point: image at infinity.
        return None
    image_distance = 1.0 / denom
    answer = format_best_unit(image_distance, "m")
    relation = "1/f = 1/d_o + 1/d_i"
    return AdapterSolution(
        answer=answer,
        explanation=(
            f"Applied the thin {kind} equation {relation} with f={focal_length:.4g} m and "
            f"d_o={object_distance:.4g} m: d_i = 1/(1/f − 1/d_o) = {image_distance:.4g} m. "
            "Computed with backend arithmetic."
        ),
        formula_id=f"optics_{kind}_equation",
        variables={
            "focal_length_m": focal_length,
            "object_distance_m": object_distance,
            "image_distance_m": image_distance,
        },
        cot=[
            f"Thin {kind} equation: {relation}",
            f"1/d_i = 1/{focal_length:.4g} − 1/{object_distance:.4g} = {denom:.6g}",
            f"d_i = {image_distance:.6g} m",
        ],
        confidence=0.9,
        trace={"method": f"thin_{kind}_equation"},
    )
