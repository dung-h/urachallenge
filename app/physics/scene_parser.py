"""Physics scene parser and vector solver.

This module parses descriptions of point charge arrangements (e.g. on triangles,
squares, or collinear configurations), assigns coordinates, performs vector
summation of Coulomb forces and electric fields, and returns structured solutions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.physics.dimensions import dimension_for_unit, dimensions_compatible
from app.physics.formulas import K_COULOMB
from app.physics.unit_converter import (
    NUMBER_PATTERN,
    _normalize_superscript_powers_of_ten,
    extract_quantities,
    format_best_unit,
)


@dataclass(frozen=True)
class SceneCharge:
    """Represents a point charge in the scene.

    Attributes:
        label: The name of the charge (e.g., 'q1', 'q2').
        value: The electrical charge value in Coulombs.
        point: The geometry vertex/point where this charge is placed.
    """
    label: str
    value: float | None
    point: str | None = None


@dataclass(frozen=True)
class SceneTarget:
    """Represents the target quantity being calculated.

    Attributes:
        quantity: The type of target (e.g., 'electric_field', 'force_magnitude').
        charge_label: The label of the charge under consideration, if any.
        point_label: The label of the point under consideration, if any.
    """
    quantity: str
    charge_label: str | None = None
    point_label: str | None = None


@dataclass
class PhysicsScene:
    """A parsed configuration of charges, points, and target quantities.

    Attributes:
        question: The raw text of the question.
        charges: Dict mapping charge labels to SceneCharge instances.
        points: Dict mapping point names to 2D coordinates.
        target: The target SceneTarget being solved for.
        geometry: The identified geometry configuration name.
        evidence: Supporting extraction evidence logs.
        errors: Error messages encountered during parsing.
    """
    question: str
    charges: dict[str, SceneCharge] = field(default_factory=dict)
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    target: SceneTarget | None = None
    geometry: str | None = None
    evidence: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SceneSolveResult:
    """The structured result of solving a physics scene.

    Attributes:
        success: True if the scene was successfully parsed and solved.
        answer: The final formatted answer string with units.
        explanation: Natural-language explanation of the solving steps.
        formula_id: The ID of the primary physics formula used.
        variables: Extracted variables and intermediate values in SI.
        cot: Chain-of-thought trace steps.
        confidence: Confidence score of the solution.
        scene: The underlying PhysicsScene instance.
        error: Error code or category if the solve failed.
        contributions: Per-charge force/field component vector steps.
        resultant: Final resultant vector components and magnitude.
    """
    success: bool
    answer: str
    explanation: str
    formula_id: str | None
    variables: dict[str, float]
    cot: list[str]
    confidence: float
    scene: PhysicsScene
    error: str | None = None
    # Structured per-charge field/force contributions (magnitude + direction +
    # components) plus the resultant vector, used to build the computation trace
    # (Req 5.5). Empty for symbolic/symmetry results that have no numeric vectors.
    contributions: list[dict[str, Any]] = field(default_factory=list)
    resultant: dict[str, float] = field(default_factory=dict)



def _low(question: str) -> str:
    """Normalizes and lowercases the question text."""
    normalized = _normalize_superscript_powers_of_ten(question)
    return normalized.replace("−", "-").replace("–", "-").replace("—", "-").replace("µ", "μ").lower()


def _charge_value(value: str, unit: str) -> float | None:
    """Extracts a charge value in Coulombs from value and unit string."""
    quantities = extract_quantities(f"{value} {unit}")
    for quantity in quantities:
        if quantity.si_unit == "C":
            return quantity.si_value
    return None


def _charge_unit_pattern() -> str:
    """Returns a regex pattern string matching typical charge units."""
    return r"(?:microcoulombs?|microc|μc|uc|nanocoulombs?|nanoc|nc|coulombs?|c)"


def _normalize_label(label: str) -> str:
    """Normalizes charge labels by lowercasing and standardizing primes."""
    return label.lower().replace("qo", "q0").replace("q′", "qprime").replace("q'", "qprime")


def _extract_charges(question: str) -> dict[str, SceneCharge]:
    """Extracts all named charges from a question text."""
    text = _low(question).replace("q′", "qprime").replace("q'", "qprime")
    unit = _charge_unit_pattern()
    charges: dict[str, float] = {}

    equal_match = re.search(
        rf"\b(?P<labels>q[a-z0-9]?(?:\s*=\s*q[a-z0-9]?)+)\s*=\s*(?P<value>{NUMBER_PATTERN})\s*(?P<unit>{unit})\b",
        text,
        re.I,
    )
    if equal_match:
        value = _charge_value(equal_match.group("value"), equal_match.group("unit"))
        if value is not None:
            for label in re.findall(r"q[a-z0-9]?", equal_match.group("labels"), re.I):
                charges[_normalize_label(label)] = value

    opposite_match = re.search(
        rf"\b(q[0-9a-z]?)\s*=\s*-\s*(q[0-9a-z]?)\s*=\s*(?P<value>{NUMBER_PATTERN})\s*(?P<unit>{unit})\b",
        text,
        re.I,
    )
    if opposite_match:
        value = _charge_value(opposite_match.group("value"), opposite_match.group("unit"))
        if value is not None:
            charges[_normalize_label(opposite_match.group(1))] = value
            charges[_normalize_label(opposite_match.group(2))] = -value

    for match in re.finditer(rf"\b(?P<label>q(?:prime|[a-z0-9])?)\s*=\s*(?P<value>[+-]?\s*{NUMBER_PATTERN})\s*(?P<unit>{unit})\b", text, re.I):
        value = _charge_value(match.group("value").replace(" ", ""), match.group("unit"))
        if value is not None:
            charges[_normalize_label(match.group("label"))] = value

    # Re-apply after single-label extraction so q1=-q2=... is not flattened to both positive.
    if opposite_match:
        value = _charge_value(opposite_match.group("value"), opposite_match.group("unit"))
        if value is not None:
            charges[_normalize_label(opposite_match.group(1))] = value
            charges[_normalize_label(opposite_match.group(2))] = -value

    identical = re.search(rf"\b(?:identical charges|same magnitude q|charges q)\s*=\s*(?P<value>[+-]?\s*{NUMBER_PATTERN})\s*(?P<unit>{unit})\b", text, re.I)
    if identical:
        value = _charge_value(identical.group("value").replace(" ", ""), identical.group("unit"))
        if value is not None:
            charges.setdefault("q", value)

    two_unlabeled = re.search(rf"\btwo\s+(?P<value>[+-]\s*{NUMBER_PATTERN})\s*(?P<unit>{unit})\s+charges\b", text, re.I)
    if two_unlabeled:
        value = _charge_value(two_unlabeled.group("value").replace(" ", ""), two_unlabeled.group("unit"))
        if value is not None:
            charges.setdefault("q1", value)
            charges.setdefault("q2", value)

    if "test charge" in text and "direction" in text and "q0" not in charges:
        # A direction-only question only needs the conventional positive test-charge direction.
        charges["q0"] = 1.0

    result = {label: SceneCharge(label=label, value=value) for label, value in charges.items()}

    if "two identical charges" in text and "qprime" in result and "q" in result:
        result.setdefault("q1", SceneCharge("q1", result["q"].value))
        result.setdefault("q2", SceneCharge("q2", result["q"].value))
        result.pop("q", None)
    if "three identical charges" in text and "q" in result:
        value = result["q"].value
        result = {
            "q1": SceneCharge("q1", value),
            "q2": SceneCharge("q2", value),
            "q3": SceneCharge("q3", value),
        }
    return result


def _extract_distances(question: str) -> list[float]:
    """Extracts all distances in meters from the question text."""
    return [quantity.si_value for quantity in extract_quantities(question) if quantity.si_unit == "m"]


def _extract_labeled_distances(question: str) -> dict[str, float]:
    """Extracts labeled distances (like AB = 5 cm) from the question text."""
    text = _low(question)
    pairs: dict[str, float] = {}
    for match in re.finditer(r"\b([a-z])([a-z])\s*=\s*([a-z])([a-z])\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b", text, re.I):
        a, b, c, d, value_text, unit = match.groups()
        if any(point not in "abcdmo" for point in [a, b, c, d]):
            continue
        value = float(value_text)
        if unit == "cm":
            value *= 1e-2
        elif unit == "mm":
            value *= 1e-3
        for first, second in [(a, b), (c, d)]:
            pairs[first + second] = value
            pairs[second + first] = value
    for match in re.finditer(r"\b([a-z])([a-z])\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b", text, re.I):
        a, b, value_text, unit = match.groups()
        if a not in "abcdmo" or b not in "abcdmo":
            continue
        value = float(value_text)
        if unit == "cm":
            value *= 1e-2
        elif unit == "mm":
            value *= 1e-3
        pairs[a + b] = value
        pairs[b + a] = value

    apart = re.search(r"\b(?:points?\s+)?([a-z])\s+and\s+([a-z]).{0,80}?([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\s+apart\b", text, re.I)
    if apart:
        a, b, value_text, unit = apart.groups()
        if a in "abcdmo" and b in "abcdmo":
            value = float(value_text)
            if unit == "cm":
                value *= 1e-2
            elif unit == "mm":
                value *= 1e-3
            pairs[a + b] = value
            pairs[b + a] = value
    separated_ab = re.search(r"\bpoints?\s+a\s+and\s+b.{0,100}?(?:separated|apart).{0,30}?([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b", text, re.I)
    if not separated_ab:
        separated_ab = re.search(r"\bseparated\s+by\s+(?:a\s+distance\s+of\s+)?([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b", text, re.I)
    if separated_ab:
        value = float(separated_ab.group(1))
        unit = separated_ab.group(2)
        if unit == "cm":
            value *= 1e-2
        elif unit == "mm":
            value *= 1e-3
        pairs.setdefault("ab", value)
        pairs.setdefault("ba", value)
    return pairs


def _extract_side_length(question: str) -> float | None:
    """Extracts side length of regular geometric figures from the question text."""
    text = _low(question)
    patterns = [
        r"\bside(?:\s+length)?(?:\s+of)?\s*(?:a\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b",
        r"\bsides?\s+of\s+(?:length\s+)?(?:a\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b",
        r"\bwith\s+sides?\s+of\s+length\s+a\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b",
        r"\blegs?\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b",
        r"\ba\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "cm":
            return value * 1e-2
        if unit == "mm":
            return value * 1e-3
        return value
    return None



def _charge_count_phrase(question: str) -> int | None:
    """Number of identically-described charges mentioned ("three equal charges")."""

    text = _low(question)
    words = {"two": 2, "three": 3, "four": 4}
    for word, count in words.items():
        # Allow an arbitrary short run of adjectives between the count word and
        # "charges" (e.g. "three equal, like-signed electric charges").
        if re.search(rf"\b{word}\b[a-z,\s'\u2018\u2019-]{{0,40}}?\bcharges?\b", text):
            return count
    return None


def _drop_redundant_generic_q(charges: dict[str, SceneCharge]) -> dict[str, SceneCharge]:
    """Remove a leftover generic ``q`` when ``q1..qn`` already cover it.

    The charge extractor can capture both ``q1 = q2 = q3 = q`` (as q1..q3) and a
    standalone ``q`` magnitude. Keeping the orphan ``q`` would fail validation
    because it has no assigned point, so it is dropped when its magnitude is
    already represented by the numbered charges.
    """

    numbered = {label: charge for label, charge in charges.items() if re.fullmatch(r"q[1-9]", label)}
    if "q" in charges and numbered:
        q_value = charges["q"].value
        if q_value is None or any(
            charge.value is not None and math.isclose(charge.value, q_value, rel_tol=1e-9, abs_tol=1e-24)
            for charge in numbered.values()
        ):
            return {label: charge for label, charge in charges.items() if label != "q"}
    return charges


def _expand_identical_charges(question: str, charges: dict[str, SceneCharge], needed: int) -> dict[str, SceneCharge]:
    """Expand a single described magnitude into ``needed`` per-vertex charges.

    Handles phrasings such as "three equal positive point charges, q = 5e-9 C"
    or "three positive charges q1 = q2 = q3 = q" where the parser captures one
    magnitude (and possibly a stray generic ``q``). Component-level, not
    question-specific: it keys on the count phrase and a shared magnitude.
    """

    labeled = {label: charge for label, charge in charges.items() if re.fullmatch(r"q[1-9]", label)}
    generic = charges.get("q")
    magnitude: float | None = None
    if generic is not None and generic.value is not None:
        magnitude = generic.value
    elif labeled:
        values = {round(charge.value, 18) for charge in labeled.values() if charge.value is not None}
        if len(values) == 1:
            magnitude = next(iter(labeled.values())).value
    if magnitude is None:
        return charges
    expanded = {f"q{index}": SceneCharge(label=f"q{index}", value=magnitude) for index in range(1, needed + 1)}
    # Preserve any explicitly different labeled charges (e.g. a separate test charge q0).
    for label, charge in charges.items():
        if label not in expanded and not re.fullmatch(r"q[1-9]?", label):
            expanded[label] = charge
    return expanded


def _target_label(question: str, charges: dict[str, SceneCharge]) -> str | None:
    text = _low(question).replace("q′", "qprime").replace("q'", "qprime")
    for pattern in [
        r"\b(?:force|forces).{0,80}?\b(?:on|acting on|exerted on)\s+(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:force|forces).{0,100}?\b(?:on|acting on|exerted on)\s+(?:charge\s+)?(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:force|forces).{0,100}?\b(?:on|acting on|exerted on)\s+(?:a\s+|the\s+)?charge\s+(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:force|forces).{0,100}?\b(?:on|acting on|exerted on)\s+(?:a\s+|the\s+)?test\s+charge\s+(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:on|acting on|exerted on)\s+(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:on|acting on|exerted on)\s+(?:charge\s+)?(q(?:prime|[a-z0-9])?)\b",
        r"\b(?:on|acting on|exerted on)\s+(?:a\s+|the\s+)?charge\s+(q(?:prime|[a-z0-9])?)\b",
        r"\bforce\s+acting\s+on\s+(?:the\s+)?charge\s+([a-z])\b",
        r"\bcharge\s+at\s+(?:the\s+)?right\s+angle\s+vertex\b",
    ]:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        if "right angle vertex" in match.group(0):
            if "q3" in charges:
                return "q3"
            if "q0" in charges:
                return "q0"
            continue
        if not match.lastindex:
            continue
        raw_label = match.group(1)
        label = _normalize_label("q" + raw_label) if len(raw_label) == 1 and not raw_label.lower().startswith("q") else _normalize_label(raw_label)
        if label in charges:
            return label
    if "qprime" in charges and "remaining vertex" in text:
        return "qprime"
    if "q0" in charges and ("test charge" in text or "charge q0" in text):
        return "q0"
    if "q0" in charges and "direction" in text:
        return "q0"
    if "q3" in charges and any(token in text for token in ["acting on q3", "on q3", "net force vector acting on q3"]):
        return "q3"
    if "q2" in charges and "force acting on q2" in text:
        return "q2"
    if "q" in charges and "acting on q" in text:
        return "q"
    if "q" in charges and "test charge q" in text and any(token in text for token in ["force", "acting on", "exerted on"]):
        return "q"
    return None


def _target_quantity(question: str) -> str:
    text = _low(question)
    if "electric field" in text or "field strength" in text or "field intensity" in text:
        return "electric_field"
    if "direction" in text:
        return "force_direction"
    return "force_magnitude"


# Filler words that can sit between "at" and the actual point/vertex label.
_POINT_FILLER = r"(?:the\s+|point\s+|vertex\s+|corner\s+)*"


def _target_point(question: str) -> str | None:
    text = _low(question)
    if "center" in text or "centre" in text or ("intersection" in text and "diagonal" in text):
        return "O"
    if "foot of the altitude" in text or "foot of altitude" in text:
        return "H"
    if "fourth vertex" in text or "the fourth" in text or "4th vertex" in text:
        # The square builder resolves this to the unoccupied vertex.
        return "FOURTH"
    if "midpoint" in text or "mid-point" in text or "mid point" in text:
        return "M"
    if "perpendicular bisector" in text and "point m" not in text:
        return "M"
    # Apex point N that forms a named triangle with the two source points A and B.
    if re.search(r"\bpoint\s+n\b", text):
        return "N"
    for pattern in [
        rf"\b(?:electric field|field strength|field intensity).{{0,80}}?\bat\s+{_POINT_FILLER}([a-z])\b",
        rf"\bat\s+{_POINT_FILLER}([a-z])\b.{{0,80}}?\b(?:electric field|field strength|field intensity)\b",
        r"\bpoint\s+([a-z])\s+(?:lies|is|which|where)\b",
    ]:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return None


def _non_air_dielectric(question: str) -> bool:
    """Detect a non-vacuum/non-air dielectric medium.

    The deterministic vector solver uses the vacuum Coulomb constant. A medium
    with a relative permittivity other than 1 (oil, alcohol, water, glass, an
    explicit epsilon, ...) would change the result, so such scenes must abstain
    rather than report a vacuum-only answer (Req 5.7).
    """

    text = _low(question)
    if re.search(r"dielectric\s+constant", text) or re.search(r"relative\s+permittivit", text):
        return True
    if re.search(r"\b(?:epsilon|ε)\s*=\s*", text):
        # epsilon = 1 (vacuum) is fine; anything else changes the constant.
        match = re.search(r"\b(?:epsilon|ε)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match and abs(float(match.group(1)) - 1.0) > 1e-9:
            return True
        if not match:
            return True
    for medium in ("alcohol", "kerosene", "glycerin", "in oil", "in water", "paraffin"):
        if medium in text:
            return True
    return False


def _named_charge_vertices(question: str, charge_labels: list[str]) -> dict[str, str]:
    """Map ordered charge labels to explicitly named points/vertices.

    Handles phrasings such as "q1 and q2 are placed at vertices B and C" or
    "charges are placed at points A, B and C". The assignment is positional:
    the n-th charge (in label order) is assigned to the n-th named point.
    """

    text = _low(question)
    match = re.search(
        r"\bat\s+(?:vertices|points|vertex|point)\s+"
        r"([a-z])(?:\s*,\s*([a-z]))?(?:\s*,\s*([a-z]))?(?:\s*,?\s*and\s+([a-z]))?\b",
        text,
    )
    if not match:
        return {}
    letters = [group.upper() for group in match.groups() if group]
    # Only accept geometry-point letters; reject accidental matches on stray words.
    if any(letter.lower() not in "abcdmnoh" for letter in letters):
        return {}
    ordered = sorted(charge_labels)
    mapping: dict[str, str] = {}
    for label, letter in zip(ordered, letters):
        mapping[label] = letter
    return mapping


def _assign_default_triangle_points(charges: dict[str, SceneCharge], target: str | None, geometry: str) -> dict[str, str]:
    labels = set(charges)
    points: dict[str, str] = {}
    for label in labels:
        if len(label) == 2 and label[1].isalpha():
            points[label] = label[1].upper()
    if {"q1", "q2", "q3"}.issubset(labels):
        if geometry == "isosceles_right":
            points.update({"q3": "A", "q1": "B", "q2": "C"})
        else:
            points.update({"q1": "A", "q2": "B", "q3": "C"})
    if {"q1", "q2", "qprime"}.issubset(labels):
        points.update({"q1": "A", "q2": "B", "qprime": "C"})
    if {"q1", "q2"}.issubset(labels):
        points.setdefault("q1", "A")
        points.setdefault("q2", "B")
    if target == "q0" and "q0" in labels:
        points["q0"] = "M"
    return points


def _scene_with_updated_charges(scene: PhysicsScene, point_map: dict[str, str]) -> None:
    scene.charges = {
        label: SceneCharge(label=charge.label, value=charge.value, point=point_map.get(label, charge.point))
        for label, charge in scene.charges.items()
    }


def _build_triangle_scene(question: str, charges: dict[str, SceneCharge], target: str | None) -> PhysicsScene | None:
    text = _low(question)
    if "equilateral" not in text and "isosceles right" not in text and "right-angled triangle" not in text and "right angled triangle" not in text and "right isosceles" not in text:
        return None
    geometry = "equilateral" if "equilateral" in text else "isosceles_right"
    quantity = _target_quantity(question)
    # Expand "three equal charges at the vertices" into three coordinated charges.
    if not {"q1", "q2", "q3"}.issubset(charges):
        count = _charge_count_phrase(question)
        if count in (2, 3) and ("vertices" in text or "vertex" in text or "vertices" in text):
            charges = _expand_identical_charges(question, charges, count)
    charges = _drop_redundant_generic_q(charges)
    side = _extract_side_length(question)
    pairs = _extract_labeled_distances(question)
    if side is None and geometry == "equilateral":
        # An equilateral triangle's side may be stated only as a labeled edge AB.
        for edge in ("ab", "bc", "ac"):
            if edge in pairs:
                side = pairs[edge]
                break
    target_point = _target_point(question) if quantity == "electric_field" else None
    scene = PhysicsScene(question=question, charges=dict(charges), target=SceneTarget(quantity, target, target_point), geometry=geometry)
    if geometry == "equilateral":
        if side is None:
            scene.errors.append("equilateral triangle side length missing")
            return scene
        scene.points = {"A": (0.0, 0.0), "B": (side, 0.0), "C": (side / 2.0, math.sqrt(3.0) * side / 2.0)}
        centroid = (side / 2.0, math.sqrt(3.0) * side / 6.0)
        if "center" in text or "centre" in text or target == "q0" or target_point == "O":
            scene.points["O"] = centroid
        # Apex N forms an equilateral triangle with the two source points A and B.
        if target_point == "N":
            scene.points["N"] = (side / 2.0, math.sqrt(3.0) * side / 2.0)
    else:
        leg = side
        if leg is None and pairs.get("ab") is not None and pairs.get("ac") is not None:
            leg = min(pairs["ab"], pairs["ac"])
        if leg is None:
            scene.errors.append("isosceles/right triangle leg length missing")
            return scene
        scene.points = {"A": (0.0, 0.0), "B": (leg, 0.0), "C": (0.0, leg)}
    # Charge placement: explicit named vertices take precedence over defaults.
    named = _named_charge_vertices(question, list(charges))
    point_map = _assign_default_triangle_points(charges, target, geometry)
    point_map.update({label: letter for label, letter in named.items() if letter in scene.points})
    if "O" in scene.points and target == "q0":
        point_map["q0"] = "O"
    _scene_with_updated_charges(scene, point_map)
    return scene


def _build_labeled_distance_scene(question: str, charges: dict[str, SceneCharge], target: str | None) -> PhysicsScene | None:
    pairs = _extract_labeled_distances(question)
    distances = _extract_distances(question)
    text = _low(question)
    if {"q1", "q2", "q0"}.issubset(charges) and len(distances) >= 3 and not {"am", "bm"}.issubset(pairs):
        if "ab" in pairs:
            endpoint_distances = [d for d in distances if not math.isclose(d, pairs["ab"], rel_tol=1e-8, abs_tol=1e-12)]
            sorted_distances = sorted(endpoint_distances)
            if len(sorted_distances) >= 2:
                pairs.update({"am": sorted_distances[0], "ma": sorted_distances[0], "bm": sorted_distances[1], "mb": sorted_distances[1]})
        else:
            sorted_distances = sorted(distances)
            pairs.update({"am": sorted_distances[0], "ma": sorted_distances[0], "bm": sorted_distances[1], "mb": sorted_distances[1]})
            pairs.setdefault("ab", sorted_distances[2])
            pairs.setdefault("ba", sorted_distances[2])
        target = target or "q0"
    if "perpendicular bisector" in text and "ab" not in pairs and len(distances) >= 2:
        pairs["ab"] = max(distances)
        pairs["ba"] = max(distances)
    if ("equidistant" in text or "away from each" in text or "from each charge" in text) and "ab" in pairs:
        equal_distance = None
        non_ab = [d for d in distances if not math.isclose(d, pairs["ab"], rel_tol=1e-8, abs_tol=1e-12)]
        if non_ab:
            equal_distance = non_ab[0]
        elif "distance equal to" in text or "distance equal to 'a'" in text or "distance equal to a" in text:
            equal_distance = pairs["ab"]
        if equal_distance is not None:
            pairs["am"] = pairs["ma"] = equal_distance
            pairs["bm"] = pairs["mb"] = equal_distance
    if not pairs:
        return None
    quantity = _target_quantity(question)
    scene = PhysicsScene(question=question, charges=dict(charges), target=SceneTarget(quantity, target, _target_point(question) if quantity == "electric_field" else None), geometry="labeled_distance")
    charge_points = _assign_default_triangle_points(charges, target, "triangle")
    if "q0" in charges:
        charge_points["q0"] = "M"
    if "q" in charges and "test charge q" in text:
        charge_points["q"] = "M"
    if target == "q0":
        charge_points["q0"] = "M"
    if "ab" in pairs and {"q1", "q2"}.issubset(charges) and target in {"q1", "q2"}:
        ab = pairs["ab"]
        scene.points = {"A": (0.0, 0.0), "B": (ab, 0.0)}
        charge_points.update({"q1": "A", "q2": "B"})
    elif {"ab", "am", "bm"}.issubset(pairs):
        ab, am, bm = pairs["ab"], pairs["am"], pairs["bm"]
        x = (am**2 + ab**2 - bm**2) / (2.0 * ab)
        y = math.sqrt(max(0.0, am**2 - x**2))
        scene.points = {"A": (0.0, 0.0), "B": (ab, 0.0), "M": (x, y)}
    elif {"ab", "ac", "bc"}.issubset(pairs):
        ab, ac, bc = pairs["ab"], pairs["ac"], pairs["bc"]
        x = (ac**2 + ab**2 - bc**2) / (2.0 * ab)
        y = math.sqrt(max(0.0, ac**2 - x**2))
        scene.points = {"A": (0.0, 0.0), "B": (ab, 0.0), "C": (x, y)}
        if "foot of the altitude from a" in text and target in charges:
            ax, ay = scene.points["A"]
            bx, by = scene.points["B"]
            cx, cy = scene.points["C"]
            vx, vy = cx - bx, cy - by
            denom = vx * vx + vy * vy
            if denom > 0:
                t = ((ax - bx) * vx + (ay - by) * vy) / denom
                scene.points["H"] = (bx + t * vx, by + t * vy)
                charge_points[target] = "H"
    elif "perpendicular bisector" in text and "ab" in pairs:
        ab = pairs["ab"]
        offset_candidates = [d for d in distances if not math.isclose(d, ab, rel_tol=1e-8)]
        offset = offset_candidates[0] if offset_candidates else None
        if offset is None:
            scene.errors.append("perpendicular bisector offset missing")
            return scene
        scene.points = {"A": (0.0, 0.0), "B": (ab, 0.0), "M": (ab / 2.0, offset)}
        if target in charges:
            charge_points[target] = "M"
    else:
        return None
    _scene_with_updated_charges(scene, charge_points)
    return scene


def _build_collinear_scene(question: str, charges: dict[str, SceneCharge], target: str | None) -> PhysicsScene | None:
    text = _low(question)
    if not any(token in text for token in ["straight line", "line segment", "midpoint", "extension of line", "opposite sides"]):
        return None
    quantity = _target_quantity(question)
    scene = PhysicsScene(question=question, charges=dict(charges), target=SceneTarget(quantity, target, _target_point(question) if quantity == "electric_field" else None), geometry="collinear")
    distances = _extract_distances(question)
    if "opposite sides" in text and target in charges and len(distances) >= 2:
        labels = [label for label in charges if label != target]
        if len(labels) >= 2:
            scene.points = {"T": (0.0, 0.0), "L": (-distances[0], 0.0), "R": (distances[1], 0.0)}
            point_map = {target: "T", labels[0]: "L", labels[1]: "R"}
            _scene_with_updated_charges(scene, point_map)
            return scene
    if "midpoint" in text and len(distances) >= 1:
        sep = max(distances)
        scene.points = {"A": (0.0, 0.0), "B": (sep, 0.0), "M": (sep / 2.0, 0.0)}
        point_map = _assign_default_triangle_points(charges, target, "triangle")
        if target in charges:
            point_map[target] = "M"
        if "q1" in charges:
            point_map["q1"] = "A"
        if "q2" in charges:
            point_map["q2"] = "B"
        if "q3" in charges and target == "q3":
            point_map["q3"] = "M"
        _scene_with_updated_charges(scene, point_map)
        return scene
    if ("line segment" in text or "extension" in text) and len(distances) >= 2:
        sep = max(distances)
        target_distance = min(distances)
        if "extension" in text and len(distances) >= 3:
            sep = sorted(distances)[1]
            target_distance = sorted(distances)[0]
        if "extension" in text:
            target_x = -target_distance
        else:
            target_x = target_distance
        scene.points = {"A": (0.0, 0.0), "B": (sep, 0.0), "M": (target_x, 0.0)}
        point_map = _assign_default_triangle_points(charges, target, "triangle")
        point_map.setdefault("q1", "A")
        point_map.setdefault("q2", "B")
        if target in charges:
            point_map[target] = "M"
        if "q0" in charges:
            point_map["q0"] = "M"
        _scene_with_updated_charges(scene, point_map)
        return scene
    if "straight line" in text and len(distances) >= 1 and {"q1", "q2", "q3"}.issubset(charges):
        d = distances[0]
        scene.points = {"A": (0.0, 0.0), "B": (d, 0.0), "C": (2.0 * d, 0.0)}
        point_map = {"q1": "A", "q2": "B", "q3": "C"}
        _scene_with_updated_charges(scene, point_map)
        return scene
    return None


def _build_square_center_scene(question: str, charges: dict[str, SceneCharge], target: str | None) -> PhysicsScene | None:
    text = _low(question)
    if "square" not in text:
        return None
    quantity = _target_quantity(question)
    target_point = _target_point(question) if quantity == "electric_field" else None
    wants_center = "center" in text or "centre" in text or ("intersection" in text and "diagonal" in text) or target_point == "O"
    wants_fourth = target_point == "FOURTH"
    if quantity != "electric_field" and not wants_center:
        # Only the center is a supported square force configuration; defer
        # everything else (e.g. a plain two-charge force) to other builders.
        return None
    if quantity == "electric_field" and not (wants_center or wants_fourth or target_point in {"A", "B", "C", "D"}):
        return None
    # Expand "three/four equal charges at the vertices of a square".
    count = _charge_count_phrase(question)
    if count in (3, 4) and not {"q1", "q2", "q3"}.issubset(charges):
        charges = _expand_identical_charges(question, charges, count)
    charges = _drop_redundant_generic_q(charges)
    side = _extract_side_length(question)
    scene = PhysicsScene(question=question, charges=dict(charges), target=SceneTarget(quantity, target, target_point), geometry="square")
    if side is None:
        scene.errors.append("square side length missing")
        return scene
    scene.points = {
        "A": (0.0, 0.0),
        "B": (side, 0.0),
        "C": (side, side),
        "D": (0.0, side),
        "O": (side / 2.0, side / 2.0),
    }
    vertices = ["A", "B", "C", "D"]
    named = _named_charge_vertices(question, list(charges))
    point_map = {"q1": "A", "q2": "B", "q3": "C", "q4": "D"}
    point_map.update({label: letter for label, letter in named.items() if letter in vertices})
    if target in charges and wants_center:
        point_map[target] = "O"
    # Resolve "field at the fourth vertex": place the source charges on the
    # occupied vertices and target the remaining (unoccupied) vertex.
    if wants_fourth:
        occupied = {letter for label, letter in point_map.items() if label in charges and letter in vertices}
        free = [letter for letter in vertices if letter not in occupied]
        if free:
            scene.target = SceneTarget(quantity, target, free[0])
    _scene_with_updated_charges(scene, point_map)
    return scene


def _build_rectangle_scene(question: str, charges: dict[str, SceneCharge], target: str | None) -> PhysicsScene | None:
    text = _low(question)
    if "rectangle" not in text:
        return None
    quantity = _target_quantity(question)
    # Expand "four identical charges at the vertices of a rectangle".
    count = _charge_count_phrase(question)
    if count in (3, 4) and not {"q1", "q2", "q3"}.issubset(charges):
        charges = _expand_identical_charges(question, charges, count)
    charges = _drop_redundant_generic_q(charges)
    pairs = _extract_labeled_distances(question)
    ab = pairs.get("ab")
    ad = pairs.get("ad")
    if ab is None or ad is None:
        return None
    target_point = _target_point(question) if quantity == "electric_field" else None
    scene = PhysicsScene(question=question, charges=dict(charges), target=SceneTarget(quantity, target, target_point), geometry="rectangle")
    scene.points = {"A": (0.0, 0.0), "B": (ab, 0.0), "C": (ab, ad), "D": (0.0, ad)}
    vertices = ["A", "B", "C", "D"]
    named = _named_charge_vertices(question, list(charges))
    point_map = {"q1": "A", "q2": "B", "q3": "C", "q4": "D"}
    point_map.update({label: letter for label, letter in named.items() if letter in vertices})
    # For a force on a vertex charge, the charge stays on its own vertex (no
    # relocation). For a field at an unoccupied vertex, resolve it below.
    if quantity == "electric_field" and target_point == "FOURTH":
        occupied = {letter for label, letter in point_map.items() if label in charges and letter in vertices}
        free = [letter for letter in vertices if letter not in occupied]
        if free:
            scene.target = SceneTarget(quantity, target, free[0])
    _scene_with_updated_charges(scene, point_map)
    return scene


def _is_inverse_charge_problem(question: str) -> bool:
    """Detect "what charge must be placed ... so that ... is zero" design problems.

    These ask for an unknown source charge (answer is in coulombs), not a field
    or force magnitude, so the forward vector solver must not answer them.
    """

    text = _low(question)
    if re.search(r"\b(?:what|which|find|determine|calculate)\b[^.?]*\bcharge\s+(?:q\w*\s+)?(?:must|to|that)\b", text):
        if "so that" in text or "such that" in text or "in order" in text or "zero" in text:
            return True
    if re.search(r"\bdetermine\s+the\s+charge\s+q\w*\s+(?:placed|to be placed)\b", text) and ("zero" in text or "such that" in text):
        return True
    return False


def parse_physics_scene(question: str, llm_client: Any | None = None) -> PhysicsScene | None:
    """Parses a physics question text to build a PhysicsScene instance.

    Args:
        question: The text of the physics question.
        llm_client: An optional LLM client for backup parsing (not authoritative).

    Returns:
        A PhysicsScene instance, or None if the question does not match.
    """
    charges = _extract_charges(question)
    if not charges:
        return None
    target = _target_label(question, charges)

    builders = [
        _build_square_center_scene,
        _build_rectangle_scene,
        _build_labeled_distance_scene,
        _build_collinear_scene,
        _build_triangle_scene,
    ]
    for builder in builders:
        scene = builder(question, charges, target)
        if scene is not None:
            # Soundness guards applied uniformly to every geometry (Req 5.7):
            # the vacuum vector solver must abstain on a non-air dielectric and
            # on inverse "find the unknown charge" design problems.
            if _non_air_dielectric(question):
                if "non-air dielectric medium not supported by vacuum vector solver" not in scene.errors:
                    scene.errors.append("non-air dielectric medium not supported by vacuum vector solver")
            if _is_inverse_charge_problem(question):
                if "inverse charge-design problem not supported by forward vector solver" not in scene.errors:
                    scene.errors.append("inverse charge-design problem not supported by forward vector solver")
            if llm_client is not None and hasattr(llm_client, "parse_physics_scene"):
                scene.evidence.append("llm_scene_parser_hook_available_not_authoritative")
            return scene
    return None


def _validate_scene(scene: PhysicsScene) -> list[str]:
    errors = list(scene.errors)
    if scene.target is None:
        errors.append("target is missing")
    elif scene.target.quantity == "electric_field":
        if not scene.target.point_label:
            errors.append("target field point is missing")
        elif scene.target.point_label not in scene.points:
            errors.append(f"target field point {scene.target.point_label} has no coordinates")
    elif not scene.target.charge_label:
        errors.append("target charge is missing")
    elif scene.target.charge_label not in scene.charges:
        errors.append("target charge not found in scene")
    for charge in scene.charges.values():
        if charge.value is None:
            errors.append(f"charge {charge.label} magnitude missing")
        if charge.point is None:
            errors.append(f"charge {charge.label} point missing")
        elif charge.point not in scene.points:
            errors.append(f"charge {charge.label} point {charge.point} has no coordinates")
    return errors


# Maps a scene target quantity to the SI unit its computed magnitude must carry.
# The deterministic vector solver only ever produces an electric-field strength
# (N/C ≡ V/m) or a force magnitude (N). ``force_direction`` carries no numeric
# magnitude, so it has no dimensional expectation.
_SCENE_TARGET_SI_UNIT: dict[str, str] = {
    "electric_field": "N/C",
    "force_magnitude": "N",
}


def _result_dimension_errors(target_quantity: str | None, si_unit: str) -> list[str]:
    """Validate the produced SI unit against the asked target quantity (Req 5.7).

    Returns a list of abstention reasons; empty when the produced unit is
    dimensionally consistent with the requested quantity. This is a component-
    level soundness gate (no per-question logic): a multi-charge result whose
    produced quantity does not match the requested quantity's dimension is not
    trustworthy and the solver must abstain rather than return it.
    """

    expected_unit = _SCENE_TARGET_SI_UNIT.get(target_quantity or "")
    if expected_unit is None:
        # No numeric-magnitude expectation for this target (e.g. a direction).
        return []
    produced_dim = dimension_for_unit(si_unit)
    expected_dim = dimension_for_unit(expected_unit)
    if produced_dim is None or expected_dim is None:
        return [
            f"result unit {si_unit!r} or expected unit {expected_unit!r} for "
            f"{target_quantity} has no known dimension"
        ]
    if not dimensions_compatible(produced_dim, expected_dim):
        return [
            f"computed result in {si_unit} is dimensionally incompatible with the requested "
            f"{target_quantity} ({expected_unit})"
        ]
    return []


def _scene_abstention(scene: PhysicsScene, errors: list[str], error_code: str) -> SceneSolveResult:
    """Build the standard ``unknown`` abstention result for a scene (Req 5.7)."""

    scene.errors.extend(error for error in errors if error not in scene.errors)
    return SceneSolveResult(
        success=False,
        answer="unknown",
        explanation=(
            "The answer is unknown because the physics scene could not be verified: "
            + "; ".join(errors)
            + "."
        ),
        formula_id=None,
        variables={},
        cot=["PhysicsScene verification failed: " + "; ".join(errors)],
        confidence=0.2,
        scene=scene,
        error=error_code,
    )


def _force_vector_on_target(scene: PhysicsScene) -> tuple[float, float, dict[str, float], list[dict[str, Any]]]:
    assert scene.target is not None and scene.target.charge_label is not None
    target = scene.charges[scene.target.charge_label]
    assert target.value is not None and target.point is not None
    tx, ty = scene.points[target.point]
    fx = fy = 0.0
    variables = {"q_target": target.value}
    contributions: list[dict[str, Any]] = []
    source_index = 1
    for label, source in scene.charges.items():
        if label == target.label or source.value is None or source.point is None:
            continue
        sx, sy = scene.points[source.point]
        dx = tx - sx
        dy = ty - sy
        r2 = dx * dx + dy * dy
        if r2 <= 0:
            raise ValueError(f"source charge {label} overlaps target charge")
        r = math.sqrt(r2)
        magnitude = K_COULOMB * abs(target.value * source.value) / r2
        # Positive product repels target away from source; negative attracts it toward source.
        direction = 1.0 if target.value * source.value > 0 else -1.0
        cx = direction * magnitude * dx / r
        cy = direction * magnitude * dy / r
        fx += cx
        fy += cy
        variables[f"q{source_index}"] = source.value
        variables[f"r{source_index}"] = r
        contributions.append(
            {
                "source": label,
                "kind": "force",
                "q": source.value,
                "r": r,
                "magnitude": magnitude,
                "cx": cx,
                "cy": cy,
                "direction_deg": math.degrees(math.atan2(cy, cx)),
                "sense": "repulsive" if direction > 0 else "attractive",
            }
        )
        source_index += 1
    variables["Fx"] = fx
    variables["Fy"] = fy
    return fx, fy, variables, contributions


def _electric_field_vector_at_target(scene: PhysicsScene) -> tuple[float, float, dict[str, float], list[dict[str, Any]]]:
    assert scene.target is not None and scene.target.point_label is not None
    tx, ty = scene.points[scene.target.point_label]
    ex = ey = 0.0
    variables: dict[str, float] = {}
    contributions: list[dict[str, Any]] = []
    source_index = 1
    for label, source in scene.charges.items():
        if source.value is None or source.point is None:
            continue
        sx, sy = scene.points[source.point]
        dx = tx - sx
        dy = ty - sy
        r2 = dx * dx + dy * dy
        if r2 <= 0:
            raise ValueError(f"source charge {label} overlaps target field point")
        r = math.sqrt(r2)
        scale = K_COULOMB * source.value / (r2 * r)
        cx = scale * dx
        cy = scale * dy
        ex += cx
        ey += cy
        variables[f"q{source_index}"] = source.value
        variables[f"r{source_index}"] = r
        contributions.append(
            {
                "source": label,
                "kind": "field",
                "q": source.value,
                "r": r,
                # Field-contribution magnitude k|q|/r^2 (= |(cx,cy)|).
                "magnitude": K_COULOMB * abs(source.value) / r2,
                "cx": cx,
                "cy": cy,
                "direction_deg": math.degrees(math.atan2(cy, cx)),
                "sense": "away from source" if source.value > 0 else "toward source",
            }
        )
        source_index += 1
    variables["Ex"] = ex
    variables["Ey"] = ey
    return ex, ey, variables, contributions


def _geometry_label(scene: PhysicsScene) -> str:
    """Human-readable geometry name for the computation trace (Req 5.5)."""

    raw = scene.geometry or "point-charge"
    return raw.replace("_", " ")


def _format_contribution_lines(contributions: list[dict[str, Any]], symbol: str) -> list[str]:
    """One trace line per source charge: contribution magnitude and direction.

    Derived entirely from the computed per-charge contributions, so it
    generalizes to any multi-charge scene (Req 5.5) with no per-question logic.
    """

    lines: list[str] = []
    for contribution in contributions:
        source = contribution["source"]
        magnitude = contribution["magnitude"]
        direction = contribution["direction_deg"]
        sense = contribution.get("sense", "")
        cx = contribution["cx"]
        cy = contribution["cy"]
        sense_suffix = f", {sense}" if sense else ""
        lines.append(
            f"{symbol} from {source} (q={contribution['q']:.6g} C, r={contribution['r']:.6g} m): "
            f"|{symbol}|={magnitude:.6g}, direction={direction:.6g}° "
            f"({symbol}x={cx:.6g}, {symbol}y={cy:.6g}{sense_suffix})"
        )
    return lines


def _build_scene_cot(
    scene: PhysicsScene,
    contributions: list[dict[str, Any]],
    symbol: str,
    resultant_x: float,
    resultant_y: float,
    final_line: str,
) -> list[str]:
    """Assemble the multi-charge computation trace.

    Names the geometry, lists each per-charge contribution magnitude and
    direction, and records the resultant vector sum (Req 5.5).
    """

    cot = [f"Parsed {_geometry_label(scene)} scene with {len(contributions)} source charge(s)"]
    cot.extend(_format_contribution_lines(contributions, symbol))
    cot.append(
        f"Resultant vector sum: {symbol}x={resultant_x:.6g}, {symbol}y={resultant_y:.6g}"
    )
    cot.append(final_line)
    return cot


def solve_physics_scene(question: str, llm_client: Any | None = None) -> SceneSolveResult | None:
    """Solves a physics question by parsing the scene and calculating force or field vectors.

    Args:
        question: The text of the physics question.
        llm_client: An optional LLM client helper.

    Returns:
        A SceneSolveResult containing the answer and trace steps, or None.
    """
    symbolic = _solve_symbolic_scene(question)
    if symbolic is not None:
        return symbolic

    scene = parse_physics_scene(question, llm_client=llm_client)

    if scene is None:
        return None
    errors = _validate_scene(scene)
    if errors:
        return _scene_abstention(scene, errors, "physics_scene_verification_failed")
    try:
        if scene.target and scene.target.quantity == "electric_field":
            ex, ey, variables, contributions = _electric_field_vector_at_target(scene)
            magnitude = math.hypot(ex, ey)
            answer = format_best_unit(magnitude, "N/C")
            # Result-level dimensional/unit validation (Req 5.7): the produced
            # quantity must match the requested quantity's dimension, else abstain.
            dimension_errors = _result_dimension_errors(scene.target.quantity, "N/C")
            if dimension_errors:
                return _scene_abstention(scene, dimension_errors, "physics_scene_dimensional_validation_failed")
            geometry_name = _geometry_label(scene)
            explanation = (
                f"Built a verified {geometry_name} scene, assigned coordinates to the source charges, "
                f"summed Coulomb electric-field vectors at {scene.target.point_label}, and computed {answer}."
            )
            cot = _build_scene_cot(
                scene,
                contributions,
                "E",
                ex,
                ey,
                f"Resultant magnitude at {scene.target.point_label}: {answer}",
            )
            fid = "coulomb_vector_field_scene"
            if "midpoint" in scene.question.lower() and scene.geometry == "collinear" and len(scene.charges) >= 2:
                fid = "electric_field_two_charge_midpoint"
            return SceneSolveResult(
                success=True,
                answer=answer,
                explanation=explanation,
                formula_id=fid,
                variables=variables,
                cot=cot,
                confidence=0.9,
                scene=scene,
                contributions=contributions,
                resultant={"Ex": ex, "Ey": ey, "magnitude": magnitude},
            )
        fx, fy, variables, contributions = _force_vector_on_target(scene)
    except Exception as exc:
        return SceneSolveResult(
            success=False,
            answer="unknown",
            explanation=f"The answer is unknown because the verified scene could not be solved deterministically: {exc}.",
            formula_id=None,
            variables={},
            cot=[f"PhysicsScene solve failed: {exc}"],
            confidence=0.2,
            scene=scene,
            error="physics_scene_solve_failed",
        )

    magnitude = math.hypot(fx, fy)
    geometry_name = _geometry_label(scene)
    if scene.target and scene.target.quantity == "force_direction":
        direction = _direction_answer(scene, fx, fy)
        cot = _build_scene_cot(
            scene,
            contributions,
            "F",
            fx,
            fy,
            f"Net force direction: {direction}",
        )
        return SceneSolveResult(
            success=True,
            answer=direction,
            explanation=f"Built a verified {geometry_name} scene, summed Coulomb force vectors, and the net vector points {direction}.",
            formula_id="coulomb_vector_scene_direction",
            variables=variables,
            cot=cot,
            confidence=0.88,
            scene=scene,
            contributions=contributions,
            resultant={"Fx": fx, "Fy": fy, "magnitude": magnitude},
        )

    answer = format_best_unit(magnitude, "N")
    # Result-level dimensional/unit validation (Req 5.7): a force magnitude must
    # be dimensionally a force (N), else abstain rather than return it.
    dimension_errors = _result_dimension_errors(
        scene.target.quantity if scene.target else "force_magnitude", "N"
    )
    if dimension_errors:
        return _scene_abstention(scene, dimension_errors, "physics_scene_dimensional_validation_failed")
    target_label = scene.target.charge_label if scene.target else "the target"
    explanation = (
        f"Built a verified {geometry_name} scene, assigned coordinates to the charges, "
        f"summed Coulomb force vectors on {target_label}, and computed {answer}."
    )
    cot = _build_scene_cot(
        scene,
        contributions,
        "F",
        fx,
        fy,
        f"Resultant magnitude on {target_label}: {answer}",
    )
    fid = "coulomb_vector_scene"
    if "midpoint" in scene.question.lower() and scene.geometry == "collinear" and len(scene.charges) >= 3:
        fid = "force_two_charge_midpoint"
    return SceneSolveResult(
        success=True,
        answer=answer,
        explanation=explanation,
        formula_id=fid,
        variables=variables,
        cot=cot,
        confidence=0.9,
        scene=scene,
        contributions=contributions,
        resultant={"Fx": fx, "Fy": fy, "magnitude": magnitude},
    )


def _direction_answer(scene: PhysicsScene, fx: float, fy: float) -> str:
    assert scene.target is not None and scene.target.charge_label is not None
    target = scene.charges[scene.target.charge_label]
    assert target.point is not None
    tx, ty = scene.points[target.point]
    best_label = None
    best_cos = -2.0
    norm = math.hypot(fx, fy)
    if norm == 0:
        return "zero net force"
    for label, charge in scene.charges.items():
        if label == target.label or charge.point is None:
            continue
        sx, sy = scene.points[charge.point]
        vx, vy = sx - tx, sy - ty
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            continue
        cos = (fx * vx + fy * vy) / (norm * vnorm)
        if cos > best_cos:
            best_cos = cos
            best_label = label
    if best_label:
        return f"toward {best_label}"
    return "along the net force vector"


def _solve_symbolic_scene(question: str) -> SceneSolveResult | None:
    text = _low(question)
    if "force" not in text:
        return None

    if (
        "midpoint" in text
        and "equal magnitude" in text
        and "same sign" in text
        and "q1" in text
        and "q2" in text
        and ("q3" in text or "third point charge" in text)
    ):
        scene = PhysicsScene(
            question=question,
            geometry="symbolic_midpoint_symmetry",
            target=SceneTarget("force_magnitude", "q3"),
            evidence=["equal same-sign endpoint charges", "target at midpoint"],
        )
        return SceneSolveResult(
            success=True,
            answer="0 N",
            explanation=(
                "The endpoint charges have equal magnitude and the same sign, and q3 is at the midpoint. "
                "The two Coulomb forces on q3 have equal magnitude and opposite directions, so the net force is zero."
            ),
            formula_id="coulomb_vector_symmetry_zero",
            variables={},
            cot=["Recognized midpoint symmetry", "Equal opposite force vectors cancel"],
            confidence=0.92,
            scene=scene,
        )

    if (
        "f0" in text
        and ("isosceles right triangle" in text or "right triangle" in text)
        and "adjacent vertices" in text
        and "remaining vertex" in text
        and ("same magnitude q" in text or "charges of the same magnitude" in text)
    ):
        scene = PhysicsScene(
            question=question,
            geometry="symbolic_isosceles_right_triangle",
            target=SceneTarget("force_magnitude", "q0"),
            evidence=["two perpendicular equal force components", "F0 given as single interaction magnitude"],
        )
        return SceneSolveResult(
            success=True,
            answer="sqrt(2) × F0",
            explanation=(
                "Each adjacent charge exerts a Coulomb force of magnitude F0 on the test charge at the remaining vertex. "
                "Using the perpendicular-vector resultant formula F = sqrt(F0^2 + F0^2), the magnitude is sqrt(2)*F0."
            ),
            formula_id="coulomb_vector_symbolic_right_triangle",
            variables={},
            cot=["Recognized two perpendicular equal components", "Resultant = sqrt(F0^2 + F0^2)"],
            confidence=0.88,
            scene=scene,
        )

    if (
        "equilateral triangle" in text
        and "center" in text
        and ("three identical charges" in text or ("three" in text and "identical charges" in text))
        and "test charge" in text
    ):
        scene = PhysicsScene(
            question=question,
            geometry="symbolic_equilateral_center_symmetry",
            target=SceneTarget("force_magnitude", "q0"),
            evidence=["three identical vertex charges", "test charge at center"],
        )
        return SceneSolveResult(
            success=True,
            answer="0 N",
            explanation=(
                "The three identical charges sit symmetrically at the vertices of an equilateral triangle. "
                "At the center, the three Coulomb force vectors have equal magnitudes and are separated by 120 degrees, so they cancel."
            ),
            formula_id="coulomb_vector_center_symmetry_zero",
            variables={},
            cot=["Recognized equilateral-center symmetry", "Three equal 120-degree force vectors cancel"],
            confidence=0.92,
            scene=scene,
        )

    return None


def label_aware_capacitor_energy(question: str) -> SceneSolveResult | None:
    """Parses and solves capacitor energy questions directly from C and U values.

    Args:
        question: The text of the capacitor energy question.

    Returns:
        A SceneSolveResult if values were parsed and solved, otherwise None.
    """
    text = _low(question)
    if "capacitor" not in text or "energy" not in text:
        return None

    c_match = re.search(rf"\bc\s*=\s*({NUMBER_PATTERN})\s*(μf|uf|microfarads?|mf|f)\b", text, re.I)
    v_match = re.search(rf"\b(?:u|v)\s*=\s*({NUMBER_PATTERN})\s*(v|volt|volts)\b", text, re.I)
    if not c_match or not v_match:
        return None
    c_qty = extract_quantities(f"{c_match.group(1)} {c_match.group(2)}")
    v_qty = extract_quantities(f"{v_match.group(1)} {v_match.group(2)}")
    if not c_qty or not v_qty:
        return None
    capacitance = c_qty[0].si_value
    voltage = v_qty[0].si_value
    energy = 0.5 * capacitance * voltage * voltage
    scene = PhysicsScene(question=question, geometry="capacitor_energy", target=SceneTarget("energy"))
    return SceneSolveResult(
        success=True,
        answer=format_best_unit(energy, "J"),
        explanation=f"Parsed label-aware capacitance C={capacitance:.6g} F and voltage U={voltage:.6g} V, then computed E = 0.5*C*V^2.",
        formula_id="capacitor_energy_e_half_cv2",
        variables={"C": capacitance, "V": voltage},
        cot=["Parsed label-aware capacitor values", f"Computed E={energy:.6g} J"],
        confidence=0.95,
        scene=scene,
    )
