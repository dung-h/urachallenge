"""Intermediate Representation (IR) for physics problems.

This module defines the dataclasses used to represent physical quantities,
entities, relations, targets, and the parsed state of a physics problem.
It acts as the structured communication interface between physics parsers
and solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.physics.dimensions import Dimension, dimension_for_unit
from app.physics.unit_converter import Quantity, extract_quantities

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle at runtime
    from app.physics.scene_parser import PhysicsScene


@dataclass(frozen=True)
class PhysicalQuantity:
    """Represents a physical quantity parsed from the question.

    Attributes:
        name: A unique identifier for the quantity (e.g., 'q1').
        value: The numeric value in SI base units.
        si_unit: The SI base unit name.
        dimension: The dimensional analysis vector for this quantity.
        raw: The raw string representation from the question text.
        entity: The entity associated with this quantity, if any.
        role: The role/context of the quantity (e.g., 'radius').
    """
    name: str
    value: float
    si_unit: str
    dimension: Dimension | None
    raw: str
    entity: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class PhysicalEntity:
    """Represents a physical entity in the problem scene.

    Attributes:
        id: Unique identifier for the entity.
        kind: The category of the entity (e.g., 'charge', 'resistor').
        label: Optional display label or name.
        attributes: Additional parsed attributes of the entity.
    """
    id: str
    kind: str
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalRelation:
    """Represents a relationship between physical entities or quantities.

    Attributes:
        kind: The type of relation (e.g., 'connected', 'distance').
        args: Identifiers of the related entities or quantities.
        attributes: Properties of the relationship.
        evidence: Text segment providing evidence for this relation.
    """
    kind: str
    args: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: str | None = None


@dataclass(frozen=True)
class TargetSpec:
    """Specifies the target variable or quantity to solve for.

    Attributes:
        quantity: The name of the target physical quantity (e.g., 'voltage').
        unit_hint: Optional hint for the expected return unit.
        entity: The entity associated with the target quantity.
        multi: Whether multiple values are expected.
    """
    quantity: str | None
    unit_hint: str | None = None
    entity: str | None = None
    multi: bool = False


@dataclass
class PhysicsProblemIR:
    """The intermediate representation containing the parsed state of a physics problem.

    Attributes:
        question: The original question text.
        quantities: List of quantities parsed from the question.
        entities: List of entities identified in the problem.
        relations: List of relationships between entities/quantities.
        target: Specifications of what quantity needs to be solved.
        assumptions: List of implicit assumptions or boundary conditions.
        metadata: Domain-specific metadata.
        scene: Parsed geometric/structural scene representation.
    """
    question: str
    quantities: list[PhysicalQuantity] = field(default_factory=list)
    entities: list[PhysicalEntity] = field(default_factory=list)
    relations: list[PhysicalRelation] = field(default_factory=list)
    target: TargetSpec | None = None
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    scene: "PhysicsScene | None" = None

    @classmethod
    def from_question(cls, question: str, target_quantity: str | None = None) -> PhysicsProblemIR:
        """Helper to build a basic IR directly from a question string.

        Args:
            question: The raw physics problem text.
            target_quantity: The target quantity type to solve for.

        Returns:
            A PhysicsProblemIR populated with extracted quantities.
        """
        quantities = [_physical_quantity_from_extracted(index, quantity) for index, quantity in enumerate(extract_quantities(question), start=1)]
        return cls(
            question=question,
            quantities=quantities,
            target=TargetSpec(quantity=target_quantity),
        )

    def quantities_by_unit(self, si_unit: str) -> list[PhysicalQuantity]:
        """Filters the parsed quantities by their SI base unit.

        Args:
            si_unit: The target SI base unit (e.g., 'V', 'A', 'ohm').

        Returns:
            A list of PhysicalQuantity matching the given SI unit.
        """
        return [quantity for quantity in self.quantities if quantity.si_unit == si_unit]


def _physical_quantity_from_extracted(index: int, quantity: Quantity) -> PhysicalQuantity:
    """Converts a parsed Quantity object to a PhysicalQuantity IR object."""
    return PhysicalQuantity(
        name=f"q{index}",
        value=quantity.si_value,
        si_unit=quantity.si_unit,
        dimension=dimension_for_unit(quantity.si_unit),
        raw=quantity.raw,
    )

