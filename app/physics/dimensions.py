"""SI Dimensional Analysis vector representation.

This module provides the Dimension class and helper functions to analyze and
validate the dimensional compatibility of physical quantities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """Seven-base SI dimension vector.

    Order: mass, length, time, electric current, temperature, amount, luminous intensity.
    This is intentionally small but cross-domain: mechanics, circuits, thermal,
    waves, and fluids can all share it.
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminous_intensity: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.current + other.current,
            self.temperature + other.temperature,
            self.amount + other.amount,
            self.luminous_intensity + other.luminous_intensity,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.current - other.current,
            self.temperature - other.temperature,
            self.amount - other.amount,
            self.luminous_intensity - other.luminous_intensity,
        )

    def power(self, exponent: int) -> Dimension:
        """Raises the dimension vector components to an integer power.

        Args:
            exponent: The exponent to raise each component to.

        Returns:
            A new Dimension instance representing the power.
        """
        return Dimension(
            self.mass * exponent,
            self.length * exponent,
            self.time * exponent,
            self.current * exponent,
            self.temperature * exponent,
            self.amount * exponent,
            self.luminous_intensity * exponent,
        )


DIMENSIONLESS = Dimension()
MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
CURRENT = Dimension(current=1)
TEMPERATURE = Dimension(temperature=1)

DIMENSIONS_BY_SI_UNIT: dict[str, Dimension] = {
    "": DIMENSIONLESS,
    "dimensionless": DIMENSIONLESS,
    "%": DIMENSIONLESS,
    "kg": MASS,
    "m": LENGTH,
    "s": TIME,
    "A": CURRENT,
    "K": TEMPERATURE,
    "Hz": TIME.power(-1),
    "N": MASS * LENGTH / TIME.power(2),
    "J": MASS * LENGTH.power(2) / TIME.power(2),
    "N·m": MASS * LENGTH.power(2) / TIME.power(2),
    "W": MASS * LENGTH.power(2) / TIME.power(3),
    "C": CURRENT * TIME,
    "V": MASS * LENGTH.power(2) / (TIME.power(3) * CURRENT),
    "ohm": MASS * LENGTH.power(2) / (TIME.power(3) * CURRENT.power(2)),
    "F": TIME.power(4) * CURRENT.power(2) / (MASS * LENGTH.power(2)),
    "H": MASS * LENGTH.power(2) / (TIME.power(2) * CURRENT.power(2)),
    "T": MASS / (TIME.power(2) * CURRENT),
    "N/C": MASS * LENGTH / (TIME.power(3) * CURRENT),
    "m/s": LENGTH / TIME,
    "m/s²": LENGTH / TIME.power(2),
    "kg·m/s": MASS * LENGTH / TIME,
    "m²": LENGTH.power(2),
    "m³": LENGTH.power(3),
    "kg/m³": MASS / LENGTH.power(3),
    "J/(kg·K)": LENGTH.power(2) / (TIME.power(2) * TEMPERATURE),
    "Wb": MASS * LENGTH.power(2) / (TIME.power(2) * CURRENT),
}


def dimension_for_unit(si_unit: str) -> Dimension | None:
    """Retrieves the Dimension vector corresponding to an SI unit string.

    Args:
        si_unit: The target SI unit name.

    Returns:
        The matched Dimension object, or None if not registered.
    """
    return DIMENSIONS_BY_SI_UNIT.get(si_unit)


def dimensions_compatible(left: Dimension | None, right: Dimension | None) -> bool:
    """Determines if two Dimension vectors are identical.

    Args:
        left: The first Dimension to compare.
        right: The second Dimension to compare.

    Returns:
        True if both dimensions are not None and match component-wise.
    """
    return left is not None and right is not None and left == right

