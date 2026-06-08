"""Registry of physics domain adapters.

This module registers and exposes all available domain-specific physics
adapters in their preferred order of execution.
"""

from __future__ import annotations

from app.physics.adapters.base import PhysicsAdapter
from app.physics.adapters.circuit import CircuitAdapter
from app.physics.adapters.electrostatics import ElectrostaticsVectorAdapter
from app.physics.adapters.mechanics import MechanicsAdapter
from app.physics.adapters.measurement import MeasurementAdapter
from app.physics.adapters.optics import OpticsAdapter
from app.physics.adapters.fluids import FluidsAdapter
from app.physics.adapters.thermal import ThermalAdapter


def default_adapters() -> tuple[PhysicsAdapter, ...]:
    """Returns the registered physics adapters in prioritized order.

    Returns:
        A tuple of instantiated PhysicsAdapter protocols.
    """
    # Order matters for overlap: measurement/circuit targets are more specific
    # than broad mechanics/electrostatics keyword matches. Optics, fluids, and
    # thermal are specific domains gated by their own structural detectors,
    # placed before mechanics so a domain-specific question is not captured by a
    # broad mechanics keyword.
    return (
        MeasurementAdapter(),
        CircuitAdapter(),
        ElectrostaticsVectorAdapter(),
        OpticsAdapter(),
        FluidsAdapter(),
        ThermalAdapter(),
        MechanicsAdapter(),
    )

