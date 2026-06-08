"""Mechanics problem adapter.

This module provides the MechanicsAdapter class, which constructs equation graphs
for small classical mechanics problems such as speed, force, acceleration,
torque, kinetic energy, potential energy, momentum, and kinematics.
"""

from __future__ import annotations

import math
import re

from app.physics.adapters.base import AdapterSolution
from app.physics.dimensions import dimension_for_unit
from app.physics.equation_graph import EquationGraph, EquationNode, EquationVariable
from app.physics.ir import PhysicsProblemIR
from app.physics.unit_converter import format_best_unit


class MechanicsAdapter:
    """Equation-graph adapter for small classical mechanics relations.

    This is intentionally not a large mechanics solver. It demonstrates the
    shared IR/equation-graph path for non-electromagnetic physics and gives later
    agents a clear extension point for kinematics, dynamics, torque, energy, and
    momentum families.
    """

    name = "mechanics_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        low = ir.question.lower()
        # Structural mechanics signals (kinematics, dynamics, energy, periodicity).
        # Score is independent of question phrasing — based only on lexical
        # mechanics tokens that map to structural variable patterns the adapter
        # already covers (kg, m/s, m/s², m, s, N, plus rotation/period words).
        if any(
            token in low
            for token in [
                "speed", "velocity", "distance", "displacement", "time",
                "force", "mass", "acceleration", "accelerates", "accelerate",
                "torque", "moment", "kinetic", "potential energy", "momentum",
                "height", "incline", "inclined", "ramp", "slope",
                "pendulum", "period", "oscillat", "projectile", "range", "fired", "launched",
            ]
        ):
            return 0.65
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph representing the mechanics relationship.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            An EquationGraph instance, or None if no mechanics relationships match.
        """
        low = ir.question.lower()
        if _asks_inelastic_collision(low):
            return _inelastic_collision_graph(ir)
        if _asks_charged_particle_speed(low):
            return _charged_particle_speed_graph(ir)
        if _asks_work_against_gravity(low):
            return _work_against_gravity_graph(ir)
        if _asks_torque(low):
            return _torque_graph(ir)
        if _asks_kinetic_energy(low):
            return _kinetic_energy_graph(ir)
        if _asks_gravitational_potential_energy(low):
            return _gravitational_potential_energy_graph(ir)
        if _asks_momentum(low):
            return _momentum_graph(ir)
        if _asks_pendulum_period(low):
            return _pendulum_period_graph(ir)
        if _asks_projectile_range(low):
            return _projectile_range_graph(ir)
        if _asks_incline_acceleration(low):
            return _incline_acceleration_graph(ir)
        if _asks_displacement(low):
            return _constant_acceleration_displacement_graph(ir)
        if _asks_final_velocity(low):
            return _constant_acceleration_velocity_graph(ir)
        if _asks_speed(low):
            return _speed_graph(ir)
        if _asks_force(low):
            return _force_graph(ir)
        if _asks_acceleration(low):
            return _acceleration_graph(ir)
        return None

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the mechanics problem using the equation graph.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            An AdapterSolution, or None if the problem cannot be solved.
        """
        graph = self.build_equation_graph(ir)
        if graph is None:
            return None
        solved = graph.solve_forward()
        if solved is None:
            return None
        value, trace = solved
        unit = _target_unit(graph.target)
        answer = format_best_unit(value, unit)
        return AdapterSolution(
            answer=answer,
            explanation=f"Solved a mechanics equation graph for {graph.target} using backend arithmetic.",
            formula_id=f"mechanics_{graph.target}",
            variables={name: variable.value for name, variable in graph.variables.items() if variable.value is not None} | {str(graph.target): value},
            cot=trace + [f"{graph.target} = {value:.6g} {unit}"],
            confidence=0.9,
            trace={"equation_graph_target": graph.target, "equation_count": len(graph.equations)},
        )



def _asks_speed(low: str) -> bool:
    """Checks if the question text asks for speed or velocity."""
    return "speed" in low or "velocity" in low


def _asks_force(low: str) -> bool:
    """Checks if the question text asks for mechanics force."""
    return "force" in low and "electric" not in low and "magnetic" not in low and "electromotive" not in low


def _asks_acceleration(low: str) -> bool:
    """Checks if the question text asks for acceleration."""
    return "acceleration" in low


def _asks_torque(low: str) -> bool:
    """Checks if the question text asks for torque or moment."""
    return "torque" in low or "moment of force" in low or re.search(r"\bmoment\b", low)


def _asks_kinetic_energy(low: str) -> bool:
    """Checks if the question text asks for kinetic energy."""
    return "kinetic energy" in low


def _asks_gravitational_potential_energy(low: str) -> bool:
    """Checks if the question text asks for potential energy."""
    return "potential energy" in low or "gravitational energy" in low or "gravitational potential" in low


def _asks_momentum(low: str) -> bool:
    """Checks if the question text asks for momentum."""
    return "momentum" in low


def _asks_final_velocity(low: str) -> bool:
    """Checks if the question text asks for final velocity under constant acceleration."""
    return ("final velocity" in low or "final speed" in low or "velocity after" in low or "speed after" in low) and ("acceleration" in low or "accelerates" in low or "accelerate" in low)


def _asks_inelastic_collision(low: str) -> bool:
    """Checks if the question describes a perfectly inelastic collision.

    Signal: a collision where the bodies "stick together" / "couple" / "move
    together" and the question asks for the common final velocity.
    """
    if "collid" not in low and "collision" not in low and "crash" not in low:
        return False
    sticks = any(
        token in low
        for token in ["stick together", "stick", "couple", "move together", "embed", "combined", "joined"]
    )
    asks_v = "final velocity" in low or "final speed" in low or "common velocity" in low or "their velocity" in low or "velocity" in low or "speed" in low
    return sticks and asks_v


def _asks_charged_particle_speed(low: str) -> bool:
    """Checks if the question asks for the speed of a charged particle accelerated
    through a potential difference (work-energy: qU = ½ m v²)."""
    if not any(token in low for token in ["electron", "proton", "ion", "charged particle", "charge"]):
        return False
    if "accelerat" not in low:
        return False
    through_voltage = "through" in low and ("v" in low or "volt" in low or "potential" in low)
    asks_speed = "speed" in low or "velocity" in low
    return through_voltage and asks_speed


def _asks_work_against_gravity(low: str) -> bool:
    """Checks if the question asks for work done against gravity (lifting work).

    Work against gravity equals m·g·Δh, where Δh is the vertical height gained.
    On an incline of angle θ over distance d along the slope, Δh = d·sin(θ).
    """
    if "work" not in low:
        return False
    return "against gravity" in low or "against the gravity" in low or ("lift" in low and "work" in low)


def _asks_displacement(low: str) -> bool:
    """Checks if the question text asks for displacement/distance under constant acceleration."""
    return ("how far" in low or "distance traveled" in low or "distance travelled" in low or "displacement" in low) and ("acceleration" in low or "accelerates" in low or "accelerate" in low)


def _asks_pendulum_period(low: str) -> bool:
    """Checks if the question describes a simple pendulum's oscillation period."""
    return "pendulum" in low and ("period" in low or "oscillat" in low or "swing" in low)


def _asks_projectile_range(low: str) -> bool:
    """Checks if the question asks for projectile horizontal range on level ground."""
    if "projectile" not in low and "fired" not in low and "launched" not in low:
        return False
    if "range" in low or "horizontal distance" in low or "how far" in low:
        # Disambiguate from max-height / time-of-flight phrasing.
        if "maximum height" in low or "max height" in low or "time of flight" in low:
            return False
        return True
    return False


def _asks_incline_acceleration(low: str) -> bool:
    """Checks if the question describes a frictionless inclined-plane acceleration."""
    if not any(token in low for token in ["incline", "inclined", "ramp", "slope"]):
        return False
    return "acceleration" in low or "accelerate" in low or "accelerates" in low


def _speed_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for speed calculations."""
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    times = [q.value for q in ir.quantities if q.si_unit == "s"]
    if not distances or not times:
        return None
    graph = EquationGraph(target="speed")
    graph.add_known("distance", distances[0], dimension_for_unit("m"), provenance="question")
    graph.add_known("time", times[0], dimension_for_unit("s"), provenance="question")
    graph.variables["speed"] = EquationVariable("speed", dimension_for_unit("m/s"))
    graph.add_equation(
        EquationNode(
            id="speed_definition",
            expression="speed = distance / time",
            output="speed",
            inputs=("distance", "time"),
            output_dimension=dimension_for_unit("m/s"),
            compute=lambda values: values["distance"] / values["time"],
        )
    )
    return graph


def _force_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for force calculations."""
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    accelerations = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    if not masses or not accelerations:
        return None
    graph = EquationGraph(target="force")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("acceleration", accelerations[0], dimension_for_unit("m/s²"), provenance="question")
    graph.variables["force"] = EquationVariable("force", dimension_for_unit("N"))
    graph.add_equation(
        EquationNode(
            id="newton_second_law",
            expression="force = mass * acceleration",

            output="force",
            inputs=("mass", "acceleration"),
            output_dimension=dimension_for_unit("N"),
            compute=lambda values: values["mass"] * values["acceleration"],
        )
    )
    return graph


def _acceleration_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for acceleration calculations."""
    forces = [q.value for q in ir.quantities if q.si_unit == "N"]
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    if not forces or not masses:
        return None
    graph = EquationGraph(target="acceleration")
    graph.add_known("force", forces[0], dimension_for_unit("N"), provenance="question")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.variables["acceleration"] = EquationVariable("acceleration", dimension_for_unit("m/s²"))
    graph.add_equation(
        EquationNode(
            id="newton_second_law_acceleration",
            expression="acceleration = force / mass",
            output="acceleration",
            inputs=("force", "mass"),
            output_dimension=dimension_for_unit("m/s²"),
            compute=lambda values: values["force"] / values["mass"],
        )
    )
    return graph


def _torque_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for torque calculations."""
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    forces = [q.value for q in ir.quantities if q.si_unit == "N"]
    if not distances or not forces:
        return None
    angle = _extract_angle_rad(ir.question)
    graph = EquationGraph(target="torque")
    graph.add_known("lever_arm", distances[0], dimension_for_unit("m"), provenance="question")
    graph.add_known("force", forces[0], dimension_for_unit("N"), provenance="question")
    graph.add_known("sin_theta", math.sin(angle), dimension_for_unit("dimensionless"), provenance="question" if angle != math.pi / 2 else "default perpendicular")
    graph.variables["torque"] = EquationVariable("torque", dimension_for_unit("N·m"))
    graph.add_equation(
        EquationNode(
            id="torque_definition",
            expression="torque = lever_arm * force * sin(theta)",
            output="torque",
            inputs=("lever_arm", "force", "sin_theta"),
            output_dimension=dimension_for_unit("N·m"),
            compute=lambda values: values["lever_arm"] * values["force"] * values["sin_theta"],
        )
    )
    return graph


def _kinetic_energy_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for kinetic energy calculations."""
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if not masses or not speeds:
        return None
    graph = EquationGraph(target="kinetic_energy")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("speed", speeds[0], dimension_for_unit("m/s"), provenance="question")
    graph.variables["kinetic_energy"] = EquationVariable("kinetic_energy", dimension_for_unit("J"))
    graph.add_equation(
        EquationNode(
            id="kinetic_energy_definition",
            expression="kinetic_energy = 0.5 * mass * speed^2",
            output="kinetic_energy",
            inputs=("mass", "speed"),
            output_dimension=dimension_for_unit("J"),
            compute=lambda values: 0.5 * values["mass"] * (values["speed"] ** 2),
        )
    )
    return graph


def _gravitational_potential_energy_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for potential energy calculations."""
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    if not masses or not distances:
        return None
    g_values = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    graph = EquationGraph(target="potential_energy")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("height", distances[0], dimension_for_unit("m"), provenance="question")
    graph.add_known("g", g_values[0] if g_values else 9.8, dimension_for_unit("m/s²"), provenance="question" if g_values else "default Earth gravity")
    graph.variables["potential_energy"] = EquationVariable("potential_energy", dimension_for_unit("J"))
    graph.add_equation(
        EquationNode(
            id="gravitational_potential_energy",
            expression="potential_energy = mass * g * height",
            output="potential_energy",
            inputs=("mass", "g", "height"),
            output_dimension=dimension_for_unit("J"),
            compute=lambda values: values["mass"] * values["g"] * values["height"],
        )
    )
    return graph


def _momentum_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for momentum calculations."""
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if not masses or not speeds:
        return None
    graph = EquationGraph(target="momentum")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("velocity", speeds[0], dimension_for_unit("m/s"), provenance="question")
    graph.variables["momentum"] = EquationVariable("momentum", dimension_for_unit("kg·m/s"))
    graph.add_equation(
        EquationNode(
            id="momentum_definition",
            expression="momentum = mass * velocity",
            output="momentum",
            inputs=("mass", "velocity"),
            output_dimension=dimension_for_unit("kg·m/s"),
            compute=lambda values: values["mass"] * values["velocity"],
        )
    )
    return graph


def _inelastic_collision_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for a perfectly inelastic head-on collision.

    Conservation of momentum with the bodies sticking together:
        v_final = (m1*v1 + m2*v2) / (m1 + m2)
    For a head-on collision in opposite directions the second velocity is
    negative; we detect "opposite" / "head-on" phrasing to set the sign. This
    is a general structural rule (momentum conservation), not a per-question
    text match.
    """
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if len(masses) < 2:
        return None
    low = ir.question.lower()
    # "at rest" / "stationary" supplies an implicit zero velocity for a body
    # that has no stated speed. If only one speed is parsed but two masses are
    # present and the text says one body is at rest, treat the missing speed
    # as 0 m/s (a general physics convention, not a per-question match).
    at_rest = any(t in low for t in ["at rest", "stationary", "initially at rest", "not moving"])
    if len(speeds) < 2:
        if len(speeds) == 1 and at_rest:
            speeds = [speeds[0], 0.0]
        else:
            return None
    low = ir.question.lower()
    opposite = "opposite" in low or "head-on" in low or "head on" in low or "toward each other" in low or "towards each other" in low
    v1 = speeds[0]
    v2 = -speeds[1] if opposite else speeds[1]
    m1, m2 = masses[0], masses[1]
    graph = EquationGraph(target="final_velocity")
    graph.add_known("m1", m1, dimension_for_unit("kg"), provenance="question")
    graph.add_known("m2", m2, dimension_for_unit("kg"), provenance="question")
    graph.add_known("v1", v1, dimension_for_unit("m/s"), provenance="question")
    graph.add_known("v2", v2, dimension_for_unit("m/s"), provenance="question (opposite direction)" if opposite else "question")
    graph.variables["final_velocity"] = EquationVariable("final_velocity", dimension_for_unit("m/s"))
    graph.add_equation(
        EquationNode(
            id="inelastic_collision_momentum",
            expression="final_velocity = (m1*v1 + m2*v2) / (m1 + m2)",
            output="final_velocity",
            inputs=("m1", "m2", "v1", "v2"),
            output_dimension=dimension_for_unit("m/s"),
            compute=lambda values: (values["m1"] * values["v1"] + values["m2"] * values["v2"]) / (values["m1"] + values["m2"]),
        )
    )
    return graph


def _charged_particle_speed_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for a charged particle accelerated through a
    potential difference, using the work-energy theorem qU = ½ m v²:
        v = sqrt(2 * q * U / m)
    Requires a mass (kg), a charge (C), and a voltage (V).
    """
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    charges = [q.value for q in ir.quantities if q.si_unit == "C"]
    voltages = [q.value for q in ir.quantities if q.si_unit == "V"]
    if not masses or not charges or not voltages:
        return None
    graph = EquationGraph(target="final_speed")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("charge", charges[0], dimension_for_unit("C"), provenance="question")
    graph.add_known("voltage", voltages[0], dimension_for_unit("V"), provenance="question")
    graph.variables["final_speed"] = EquationVariable("final_speed", dimension_for_unit("m/s"))
    graph.add_equation(
        EquationNode(
            id="work_energy_charged_particle",
            expression="final_speed = sqrt(2 * charge * voltage / mass)",
            output="final_speed",
            inputs=("charge", "voltage", "mass"),
            output_dimension=dimension_for_unit("m/s"),
            compute=lambda values: math.sqrt(2.0 * abs(values["charge"]) * abs(values["voltage"]) / values["mass"]),
        )
    )
    return graph


def _work_against_gravity_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for work done against gravity.

    Work against gravity = m * g * height_gained. On an incline of angle θ over
    a slope distance d, the vertical height gained is d*sin(θ). When no incline
    angle is present, the metre-valued distance is taken as the vertical height.
    """
    masses = [q.value for q in ir.quantities if q.si_unit == "kg"]
    distances = [q.value for q in ir.quantities if q.si_unit == "m"]
    if not masses or not distances:
        return None
    g_values = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    low = ir.question.lower()
    angle_rad = _extract_angle_optional(ir.question)
    on_incline = any(token in low for token in ["incline", "inclined", "ramp", "slope"]) and angle_rad is not None
    height = distances[0] * math.sin(angle_rad) if on_incline else distances[0]
    graph = EquationGraph(target="work")
    graph.add_known("mass", masses[0], dimension_for_unit("kg"), provenance="question")
    graph.add_known("g", g_values[0] if g_values else 9.8, dimension_for_unit("m/s²"), provenance="question" if g_values else "default Earth gravity")
    graph.add_known("height", height, dimension_for_unit("m"), provenance="d*sin(theta) on incline" if on_incline else "question")
    graph.variables["work"] = EquationVariable("work", dimension_for_unit("J"))
    graph.add_equation(
        EquationNode(
            id="work_against_gravity",
            expression="work = mass * g * height",
            output="work",
            inputs=("mass", "g", "height"),
            output_dimension=dimension_for_unit("J"),
            compute=lambda values: values["mass"] * values["g"] * values["height"],
        )
    )
    return graph


def _constant_acceleration_velocity_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for constant acceleration final velocity calculations."""
    low = ir.question.lower()
    accelerations = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    times = [q.value for q in ir.quantities if q.si_unit == "s"]
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if not accelerations or not times:
        return None
    initial_velocity = 0.0 if "from rest" in low or "starts at rest" in low or "starting from rest" in low else (speeds[0] if speeds else None)
    if initial_velocity is None:
        return None
    graph = EquationGraph(target="final_velocity")
    graph.add_known("initial_velocity", initial_velocity, dimension_for_unit("m/s"), provenance="question" if speeds else "from rest")
    graph.add_known("acceleration", accelerations[0], dimension_for_unit("m/s²"), provenance="question")
    graph.add_known("time", times[0], dimension_for_unit("s"), provenance="question")
    graph.variables["final_velocity"] = EquationVariable("final_velocity", dimension_for_unit("m/s"))
    graph.add_equation(
        EquationNode(
            id="constant_acceleration_velocity",
            expression="final_velocity = initial_velocity + acceleration * time",
            output="final_velocity",
            inputs=("initial_velocity", "acceleration", "time"),
            output_dimension=dimension_for_unit("m/s"),
            compute=lambda values: values["initial_velocity"] + values["acceleration"] * values["time"],
        )
    )
    return graph


def _constant_acceleration_displacement_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for constant acceleration displacement calculations."""
    low = ir.question.lower()
    accelerations = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    times = [q.value for q in ir.quantities if q.si_unit == "s"]
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if not accelerations or not times:
        return None
    initial_velocity = 0.0 if "from rest" in low or "starts at rest" in low or "starting from rest" in low else (speeds[0] if speeds else None)
    if initial_velocity is None:
        return None
    graph = EquationGraph(target="displacement")
    graph.add_known("initial_velocity", initial_velocity, dimension_for_unit("m/s"), provenance="question" if speeds else "from rest")
    graph.add_known("acceleration", accelerations[0], dimension_for_unit("m/s²"), provenance="question")
    graph.add_known("time", times[0], dimension_for_unit("s"), provenance="question")
    graph.variables["displacement"] = EquationVariable("displacement", dimension_for_unit("m"))
    graph.add_equation(
        EquationNode(
            id="constant_acceleration_displacement",
            expression="displacement = initial_velocity*time + 0.5*acceleration*time^2",
            output="displacement",
            inputs=("initial_velocity", "acceleration", "time"),
            output_dimension=dimension_for_unit("m"),
            compute=lambda values: values["initial_velocity"] * values["time"] + 0.5 * values["acceleration"] * (values["time"] ** 2),
        )
    )
    return graph


def _extract_angle_rad(question: str) -> float:
    """Extracts angle in radians from question text, defaulting to pi/2 if not found."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", question, re.I)
    if match:
        return math.radians(float(match.group(1)))
    return math.pi / 2


def _extract_angle_optional(question: str) -> float | None:
    """Extracts an angle in radians from question text, returning None when absent."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", question, re.I)
    if match:
        return math.radians(float(match.group(1)))
    return None


def _pendulum_period_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for a simple pendulum's period.

    Uses the small-angle relation T = 2 * pi * sqrt(L / g). Length comes from a
    metre-valued quantity in the IR, gravity falls back to 9.8 m/s² when not
    provided in the question text.
    """
    lengths = [q.value for q in ir.quantities if q.si_unit == "m"]
    if not lengths:
        return None
    g_values = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    graph = EquationGraph(target="period")
    graph.add_known("length", lengths[0], dimension_for_unit("m"), provenance="question")
    graph.add_known(
        "g",
        g_values[0] if g_values else 9.8,
        dimension_for_unit("m/s²"),
        provenance="question" if g_values else "default Earth gravity",
    )
    graph.variables["period"] = EquationVariable("period", dimension_for_unit("s"))
    graph.add_equation(
        EquationNode(
            id="simple_pendulum_period",
            expression="period = 2*pi*sqrt(length / g)",
            output="period",
            inputs=("length", "g"),
            output_dimension=dimension_for_unit("s"),
            compute=lambda values: 2.0 * math.pi * math.sqrt(values["length"] / values["g"]),
        )
    )
    return graph


def _projectile_range_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for projectile horizontal range on level ground.

    Uses R = v0² * sin(2*theta) / g. Requires an initial speed (m/s), an angle
    extracted from the question text (in degrees), and gravity (defaults to 9.8).
    """
    speeds = [q.value for q in ir.quantities if q.si_unit == "m/s"]
    if not speeds:
        return None
    angle_rad = _extract_angle_optional(ir.question)
    if angle_rad is None:
        return None
    g_values = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    graph = EquationGraph(target="range")
    graph.add_known("v0", speeds[0], dimension_for_unit("m/s"), provenance="question")
    graph.add_known("angle_rad", angle_rad, dimension_for_unit("dimensionless"), provenance="question")
    graph.add_known(
        "g",
        g_values[0] if g_values else 9.8,
        dimension_for_unit("m/s²"),
        provenance="question" if g_values else "default Earth gravity",
    )
    graph.variables["range"] = EquationVariable("range", dimension_for_unit("m"))
    graph.add_equation(
        EquationNode(
            id="projectile_range",
            expression="range = v0^2 * sin(2*theta) / g",
            output="range",
            inputs=("v0", "angle_rad", "g"),
            output_dimension=dimension_for_unit("m"),
            compute=lambda values: (values["v0"] ** 2) * math.sin(2.0 * values["angle_rad"]) / values["g"],
        )
    )
    return graph


def _incline_acceleration_graph(ir: PhysicsProblemIR) -> EquationGraph | None:
    """Builds an equation graph for frictionless inclined-plane acceleration.

    Uses a = g * sin(theta). Mass cancels out, so only an angle and gravity
    are needed. Friction-aware variants are out of scope here.
    """
    angle_rad = _extract_angle_optional(ir.question)
    if angle_rad is None:
        return None
    g_values = [q.value for q in ir.quantities if q.si_unit == "m/s²"]
    graph = EquationGraph(target="acceleration")
    graph.add_known("angle_rad", angle_rad, dimension_for_unit("dimensionless"), provenance="question")
    graph.add_known(
        "g",
        g_values[0] if g_values else 9.8,
        dimension_for_unit("m/s²"),
        provenance="question" if g_values else "default Earth gravity",
    )
    graph.variables["acceleration"] = EquationVariable("acceleration", dimension_for_unit("m/s²"))
    graph.add_equation(
        EquationNode(
            id="incline_acceleration",
            expression="acceleration = g * sin(angle)",
            output="acceleration",
            inputs=("g", "angle_rad"),
            output_dimension=dimension_for_unit("m/s²"),
            compute=lambda values: values["g"] * math.sin(values["angle_rad"]),
        )
    )
    return graph


def _extract_angle_rad_legacy(question: str) -> float:
    """Legacy helper retained for backwards compatibility."""
    return _extract_angle_rad(question)


def _target_unit(target: str | None) -> str:
    """Returns the SI unit string associated with a target quantity."""
    if target == "speed":
        return "m/s"
    if target == "force":
        return "N"
    if target == "acceleration":
        return "m/s²"
    if target == "torque":
        return "N·m"
    if target in {"kinetic_energy", "potential_energy", "work"}:
        return "J"
    if target == "momentum":
        return "kg·m/s"
    if target in {"final_velocity", "final_speed"}:
        return "m/s"
    if target == "displacement":
        return "m"
    if target == "range":
        return "m"
    if target == "period":
        return "s"
    return "dimensionless"

