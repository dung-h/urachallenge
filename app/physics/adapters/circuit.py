"""Circuit problem adapter.

This module provides the CircuitAdapter class which handles RLC circuit problems
involving frequency scaling, reactance, resonance, and AC current.
"""

from __future__ import annotations

import re

from app.physics.adapters.base import AdapterSolution
from app.physics.circuit_solver import (
    SeriesRLCPhasorIR,
    series_rlc_current_after_frequency_scale,
    series_rlc_inductive_reactance_from_scaled_current,
)
from app.physics.equation_graph import EquationGraph
from app.physics.ir import PhysicsProblemIR
from app.physics.unit_converter import format_best_unit


class CircuitAdapter:
    """Adapter for circuit-related physics problems.

    This adapter handles queries about RLC circuits, specifically resonance,
    frequency scaling, reactances, and computing the resulting AC current or reactance.
    """
    name = "circuit_adapter"

    def can_handle(self, ir: PhysicsProblemIR) -> float:
        """Determines if the adapter can solve the given problem IR.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        low = ir.question.lower()
        if any(token in low for token in ["rlc", "reactance", "x_l", "xl", "x_c", "xc", "resonance"]):
            return 0.85
        if "parallel" in low and "series" in low and any(token in low for token in ["resistor", "ohm", "Ω"]):
            return 0.90
        return 0.0

    def build_equation_graph(self, ir: PhysicsProblemIR) -> EquationGraph | None:
        """Builds an equation graph for the problem.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            None, as this adapter does not use equation graphs.
        """
        return None

    def _solve_series_parallel_resistors(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        import re
        from app.physics.unit_converter import format_best_unit
        
        # Robust variable extraction (SI base units)
        low = ir.question.lower().replace("Ω", "ohm").replace("ω", "ohm").replace("μ", "u").replace("µ", "u")
        variables = {}
        
        pattern = r"\b(r1|r2|r3|r4|v|i|c1|c2|c3|c|r_1|r_2|r_3|r_total|v_source)\b\s*(?:=|\bis\b|\bof\b|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(k?ohm|k?omega|ohm|v|volt|volts|a|amp|amps|ampere|amperes|ma|ua|uf|f)\b"
        for match in re.finditer(pattern, low):
            label = match.group(1).replace("_", "")
            val_str = match.group(2)
            unit_str = match.group(3)
            try:
                val = float(val_str)
                if unit_str.startswith("k"):
                    val *= 1e3
                elif unit_str.startswith("m") and unit_str != "m":
                    val *= 1e-3
                elif unit_str.startswith("u"):
                    val *= 1e-6
                variables[label] = val
            except ValueError:
                continue
                
        if "v" not in variables:
            v_match = re.search(r"\b(?:source\s+)?v\s*(?:=|\bis\b|\bof\b|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(v|volt|volts)\b", low)
            if v_match:
                variables["v"] = float(v_match.group(1))

        # Check required variables: R1, R2, R3 and V must be present
        if not ("r1" in variables and "r2" in variables and "r3" in variables and "v" in variables):
            return None
            
        r1 = variables["r1"]
        r2 = variables["r2"]
        r3 = variables["r3"]
        v = variables["v"]
        
        # Determine topology: find which pair is in parallel (default r1 and r2)
        parallel_pair = ("r1", "r2")
        series_res = "r3"
        if re.search(r"\b(r2|r3)\b.+\b(r2|r3)\b.+\bparallel\b", low) or "r2 and r3 are in parallel" in low or "r2 and r3 in parallel" in low:
            parallel_pair = ("r2", "r3")
            series_res = "r1"
        elif re.search(r"\b(r1|r3)\b.+\b(r1|r3)\b.+\bparallel\b", low) or "r1 and r3 are in parallel" in low or "r1 and r3 in parallel" in low:
            parallel_pair = ("r1", "r3")
            series_res = "r2"
            
        # Get values for parallel and series parts
        rp1 = variables[parallel_pair[0]]
        rp2 = variables[parallel_pair[1]]
        rs = variables[series_res]
        
        # Computations
        if rp1 == 0 or rp2 == 0:
            r_parallel = 0.0
        else:
            r_parallel = 1.0 / (1.0 / rp1 + 1.0 / rp2)
            
        r_total = r_parallel + rs
        if r_total == 0:
            return None
            
        i_total = v / r_total
        v_rs = i_total * rs
        v_parallel = v - v_rs
        
        i_rp1 = v_parallel / rp1 if rp1 != 0 else 0.0
        i_rp2 = v_parallel / rp2 if rp2 != 0 else 0.0
        
        res_voltages = {
            parallel_pair[0]: v_parallel,
            parallel_pair[1]: v_parallel,
            series_res: v_rs
        }
        res_currents = {
            parallel_pair[0]: i_rp1,
            parallel_pair[1]: i_rp2,
            series_res: i_total
        }
        
        # Check target quantities from query
        asks_v = {}
        asks_i = {}
        for r_name in ["r1", "r2", "r3"]:
            v_patterns = [
                rf"voltage\s+(?:drop\s+)?(?:across|of|at|on)?\s*{r_name}\b",
                rf"potential\s+(?:difference|drop)\s+(?:across|of|at|on)?\s*{r_name}\b",
                rf"voltage\s+(?:drop\s+)?{r_name}\b"
            ]
            if any(re.search(pat, low) for pat in v_patterns):
                asks_v[r_name] = True
            
            i_patterns = [
                rf"current\s+(?:through|in|of)?\s*(?:branch\s+)?{r_name}\b",
                rf"current\s+{r_name}\b"
            ]
            if any(re.search(pat, low) for pat in i_patterns):
                asks_i[r_name] = True
                
        asks_v_parallel = any(phrase in low for phrase in ["voltage across the parallel", "voltage drop across the parallel", "voltage across parallel", "voltage drop across parallel"])
        
        answers = []
        explanation_parts = []
        cot_steps = [
            f"Parsed resistor values: R1 = {r1:.6g} ohm, R2 = {r2:.6g} ohm, R3 = {r3:.6g} ohm",
            f"Source voltage V = {v:.6g} V",
            f"Parallel pair: {parallel_pair[0].upper()} and {parallel_pair[1].upper()} (R_parallel = {r_parallel:.6g} ohm)",
            f"Series resistor: {series_res.upper()}",
            f"Total equivalent resistance R_total = {r_total:.6g} ohm",
            f"Total current I_total = {i_total:.6g} A"
        ]
        
        query_items = []
        for r_name in ["r1", "r2", "r3"]:
            if asks_v.get(r_name):
                idx = low.find(r_name, low.find("voltage") if low.find("voltage") != -1 else 0)
                query_items.append((idx, "V", r_name, res_voltages[r_name], "V"))
            if asks_i.get(r_name):
                idx = low.find(r_name, low.find("current") if low.find("current") != -1 else 0)
                query_items.append((idx, "I", r_name, res_currents[r_name], "A"))
        if asks_v_parallel:
            idx = low.find("parallel")
            query_items.append((idx, "V_parallel", "parallel block", v_parallel, "V"))
            
        query_items.sort(key=lambda x: x[0])
        
        if not query_items:
            if ir.target and ir.target.quantity == "voltage":
                query_items.append((0, "V", "r3", v_rs, "V"))
            elif ir.target and ir.target.quantity == "current":
                query_items.append((0, "I", "r1", i_rp1, "A"))
            else:
                return None
                
        for _, q_type, name, val, unit in query_items:
            answers.append(format_best_unit(val, unit))
            if q_type == "V_parallel":
                explanation_parts.append(f"voltage across parallel block = {format_best_unit(val, unit)}")
                cot_steps.append(f"V_parallel = {val:.6g} V")
            elif q_type == "V":
                explanation_parts.append(f"voltage drop across {name.upper()} = {format_best_unit(val, unit)}")
                cot_steps.append(f"V_{name} = {val:.6g} V")
            else:
                explanation_parts.append(f"current through {name.upper()} = {format_best_unit(val, unit)}")
                cot_steps.append(f"I_{name} = {val:.6g} A")
                
        answer_str = "; ".join(answers)
        explanation_str = "Solved series-parallel resistor network: " + ", ".join(explanation_parts) + "."
        
        solver_vars = {
            "r1": r1, "r2": r2, "r3": r3, "v": v,
            "r_parallel": r_parallel, "r_total": r_total,
            "i_total": i_total, "v_parallel": v_parallel,
            "v_r1": res_voltages["r1"], "v_r2": res_voltages["r2"], "v_r3": res_voltages["r3"],
            "i_r1": res_currents["r1"], "i_r2": res_currents["r2"], "i_r3": res_currents["r3"]
        }
        
        return AdapterSolution(
            answer=answer_str,
            explanation=explanation_str,
            formula_id="series_parallel_resistor_network",
            variables=solver_vars,
            cot=cot_steps,
            confidence=0.95,
            trace={"circuit_ir": "SeriesParallelResistors", "backend": "circuit_solver"},
        )

    def solve(self, ir: PhysicsProblemIR) -> AdapterSolution | None:
        """Solves the circuit problem.

        Args:
            ir: The intermediate representation of the physics problem.

        Returns:
            An AdapterSolution, or None if the problem cannot be solved.
        """
        low = ir.question.lower().replace("ω", "omega")

        if "parallel" in low and "series" in low and any(token in low for token in ["resistor", "ohm", "Ω"]):
            sol = self._solve_series_parallel_resistors(ir)
            if sol is not None:
                return sol

        r = _extract_labeled_scalar(ir.question, ["r", "resistance"], r"ohms?|Ω|ω")
        x_l = _extract_labeled_scalar(ir.question, ["x_l", "xl", "inductive reactance"], r"ohms?|Ω|ω")
        x_c = _extract_labeled_scalar(ir.question, ["x_c", "xc", "capacitive reactance"], r"ohms?|Ω|ω")
        voltage = _extract_labeled_scalar(ir.question, ["u", "v", "voltage"], r"v|volts?")
        scale = _frequency_scale(ir)

        if ir.target and ir.target.quantity == "current" and r is not None and x_l is not None and x_c is not None and voltage is not None and scale is not None:
            result = series_rlc_current_after_frequency_scale(
                SeriesRLCPhasorIR(resistance=r, inductive_reactance=x_l, capacitive_reactance=x_c, voltage=voltage, frequency_scale=scale)
            )
            if result is None:
                return None
            return _adapter_solution(result)

        asks_inductive_reactance = ir.target and ir.target.quantity == "reactance" and any(token in low for token in ["inductive reactance", "z_l", "zl", "x_l", "xl"])
        if asks_inductive_reactance and r is not None and scale is not None:
            currents = [quantity.value for quantity in ir.quantities if quantity.si_unit == "A"]
            if len(currents) >= 2:
                result = series_rlc_inductive_reactance_from_scaled_current(
                    SeriesRLCPhasorIR(resistance=r, frequency_scale=scale, resonance_current=max(currents), scaled_current=min(currents))
                )
                if result is not None:
                    return _adapter_solution(result)
        return None


def _adapter_solution(result) -> AdapterSolution:
    """Wraps a solver result into an AdapterSolution."""
    return AdapterSolution(
        answer=format_best_unit(result.value, result.unit),
        explanation=result.explanation,
        formula_id=result.formula_id,
        variables=result.variables,
        cot=result.cot,
        confidence=0.92,
        trace={"circuit_ir": "SeriesRLCPhasorIR", "backend": "phasor_impedance"},
    )



def _extract_labeled_scalar(question: str, labels: list[str], unit_pattern: str | None = None) -> float | None:
    """Extracts a scalar value associated with one of the target labels from the question."""
    low = question.replace("Ω", "ohm").replace("ω", "ohm").lower()
    for label in labels:
        label_pattern = re.escape(label.lower()).replace("\\ ", r"\s*")
        unit = rf"\s*(?:{unit_pattern})(?=$|\s|[\.,;:\?\)])" if unit_pattern else ""
        match = re.search(rf"\b{label_pattern}\b\s*(?:=|is|of)?\s*([-+]?\d+(?:\.\d+)?){unit}", low, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _frequency_scale(ir: PhysicsProblemIR) -> float | None:
    """Extracts the frequency scaling factor (e.g. doubled -> 2.0) from the problem."""
    low = ir.question.lower().replace("ω", "omega")
    if any(token in low for token in ["doubled", "double", "twice"]):
        return 2.0
    if any(token in low for token in ["tripled", "triple"]):
        return 3.0
    if any(token in low for token in ["halved", "half"]):
        return 0.5
    match = re.search(r"\b(?:frequency|omega).{0,40}?(?:factor\s+of|multiplied\s+by|times)\s*([0-9]+(?:\.[0-9]+)?)", low, re.I)
    if match:
        return float(match.group(1))
    if "resonance" in low:
        frequencies = [quantity.value for quantity in ir.quantities if quantity.si_unit == "Hz"]
        if len(frequencies) >= 2 and min(frequencies) > 0:
            return max(frequencies) / min(frequencies)
    return None
