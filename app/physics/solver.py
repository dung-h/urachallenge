from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.guardrails import guardrail_prompt_text
from app.physics.formulas import get_formula
from app.physics.parser import ParsedPhysicsProblem, parse_physics_question
from app.physics.templates import physics_explanation
from app.physics.unit_converter import format_best_unit


@dataclass
class PhysicsSolution:
    success: bool
    answer: str
    explanation: str
    formula_id: str | None
    variables: dict[str, float] = field(default_factory=dict)
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parsed: ParsedPhysicsProblem | None = None
    error: str | None = None
    fallback_used: bool = False
    model_calls: int = 0


def _unsupported_context(question: str) -> str | None:
    low = question.lower()
    if "open switch" in low or "switch is open" in low:
        return "open_switch_context"
    if "ladder network" in low or " ladder " in low:
        return "unsupported_ladder_topology"
    return None


def _compute(parsed: ParsedPhysicsProblem, fallback_used: bool = False, model_calls: int = 0) -> PhysicsSolution:
    if not parsed.formula_id:
        reason = "no deterministic physics formula matched the supplied information."
        if getattr(parsed, "ambiguity", None):
            context = str(parsed.ambiguity[0]).replace("_", " ").strip()
            if context:
                reason = f"{context} means no deterministic physics formula matched the supplied information."
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=f"The answer is unknown because {reason}",
            formula_id=None,
            confidence=0.2,
            parsed=parsed,
            error="formula_not_matched",
            fallback_used=fallback_used,
            model_calls=model_calls,
        )
    try:
        formula = get_formula(parsed.formula_id)
        answer_value = formula.compute(parsed.variables)
    except Exception as exc:
        return PhysicsSolution(
            success=False,
            answer="unknown",
            explanation=f"The answer is unknown because the deterministic solver matched {parsed.formula_id} but computation failed: {exc}.",
            formula_id=parsed.formula_id,
            variables=parsed.variables,
            confidence=0.25,
            parsed=parsed,
            error=str(exc),
            fallback_used=fallback_used,
            model_calls=model_calls,
        )
    answer = format_best_unit(answer_value, formula.target_unit)
    cot = [
        f"Parsed target: {parsed.target_quantity}",
        f"Selected formula: {formula.expression} ({formula.formula_id})",
        f"Computed with Python: {answer}",
    ]
    if formula.formula_id == "transformer_secondary_voltage":
        cot.insert(
            2,
            (
                "Substituted: V_secondary = "
                f"{parsed.variables['V_primary']:.6g} * "
                f"{parsed.variables['N_secondary']:.6g}/{parsed.variables['N_primary']:.6g}"
            ),
        )
    confidence = 0.95 if not getattr(parsed, "ambiguity", False) else 0.8
    return PhysicsSolution(
        success=True,
        answer=answer,
        explanation=physics_explanation(formula, parsed.variables, answer_value),
        formula_id=formula.formula_id,
        variables=parsed.variables,
        cot=cot,
        confidence=confidence,
        parsed=parsed,
        fallback_used=fallback_used,
        model_calls=model_calls,
    )


def solve(question: str, use_llm_extraction: bool = True, use_search: bool = False, llm_client: Any = None) -> PhysicsSolution:
    """Deterministic-first physics solver (LLM/experimental fallbacks disabled).

    This minimal implementation parses the question and runs the deterministic
    formula-based computation. It intentionally disables LLM and experimental
    fallback branches so the module remains importable and predictable while
    we prune archived helpers.
    """
    normalized = guardrail_prompt_text(question).normalized_text
    unsupported = _unsupported_context(normalized)
    if unsupported:
        parsed = ParsedPhysicsProblem(question=normalized, formula_id=None, target_quantity=None, ambiguity=[unsupported])
        solution = _compute(parsed)
        solution.error = unsupported
        return solution
    parsed = parse_physics_question(normalized)
    return _compute(parsed)


def solve_from_llm_suggestion(question: str, suggestion: dict) -> PhysicsSolution:
    """Validate and recompute a suggestion produced by an LLM fallback.

    Expected `suggestion` shape: {"formula_id": str, "variables": {..}, "target_quantity": str}
    This function coerces numeric variables and calls the deterministic compute path.
    """
    if not isinstance(suggestion, dict):
        parsed = ParsedPhysicsProblem(question=question, formula_id=None, target_quantity=None, ambiguity=["Invalid LLM suggestion."])
        return _compute(parsed, fallback_used=True, model_calls=1)
    formula_id = str(suggestion.get("formula_id") or suggestion.get("formula") or "").strip()
    target = str(suggestion.get("target_quantity") or suggestion.get("target") or suggestion.get("target_quantity") or "unknown")
    raw_vars = suggestion.get("variables") or {}
    if not formula_id or not isinstance(raw_vars, dict):
        parsed = ParsedPhysicsProblem(question=question, formula_id=None, target_quantity=target, ambiguity=["Malformed LLM suggestion."])
        return _compute(parsed, fallback_used=True, model_calls=1)
    variables: dict[str, float] = {}
    for k, v in raw_vars.items():
        try:
            if isinstance(v, (int, float)):
                variables[str(k)] = float(v)
            else:
                variables[str(k)] = float(str(v).strip())
        except Exception:
            # Skip non-numeric values; validation will fail downstream if required vars missing.
            continue
    parsed = ParsedPhysicsProblem(
        question=question,
        formula_id=formula_id,
        target_quantity=target,
        variables=variables,
        quantities=[],
        ambiguity=["Variables supplied by LLM fallback."],
    )
    return _compute(parsed, fallback_used=True, model_calls=0)


def solve_from_llm_code(question: str, code: str) -> PhysicsSolution:
    """Stub executor for LLM-generated code. For safety and test predictability this returns
    an unsuccessful PhysicsSolution unless the code explicitly prints a single numeric
    value and unit on stdout as '<value> <unit>'.
    """
    try:
        import subprocess
        from tempfile import NamedTemporaryFile
        from pathlib import Path

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = Path(f.name)
        res = subprocess.run(["python", str(temp_path)], capture_output=True, text=True, timeout=3)
        temp_path.unlink(missing_ok=True)
        if res.returncode != 0:
            return PhysicsSolution(success=False, answer="unknown", explanation=f"LLM code failed: {res.stderr[:200]}", formula_id="llm_code_gen", confidence=0.3, fallback_used=True, model_calls=0)
        out = res.stdout.strip()
        parts = out.split()
        if len(parts) >= 2:
            try:
                val = float(parts[0])
                unit = parts[1]
                answer = format_best_unit(val, unit)
                return PhysicsSolution(success=True, answer=answer, explanation="LLM code produced numeric output.", formula_id="llm_code_gen", confidence=0.6, fallback_used=True, model_calls=0)
            except Exception:
                pass
        return PhysicsSolution(success=False, answer="unknown", explanation="LLM code produced no parseable numeric output.", formula_id="llm_code_gen", confidence=0.3, fallback_used=True, model_calls=0)
    except Exception as exc:
        return PhysicsSolution(success=False, answer="unknown", explanation=f"LLM code execution error: {type(exc).__name__}", formula_id="llm_code_gen", confidence=0.2, fallback_used=True, model_calls=0)
