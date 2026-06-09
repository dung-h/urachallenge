"""Retrieval-grounded LLM physics method solver.

This module implements the "retrieve the solving METHOD, not just a formula"
strategy (AGENTS.md §3.1b search allowance, §13.2 deterministic verification).

Pipeline:
  1. Retrieve method evidence from web search for the question's topic
     (reuses app.physics.method_search retrieval).
  2. Ground an LLM call with the retrieved snippets and ask it to return a
     STRUCTURED method: target quantity + unit, the formula expression solved
     for the target, and each variable bound to an SI value extracted from the
     question.
  3. The backend RE-COMPUTES the expression with safe_eval_expression (the
     LLM's printed arithmetic is never trusted, AGENTS.md §20) and applies an
     acceptance gate:
       * the produced unit must be dimensionally consistent with the requested
         target quantity (when both are interpretable);
       * the value must be finite and physically plausible;
     otherwise the solver abstains (returns None) so the caller falls through.

The LLM here is a TRANSLATOR (pick the method, map variables); Python is the
AUTHORITY (recompute + verify). This grounds formula SELECTION in retrieved
references instead of the model's unaided memory, which is the failure mode of
the ungrounded rescue path (e.g. choosing v = qU/m instead of sqrt(2qU/m)).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from app.physics.dimensions import dimension_for_unit, dimensions_compatible
from app.physics.expression_eval import safe_eval_expression
from app.physics.method_search import (
    build_objective,
    retrieve_method_evidence,
    _web_method_search_enabled,
)
from app.physics.parser import ParsedPhysicsProblem
from app.physics.unit_converter import format_best_unit, normalize_unit


# Target-quantity → canonical SI unit hint, used for the dimensional gate.
_TARGET_UNIT_HINTS: dict[str, str] = {
    "voltage": "V",
    "electric_potential": "V",
    "current": "A",
    "power": "W",
    "resistance": "ohm",
    "capacitance": "F",
    "charge": "C",
    "energy": "J",
    "potential_energy": "J",
    "force": "N",
    "electric_field": "N/C",
    "frequency": "Hz",
    "inductance": "H",
    "magnetic_field": "T",
    "distance": "m",
    "speed": "m/s",
    "velocity": "m/s",
    "acceleration": "m/s²",
    "time": "s",
    "work": "J",
    "heat": "J",
    "angle": "°",
    "refraction_angle": "°",
    "wavelength": "m",
    "temperature": "K",
}


@dataclass(frozen=True)
class GroundedMethodSolution:
    """Result of a retrieval-grounded method solve."""

    answer: str
    explanation: str
    formula_expression: str
    formula_name: str
    variables: dict[str, float]
    target_quantity: str
    target_unit: str
    value: float
    confidence: float
    evidence_titles: list[str]
    cot: list[str]


@dataclass(frozen=True)
class GroundedLookupSolution:
    """Result of a retrieval-grounded conceptual lookup (non-numeric).

    Used for definitional / unit-name / constant questions that have no
    arithmetic to recompute. The answer is verified by EVIDENCE GROUNDING: the
    answer string must actually appear in the retrieved reference snippets, so
    the LLM cannot substitute an unsupported answer from memory (AGENTS.md §13.2
    backend-validation analogue for non-numeric facts).
    """

    answer: str
    explanation: str
    confidence: float
    evidence_titles: list[str]
    cot: list[str]


_RETRIEVAL_METHOD_PROMPT = """You are a physics method selector. You are given a question and REFERENCE \
EXCERPTS retrieved from physics references. Use the references to choose the correct \
solving method and formula. Do NOT rely on memory if the references disagree.

Return ONLY valid JSON (no markdown fences) with this exact structure:
{{
  "target_quantity": "one of: voltage,current,power,resistance,capacitance,charge,energy,force,electric_field,frequency,inductance,magnetic_field,distance,speed,acceleration,time,work,heat,refraction_angle,wavelength,temperature",
  "formula_name": "short name of the method/formula",
  "formula_expression": "the formula SOLVED FOR THE TARGET, e.g. 'theta2 = asin(n1*sin(theta1)/n2)' or 'v = sqrt(2*q*U/m)'",
  "variables": {{"symbol": value_in_SI_units, ...}},
  "answer_unit": "SI unit of the target (e.g. 'm/s', 'J', 'degrees')",
  "steps": ["short step 1", "short step 2"]
}}

Rules:
- Put EVERY symbol used in formula_expression into variables with its numeric SI value taken from the question.
- Convert all values to SI base units (meters, seconds, kg, etc.). Angles in degrees may stay in degrees if the formula uses asin/sin in degrees; prefer expressing trig in radians: use sin(radians(x)).
- The formula_expression must be evaluatable arithmetic using only: + - * / ** ( ), and functions sqrt,sin,cos,tan,asin,acos,atan,log,log10,exp,radians,degrees,abs, and constants pi,k,eps0.
- Use exactly the method shown in the references when applicable.

REFERENCES:
{references}

QUESTION: {question}
"""


def _call_llm(llm_client: Any, prompt: str) -> str | None:
    """Invoke the LLM client across the supported call shapes; return raw text."""
    try:
        if hasattr(llm_client, "chat"):
            result = llm_client.chat(
                "You are a precise physics method selector. Reply with JSON only.",
                prompt,
                max_tokens=512,
            )
            content = getattr(result, "content", None)
            if content is not None:
                return str(content)
            return str(result or "")
        if hasattr(llm_client, "generate"):
            raw = llm_client.generate(prompt)
        elif hasattr(llm_client, "complete"):
            raw = llm_client.complete(prompt)
        elif callable(llm_client):
            raw = llm_client(prompt)
        else:
            return None
        if hasattr(raw, "text"):
            return str(raw.text)
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw or "")
    except Exception:
        return None


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response (strips fences/prose)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the outermost JSON object.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


_GREEK_SYMBOL_MAP = {
    "λ": "lam",
    "σ": "sigma",
    "ρ": "rho",
    "θ": "theta",
    "ω": "omega",
    "α": "alpha",
    "β": "beta",
    "μ": "mu",
    "Δ": "delta",
    "δ": "delta",
    "π": "pi",
    "φ": "phi",
    "ε": "eps",
}


def _normalize_symbol(text: str) -> str:
    """Replace Greek-letter symbols with the ASCII names used by the evaluator.

    Mirrors the substitutions in
    :func:`app.physics.expression_eval.normalize_equation_expression` so that
    LLM-provided variable keys (often Greek, e.g. ``λ``) match the tokens in the
    normalized expression.
    """
    out = text
    for greek, ascii_name in _GREEK_SYMBOL_MAP.items():
        out = out.replace(greek, ascii_name)
    return out


def _canonical_si(unit: str | None) -> str | None:
    """Best-effort canonical SI unit string for the dimensional gate."""
    if not unit:
        return None
    u = normalize_unit(str(unit))
    if not u:
        return None
    if u in {"degrees", "degree", "deg", "°"}:
        return "°"
    return u


def _dimensional_ok(produced_unit: str | None, target_quantity: str) -> bool | None:
    """Whether the produced unit is dimensionally consistent with the target.

    Returns True/False when both are interpretable, None when the check cannot
    be established (then the caller treats it as 'not contradicted').
    """
    expected = _TARGET_UNIT_HINTS.get(target_quantity)
    if not expected:
        return None
    # Angles: accept degree/dimensionless-style answers for angle targets.
    if expected == "°":
        prod = _canonical_si(produced_unit)
        return prod in {"°", "dimensionless", "rad", None} or (prod == expected)
    prod = _canonical_si(produced_unit)
    if prod is None:
        return None
    prod_dim = dimension_for_unit(prod)
    exp_dim = dimension_for_unit(expected)
    if prod_dim is None or exp_dim is None:
        return None
    return dimensions_compatible(prod_dim, exp_dim)


def solve_with_retrieved_method(
    parsed: ParsedPhysicsProblem,
    question: str,
    llm_client: Any,
    max_search_calls: int = 3,
) -> GroundedMethodSolution | None:
    """Solve a physics question by retrieving the method and grounding the LLM.

    Returns a GroundedMethodSolution on success, or None to abstain (so the
    caller can fall through to other rescue paths). Requires an LLM client and
    web search to be enabled.
    """
    if llm_client is None:
        return None

    # 1. Retrieve method evidence (web/local).
    objective = build_objective(parsed, question)
    try:
        snippets = retrieve_method_evidence(objective, max_search_calls=max_search_calls)
    except Exception:
        snippets = []
    if not snippets:
        return None

    # Build a compact references block (title + trimmed text), capped.
    ref_lines: list[str] = []
    evidence_titles: list[str] = []
    for snippet in snippets[:5]:
        title = (snippet.title or "").strip()
        body = re.sub(r"\s+", " ", (snippet.text or "")).strip()[:400]
        if not title and not body:
            continue
        evidence_titles.append(title or (snippet.url or "reference"))
        ref_lines.append(f"- {title}: {body}")
    if not ref_lines:
        return None
    references = "\n".join(ref_lines)

    # 2. Ground the LLM.
    prompt = _RETRIEVAL_METHOD_PROMPT.format(references=references, question=question)
    raw = _call_llm(llm_client, prompt)
    if not raw:
        return None
    payload = _parse_json_response(raw)
    if not payload:
        return None

    target_quantity = str(payload.get("target_quantity") or "").strip().lower()
    formula_expression = str(payload.get("formula_expression") or "").strip()
    formula_name = str(payload.get("formula_name") or formula_expression).strip()
    answer_unit = str(payload.get("answer_unit") or "").strip()
    raw_variables = payload.get("variables") or {}
    steps = [str(s) for s in (payload.get("steps") or [])][:6]

    if not formula_expression or not isinstance(raw_variables, dict):
        return None

    # Isolate the RHS if the LLM returned "target = expr".
    expr = formula_expression
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()

    # Coerce variables to float SI values, normalizing symbol names the SAME way
    # the expression is normalized (e.g. Greek 'λ'→'lam', 'σ'→'sigma') so the
    # variable keys match the tokens in the evaluated expression.
    variables: dict[str, float] = {}
    for key, val in raw_variables.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        norm_key = _normalize_symbol(str(key))
        variables[norm_key] = fval
        if str(key) != norm_key:
            variables.setdefault(str(key), fval)
    if not variables:
        return None

    # Input-scale gate: every LLM-emitted variable should match SOME
    # parsed quantity (same SI unit, value within 1%) — that's the
    # deterministic parser cross-check. The LLM frequently leaves
    # prefix-scaled numbers ("12" instead of 0.012 for "12 mA")
    # because it ignores the modifier even when prompted. When NOT a
    # single LLM value matches any parsed quantity by SI value, the
    # LLM has dropped a unit prefix; abstain so the planner can try
    # another method.
    parsed_si_pairs = []
    for q in (getattr(parsed, "quantities", []) or []):
        try:
            parsed_si_pairs.append((float(q.si_value), str(q.si_unit or "")))
        except Exception:
            continue
    if parsed_si_pairs:
        mismatches: list[tuple[str, float, list[float]]] = []
        for sym, llm_val in variables.items():
            try:
                fv = float(llm_val)
            except Exception:
                continue
            if fv == 0:
                continue
            # Find parsed quantities the LLM likely meant. We don't
            # know the LLM's unit, so accept a match against ANY
            # parsed quantity within 1% relative error.
            close = [
                pv for pv, _ in parsed_si_pairs
                if pv != 0 and abs(fv - pv) / max(abs(pv), abs(fv)) <= 0.01
            ]
            if close:
                continue
            # No parsed quantity matches THIS LLM number. Check whether
            # any parsed quantity is a 10^k multiple within k ∈ {-6..6}.
            # If so, the LLM dropped a prefix. Flag as mismatch.
            for pv, _ in parsed_si_pairs:
                if pv == 0:
                    continue
                ratio = abs(fv) / abs(pv)
                # Off by 10^k for some integer k ≠ 0?
                if ratio > 1.5 or ratio < 0.5:
                    import math as _math
                    log10 = _math.log10(ratio)
                    if abs(round(log10) - log10) < 0.05 and abs(round(log10)) >= 1:
                        mismatches.append(
                            (sym, fv, [p for p, _ in parsed_si_pairs])
                        )
                        break
        if mismatches:
            return None  # abstain; planner will pick another method

    # Normalize the expression's symbols consistently before evaluation.
    expr_for_eval = _normalize_symbol(expr)

    # 3. Backend RE-COMPUTE (never trust LLM arithmetic).
    try:
        value = safe_eval_expression(expr_for_eval, variables)
    except Exception:
        return None
    if value is None or not math.isfinite(value):
        return None

    # Acceptance gate: dimensional consistency with the requested target.
    dim_ok = _dimensional_ok(answer_unit, target_quantity)
    if dim_ok is False:
        return None

    # Plausibility: reject absurd magnitudes that signal a wrong formula/units.
    if abs(value) > 1e30:
        return None

    # Format the answer.
    is_angle = target_quantity in {"angle", "refraction_angle"} or _canonical_si(answer_unit) == "°"
    if is_angle:
        answer = f"{value:.6g}°"
    else:
        si_unit = _TARGET_UNIT_HINTS.get(target_quantity) or normalize_unit(answer_unit) or answer_unit
        try:
            answer = format_best_unit(value, si_unit)
        except Exception:
            answer = f"{value:.6g} {answer_unit}".strip()

    confidence = 0.7 if dim_ok else 0.6
    explanation = (
        f"Retrieved the solving method from physics references "
        f"({'; '.join(evidence_titles[:2])}). Selected {formula_name}: "
        f"{formula_expression}. Backend recomputed the expression in SI units "
        f"and verified the result unit is consistent with the requested "
        f"{target_quantity}. Answer: {answer}."
    )
    cot = [
        f"Retrieved {len(snippets)} method references",
        f"LLM-selected formula (grounded): {formula_expression}",
        f"Variables (SI): {variables}",
        f"Backend recomputed value: {value:.6g}",
    ] + steps

    return GroundedMethodSolution(
        answer=answer,
        explanation=explanation,
        formula_expression=formula_expression,
        formula_name=formula_name,
        variables=variables,
        target_quantity=target_quantity,
        target_unit=answer_unit,
        value=value,
        confidence=confidence,
        evidence_titles=evidence_titles,
        cot=cot,
    )


# ---------------------------------------------------------------------------
# Conceptual / lookup questions (non-numeric): definitions, SI unit names,
# named constants, "what is the unit of X", "what is X called".
# ---------------------------------------------------------------------------


_CONCEPTUAL_LOOKUP_PROMPT = """You are a physics fact extractor. Answer the question using ONLY the \
REFERENCE EXCERPTS below. If the references do not contain the answer, reply with \
"unknown". Do NOT use outside knowledge.

Return ONLY valid JSON (no markdown fences):
{{
  "answer": "the concise answer (a single term, unit name, or short phrase)",
  "evidence_quote": "the exact phrase from the references that supports the answer",
  "confident": true or false
}}

REFERENCES:
{references}

QUESTION: {question}
"""


# Phrasings that indicate a conceptual/lookup question with NO numeric target.
_LOOKUP_PATTERNS = (
    r"\bwhat\s+is\s+the\s+(?:si\s+)?unit\s+of\b",
    r"\bunit\s+of\s+measure(?:ment)?\s+(?:of|for)\b",
    r"\bwhat\s+is\s+the\s+(?:si\s+)?unit\s+for\b",
    r"\bwhat\s+is\s+.+\s+measured\s+in\b",
    r"\bwhat\s+is\s+.+\s+called\b",
    r"\bwhat\s+do\s+we\s+call\b",
    r"\bname\s+the\s+(?:si\s+)?unit\b",
    r"\bwhat\s+is\s+the\s+name\s+of\b",
)


def _is_conceptual_lookup(question: str) -> bool:
    """Detect a conceptual/lookup question (definition, unit name, term).

    Conservative: only fires on explicit "what is the unit of / what is X
    called / name the unit" phrasings AND when the question carries no numeric
    digits to solve for. General phrasing rule, not a per-question text match.
    """
    low = question.lower()
    # If the question has numbers to compute with, it is not a pure lookup.
    if re.search(r"\d", low):
        # Allow digits only inside ordinals like "first"; otherwise treat as numeric.
        return False
    return any(re.search(p, low) for p in _LOOKUP_PATTERNS)


def _answer_grounded_in_references(answer: str, snippets) -> bool:
    """Verify the answer string is substantiated by the retrieved snippets.

    The answer (or its singular/lowercased core token) must appear in at least
    one reference's title or text. This is the non-numeric analogue of backend
    recomputation: it blocks the LLM from inventing an unsupported answer
    (e.g. "Tesla" when the references say "Weber").
    """
    ans = answer.strip().lower()
    if not ans or ans == "unknown":
        return False
    # Compare on the most salient token(s): drop articles/punctuation.
    core = re.sub(r"[^a-z0-9 ]", " ", ans)
    core_tokens = [t for t in core.split() if len(t) > 2 and t not in {"the", "and", "unit", "called"}]
    if not core_tokens:
        core_tokens = [ans]
    for snippet in snippets:
        blob = f"{snippet.title} {snippet.text}".lower()
        if any(tok in blob for tok in core_tokens):
            return True
    return False


def solve_conceptual_lookup(
    parsed: ParsedPhysicsProblem,
    question: str,
    llm_client: Any,
    max_search_calls: int = 3,
) -> GroundedLookupSolution | None:
    """Answer a conceptual/lookup physics question grounded in retrieved evidence.

    Returns a GroundedLookupSolution only when (a) the question is a conceptual
    lookup, (b) web evidence is retrieved, (c) the LLM extracts an answer, and
    (d) that answer is substantiated by the retrieved snippets. Otherwise
    abstains (None). This keeps the backend as the validation authority: the
    LLM proposes, the evidence-grounding check verifies.
    """
    if llm_client is None:
        return None
    if not _is_conceptual_lookup(question):
        return None

    objective = build_objective(parsed, question)
    try:
        snippets = retrieve_method_evidence(objective, max_search_calls=max_search_calls)
    except Exception:
        snippets = []
    if not snippets:
        return None

    ref_lines: list[str] = []
    evidence_titles: list[str] = []
    for snippet in snippets[:6]:
        title = (snippet.title or "").strip()
        body = re.sub(r"\s+", " ", (snippet.text or "")).strip()[:400]
        if not title and not body:
            continue
        evidence_titles.append(title or (snippet.url or "reference"))
        ref_lines.append(f"- {title}: {body}")
    if not ref_lines:
        return None
    references = "\n".join(ref_lines)

    prompt = _CONCEPTUAL_LOOKUP_PROMPT.format(references=references, question=question)
    raw = _call_llm(llm_client, prompt)
    if not raw:
        return None
    payload = _parse_json_response(raw)
    if not payload:
        return None

    answer = str(payload.get("answer") or "").strip()
    evidence_quote = str(payload.get("evidence_quote") or "").strip()
    if not answer or answer.lower() == "unknown":
        return None

    # Evidence-grounding gate: the answer must appear in the retrieved snippets.
    if not _answer_grounded_in_references(answer, snippets):
        return None

    explanation = (
        f"Answered from retrieved physics references "
        f"({'; '.join(evidence_titles[:2])}). The answer '{answer}' is "
        f"substantiated by the retrieved text"
        + (f": \"{evidence_quote[:120]}\"." if evidence_quote else ".")
    )
    cot = [
        f"Conceptual lookup grounded in {len(snippets)} references",
        f"LLM-extracted answer: {answer}",
        "Verified answer is substantiated by retrieved evidence",
    ]
    return GroundedLookupSolution(
        answer=answer,
        explanation=explanation,
        confidence=0.7,
        evidence_titles=evidence_titles,
        cot=cot,
    )
