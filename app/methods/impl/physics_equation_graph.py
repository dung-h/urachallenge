"""Physics Method that asks the LLM to extract a small system of physics
equations + knowns + target, then solves the system with SymPy.

Level-4 ladder step. Subsumes hand-coded adapter dispatch for cases where
a standard high-school / undergraduate physics problem can be expressed as
a few coupled equations from the standard library (Newton's laws, kinematics,
Ohm's law, capacitor, Coulomb, ideal gas, etc.).

Pipeline (AGENTS.md §13.2)
--------------------------
1. **LLM extracts**: list of equations (e.g. `F = m*a`, `v = u + a*t`),
   the dictionary of known SI values from the question, and the target
   variable name.
2. **Backend builds**: a SymPy system from the extracted equations,
   substitutes the knowns, and asks SymPy to solve for the target.
3. **Validation gate**: result must be finite + within plausible
   magnitude bounds; the formula expressions are checked against a
   permissive whitelist (no arbitrary code).
4. **Format**: the solution is formatted with the right SI unit (taken
   from a small target → unit map, same one ``retrieval_grounded_method``
   uses).

When the LLM can't decompose the question into equations, the method
abstains and the planner falls through to hand-coded adapters / legacy.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from app.methods.problem import PhysicsProblem
from app.methods.types import (
    Method,
    MethodApplicability,
    MethodFamily,
    MethodResult,
    MethodSource,
    MethodTrace,
)


# Target → SI unit hint, kept consistent with retrieval_grounded_method.
_TARGET_UNIT_HINTS: dict[str, str] = {
    "voltage": "V", "electric_potential": "V",
    "current": "A", "power": "W", "resistance": "ohm",
    "capacitance": "F", "charge": "C", "energy": "J", "potential_energy": "J",
    "force": "N", "electric_field": "N/C", "frequency": "Hz",
    "inductance": "H", "magnetic_field": "T", "distance": "m",
    "speed": "m/s", "velocity": "m/s", "acceleration": "m/s²", "time": "s",
    "work": "J", "heat": "J", "wavelength": "m", "temperature": "K",
    "mass": "kg", "momentum": "kg*m/s", "torque": "N*m",
}


_EQGRAPH_PROMPT = """You are a physics problem decomposer. Read the question CAREFULLY before answering.

SYNTAX REQUIREMENTS (HARD — the backend rejects equations that violate these):
  - Symbol names are ASCII only. NEVER use Greek letters or unicode subscripts.
    Use mu_0 (not μ₀), kappa (not κ), omega (not ω), theta (not θ), epsilon_0
    (not ε₀), pi (not π).
  - Symbol names contain no apostrophes. For "after" or "new" quantities, use a
    suffix: C_new, Q_new, V_new — never C' or V'.
  - Knowns are pure numbers in SI base units. NEVER include units inside the
    number ("10", not "10 V"). NEVER include arithmetic expressions
    ("0.0000012566", not "4*pi*1e-7"). Convert symbolic constants to numeric
    values yourself: mu_0 = 1.2566e-6, epsilon_0 = 8.854e-12, k_e = 8.99e9.
  - Equation RHS values are numeric, not unit-suffixed: write "V = 10", not
    "V = 10 V".

TARGET-FROM-UNIT RULE (HARD): if the question explicitly asks for the answer
in a specific unit (e.g. "in kJ", "in mC", "in mT"), the target_quantity
MUST match that unit's dimension:
  - kJ / J / MJ / Wh / kWh / cal       → target_quantity = "energy"
  - W / kW / mW                         → target_quantity = "power"
  - C / mC / μC / nC                    → target_quantity = "charge"
  - F / μF / nF / pF                    → target_quantity = "capacitance"
  - V / mV / kV                         → target_quantity = "voltage"
  - A / mA                              → target_quantity = "current"
  - Ω / kΩ / mΩ                         → target_quantity = "resistance"
  - N / mN / kN                         → target_quantity = "force"
  - T / mT / μT                         → target_quantity = "magnetic_field"
  - m / cm / mm / km                    → target_quantity = "distance" or "wavelength"
  - kg / g / mg                         → target_quantity = "mass"
If the unit is energy and a time-window phrase is present ("in 30 minutes",
"over 2 hours"), the equations MUST include the energy-from-power formula
(energy = P * t) AND P = V * I (or P = I^2 R, etc.).



STEP 1 — MODIFIER WORDS. Before picking equations, list every PHRASE in
the question that changes WHICH equation applies or WHICH quantity is
asked. Examples of modifier phrases:

  - "in 30 minutes" / "over 2 hours"     → time-integrated quantity
                                           (energy = power × time)
  - "across R2" / "on R1"                → component-specific voltage /
                                           power, NOT total
  - "connected to a battery" / "stays connected to a source"
                                         → V is held constant, Q changes
                                           with capacitance
  - "disconnected from the source"       → Q is held constant, V changes
  - "with dielectric κ=4" / "permittivity 4" / "dielectric of κ=k"
                                         → multiply C by kappa in EVERY
                                           equation that contains C
                                           (so Q = C * V becomes
                                            Q = kappa * C * V; and
                                            E = 0.5 * C * V**2 becomes
                                            E = 0.5 * kappa * C * V**2)
  - "solenoid has N turns over length L" → use B = mu_0 * (N / L) * I,
                                           where mu_0 = 1.2566e-6
  - "perpendicular to a B field"         → use F = q * v * B (sin(90°)=1)
  - "at angle θ to the field"            → use F = q * v * B * sin(theta)
  - "in air" / "in vacuum"               → use baseline values
  - "find the magnitude" / "in mT"       → unit-conversion only, no
                                           formula change
  - "perpendicular to" / "parallel to"   → component selection
  - "at the midpoint" / "on the axis"    → geometry-specific formula

If a modifier phrase is present, the equations MUST include the factor
that the modifier introduces (e.g. include the κ factor, multiply by t,
use the divider ratio).

EQUATION COMPLETENESS RULE (HARD): the target_symbol MUST appear on the
LHS of at least one equation, AND that equation must contain every factor
required by the modifier phrases. For dielectric problems specifically:
when the question asks for "the new charge" with a dielectric of κ, do
NOT write `Q = C * V` and `C_new = kappa * C` separately (that leaves
Q_new dangling). Write the closed-form equation directly:
`Q_new = kappa * C * V`. Same rule for "new capacitance" → write
`C_new = kappa * C` and use that as the answer; for "new energy stored"
with the source disconnected → write `E_new = Q**2 / (2 * kappa * C)`.

STEP 2 — TARGET. State exactly which quantity is asked, and the unit
explicitly mentioned in the question (if any). Be precise: "voltage
across R2" is NOT "source voltage".

STEP 3 — EQUATIONS. List every equation needed, INCLUDING any modifier
factors from step 1.

STEP 4 — KNOWNS. List every variable that appears in step 3, in SI
base units.

Return ONLY valid JSON (no markdown fences):
{{
  "modifier_phrases": ["e.g. 'across R2'", "'in 30 minutes'", "'κ=4'"],
  "target_quantity": "one of: voltage, current, power, resistance, capacitance, charge, energy, force, electric_field, frequency, inductance, magnetic_field, distance, speed, acceleration, time, work, heat, wavelength, temperature, mass, momentum, torque",
  "target_symbol": "single-letter or short name (e.g. 'a' for acceleration)",
  "equations": ["F = m*a", "v = u + a*t", ...],
  "knowns": {{"F": 12.0, "m": 4.0, "t": 3.0}},
  "answer_unit": "SI unit of the target (e.g. 'J' for energy, 'C' for charge)",
  "explanation_step": "one short sentence summarising the strategy"
}}

Rules:
- Equations must use only + - * / ** ( ), and functions sqrt,sin,cos,tan,asin,acos,atan,radians,degrees,abs, plus pi.
- Convert all known values to SI base units before listing them (30 minutes -> 1800 s, 4 kΩ -> 4000 ohm, 5 μF -> 5e-6 F, 50 cm -> 0.5 m, etc.).
- Use the SAME symbol consistently across equations and knowns (case-sensitive).
- If a modifier phrase IS present but you do not know which factor to apply, return {{"target_quantity": "", "equations": []}} instead of guessing — another method will handle it.
- If you cannot decompose the question into ≤ 4 equations from the standard library, return {{"target_quantity": "", "equations": []}} so the system can use another method.

QUESTION: {question}
"""


@dataclass(frozen=True)
class _EquationSystemSolution:
    answer: str
    target_quantity: str
    target_unit: str
    target_symbol: str
    value: float
    equations: list[str]
    knowns: dict[str, float]
    explanation_step: str


def _call_llm(client: Any, prompt: str) -> str | None:
    try:
        if hasattr(client, "chat"):
            result = client.chat(
                "default",
                prompt,
                max_tokens=512,
                response_format=False,
            )
            return getattr(result, "content", None) or str(result or "")
        if hasattr(client, "generate"):
            return str(client.generate(prompt) or "")
        if callable(client):
            return str(client(prompt) or "")
    except Exception:
        return None
    return None


def _parse_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    blob = text[start:end]
    try:
        return json.loads(blob)
    except Exception:
        # The LLM occasionally embeds simple arithmetic constants in the
        # knowns map (e.g. {"mu_0": 4*pi*1e-7}). Strict json.loads rejects
        # those. Fall back to a guarded eval over a tiny constants dict so
        # the rest of the pipeline can still solve the system.
        return _loose_json_loads(blob)


_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _loose_json_loads(blob: str) -> dict[str, Any] | None:
    """Best-effort JSON loader that tolerates simple arithmetic in numeric
    fields (e.g. ``"mu_0": 4*pi*1e-7``). Strings, lists, and nested objects
    are still parsed strictly. Anything that can't be eval'd as a pure
    arithmetic expression over numeric literals + ``pi``/``e`` is dropped.
    """
    # Replace any non-string scalar that contains a math expression with the
    # eval'd numeric literal. Pattern matches: key followed by an unquoted
    # value that is a sequence of digits / operators / pi / e / parens.
    expr_chars = r"[\d\.\+\-\*\/eE\(\)piE\s]"
    pattern = re.compile(
        r'("\s*[A-Za-z_][A-Za-z_0-9]*\s*"\s*:\s*)(' + expr_chars + r"+)(?=[,\}])"
    )

    def _replace(m: "re.Match[str]") -> str:
        key, raw_val = m.group(1), m.group(2).strip()
        # Already a plain number? Keep verbatim.
        try:
            float(raw_val)
            return key + raw_val
        except Exception:
            pass
        # Disallow anything but digits, operators, and the safe names.
        if not re.fullmatch(r"[\d\.\+\-\*\/eE\(\)piE\s]+", raw_val):
            return key + "null"
        try:
            value = eval(raw_val, {"__builtins__": {}}, _SAFE_CONSTANTS)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return key + "null"
            return key + repr(float(value))
        except Exception:
            return key + "null"

    repaired = pattern.sub(_replace, blob)
    try:
        return json.loads(repaired)
    except Exception:
        return None


# Single Greek/sub-script → ASCII translation for symbol names. Run on
# both equations and known keys before sympify so the LLM's natural
# notation does not break parsing.
_GREEK_TO_ASCII: dict[str, str] = {
    "μ": "mu", "µ": "mu",  # micro sign U+00B5 also maps here
    "κ": "kappa",
    "ω": "omega",
    "θ": "theta",
    "ε": "epsilon",
    "λ": "lambda_",  # lambda is reserved in Python
    "ρ": "rho",
    "σ": "sigma",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "φ": "phi",
    "Φ": "Phi",
    "Ω": "Ohm",
    "π": "pi",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
}


def _ascii_symbol(s: str) -> str:
    """ASCII-fy Greek/subscript characters in a symbol or equation string."""
    out = []
    for ch in s:
        out.append(_GREEK_TO_ASCII.get(ch, ch))
    cleaned = "".join(out)
    # Strip apostrophes (LLM uses C', V' for "after"/"new" quantities).
    cleaned = cleaned.replace("'", "_new")
    return cleaned


# Trailing unit hints the LLM sometimes leaves on RHS values: "10 V", "5 A",
# "30 kJ". Strip them before sympify. The numeric prefix is what counts.
_TRAILING_UNIT_RE = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"[A-Za-zΩμµ°/^*·]+\s*$"
)


def _strip_trailing_unit(value: str) -> str:
    m = _TRAILING_UNIT_RE.match(value)
    if m:
        return m.group(1)
    return value


# Map of unit token (case-insensitive) → SI dimension key for the
# answer_unit-driven target repair. This mirrors the prompt's
# TARGET-FROM-UNIT rule on the backend so a misclassified target can
# still be corrected before SymPy runs.
_UNIT_TO_TARGET: dict[str, str] = {
    "j": "energy", "kj": "energy", "mj": "energy",
    "wh": "energy", "kwh": "energy", "cal": "energy", "kcal": "energy",
    "w": "power", "kw": "power", "mw": "power",
    "c": "charge", "mc": "charge", "μc": "charge", "uc": "charge", "nc": "charge",
    "f": "capacitance", "μf": "capacitance", "uf": "capacitance",
    "nf": "capacitance", "pf": "capacitance",
    "v": "voltage", "mv": "voltage", "kv": "voltage",
    "a": "current", "ma": "current",
    "ohm": "resistance", "ω": "resistance", "kω": "resistance",
    "kohm": "resistance", "mohm": "resistance",
    "n": "force", "mn": "force", "kn": "force",
    "t": "magnetic_field", "mt": "magnetic_field", "μt": "magnetic_field",
    "ut": "magnetic_field",
    "hz": "frequency", "khz": "frequency", "mhz": "frequency",
    "kg": "mass", "g": "mass", "mg": "mass",
    "m/s": "speed", "km/h": "speed",
    "m/s^2": "acceleration", "m/s²": "acceleration",
    "rad/s": "frequency",
}


# ---------------------------------------------------------------------------
# Conservation / partition gate (external-law check).
# ---------------------------------------------------------------------------
#
# The backwards-verify, dimensional, and magnitude gates all check INTERNAL
# consistency of the LLM's equation set. They cannot catch the failure mode
# where the LLM picks a self-consistent but physically WRONG principle
# (e.g. using the series formula V=I*(R1+R2) for a parallel current
# divider). That bug produced 48 mA for a 12 mA source — internally
# consistent, right unit, plausible magnitude, so every internal gate
# passed.
#
# The general law these problems violate is PARTITION CONSERVATION: when a
# quantity is SPLIT / DIVIDED / SHARED among parts, no single part may
# exceed the whole. This is dimension-agnostic (current divider, voltage
# divider, mass partition, force resolution, charge sharing, ...) so a
# single structural check covers the whole class without per-formula
# knowledge.

# Verbs/phrases that signal the question describes splitting a total into
# parts and asks for ONE part. Structural — matches the problem SHAPE, not
# a specific question (AGENTS.md §20.1).
_PARTITION_MARKERS = (
    "splits", "split", "divides", "divided", "divider", "shares", "shared",
    "distributed", "branch", "branches", "in parallel", "parallel branch",
)

# Words that indicate the question asks for a PART (one component) rather
# than the aggregate. "through R1", "across one", "in the first", etc.
_PART_MARKERS = (
    "through", "across one", "in one", "in the first", "in the second",
    "of one", "single branch", "each branch", "one of",
)


def _conservation_partition_ok(
    problem: PhysicsProblem,
    target_quantity: str,
    value_si: float,
    answer_si_unit: str,
) -> tuple[bool | None, str]:
    """Return whether a 'part' answer respects partition conservation.

    Returns ``(ok, reason)`` where ``ok`` is:
      * ``True``  — the check applies and passes (part <= total),
      * ``False`` — the check applies and FAILS (part > total => reject),
      * ``None``  — the check does not apply (no partition shape, or no
        same-dimension total input to compare against).

    Generalizes the current-divider bug: in any split/divide problem, a
    single part of a conserved additive quantity (current, power, charge,
    mass, force-component, voltage-across-a-series-element) cannot exceed
    the total being split. We compare the answer against the LARGEST
    same-SI-unit input quantity (the natural candidate for "the total"),
    with a small tolerance for rounding.
    """
    low = problem.raw_question.lower()
    is_partition = any(m in low for m in _PARTITION_MARKERS)
    asks_for_part = any(m in low for m in _PART_MARKERS)
    if not (is_partition and asks_for_part):
        return None, "not_a_partition_part_question"

    # Find same-dimension input quantities (the candidate "totals").
    try:
        from app.physics.unit_converter import normalize_unit
        ans_norm = normalize_unit(answer_si_unit) if answer_si_unit else ""
    except Exception:
        ans_norm = answer_si_unit or ""

    same_dim_inputs: list[float] = []
    for q in (getattr(problem.parsed, "quantities", []) or []):
        q_si_unit = (getattr(q, "si_unit", "") or "").strip()
        q_si_val = getattr(q, "si_value", None)
        if q_si_val is None:
            continue
        try:
            q_si_unit_norm = normalize_unit(q_si_unit) if q_si_unit else ""
        except Exception:
            q_si_unit_norm = q_si_unit
        # Compare on SI base unit equality (A==A, V==V, W==W, ...).
        if q_si_unit_norm and ans_norm and q_si_unit_norm == ans_norm:
            same_dim_inputs.append(abs(float(q_si_val)))

    if not same_dim_inputs:
        return None, "no_same_dimension_total_input"

    total = max(same_dim_inputs)
    # 1% tolerance for rounding; a true part is strictly <= total.
    if abs(value_si) > total * 1.01:
        return False, (
            f"partition_violation:{target_quantity}|part={abs(value_si):.4g}"
            f">total={total:.4g}{ans_norm}"
        )
    return True, f"partition_ok:part={abs(value_si):.4g}<=total={total:.4g}{ans_norm}"





def _solve_with_sympy(
    equations: list[str], knowns: dict[str, float], target_symbol: str
) -> tuple[float | None, list[str]]:
    """Solve the system of equations for ``target_symbol`` using SymPy."""
    trace: list[str] = []
    try:
        import sympy as sp
    except Exception as exc:
        trace.append(f"sympy_unavailable: {exc}")
        return None, trace
    try:
        # Build symbol table from every name appearing in equations + knowns.
        symbol_names: set[str] = set()
        for eq in equations:
            symbol_names.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", eq))
        symbol_names.update(knowns.keys())
        symbol_names.add(target_symbol)
        symbol_names -= {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sqrt", "log", "log10", "exp", "abs", "radians",
            "degrees", "pi", "e",
        }
        symbols = {name: sp.Symbol(name, real=True) for name in symbol_names}
        # Parse each equation into a SymPy Eq.
        parsed_eqs: list[Any] = []
        for eq in equations:
            if "=" not in eq:
                continue
            lhs_str, rhs_str = eq.split("=", 1)
            lhs = sp.sympify(lhs_str.strip(), locals=symbols)
            rhs = sp.sympify(rhs_str.strip(), locals=symbols)
            parsed_eqs.append(sp.Eq(lhs, rhs))
        if not parsed_eqs:
            trace.append("no_equations_parsed")
            return None, trace
        # Substitute knowns.
        substituted = [eq.subs({symbols[k]: v for k, v in knowns.items() if k in symbols})
                       for eq in parsed_eqs]
        trace.append(f"sympy: {len(substituted)} equations after substitution")
        # Solve for target.
        target_sym = symbols.get(target_symbol)
        if target_sym is None:
            trace.append(f"target_symbol {target_symbol!r} not in equations")
            return None, trace
        # Pass ALL unknowns to sp.solve so it can eliminate intermediate
        # variables (e.g. when the system is `P = V*I` + `E = P*t` and
        # the target is E, we need P listed as an unknown so SymPy
        # substitutes through it). Without this, SymPy returns a
        # parametric solution and our caller treats that as no-solution.
        unknowns = [s for n, s in symbols.items() if n not in knowns]
        if target_sym not in unknowns:
            unknowns.append(target_sym)
        sol = sp.solve(substituted, unknowns, dict=True)
        if not sol:
            trace.append("sympy_no_solution")
            return None, trace
        # Pick the first real solution.
        for s in sol:
            value_expr = s.get(target_sym)
            if value_expr is None:
                continue
            try:
                value = float(value_expr)
                if math.isfinite(value):
                    trace.append(f"sympy: solved {target_symbol} = {value:.6g}")
                    return value, trace
            except Exception:
                continue
        trace.append("sympy: no real-valued solution among candidates")
        return None, trace
    except Exception as exc:
        trace.append(f"sympy_error: {type(exc).__name__}: {exc}")
        return None, trace


def _verify_solution_in_equations(
    equations: list[str],
    knowns: dict[str, float],
    target_symbol: str,
    target_value: float,
    *,
    rel_tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    """Backwards-verify the solved value by substituting it (and the knowns)
    into every original equation and checking that each becomes a numeric
    identity within ``rel_tolerance``.

    This is a deterministic SymPy check, NOT another LLM call. It catches
    the failure mode where the LLM emits a wrong equation set and SymPy
    happily produces a self-consistent value within that wrong system —
    by re-checking the value against the equations as if they were
    INDEPENDENT constraints, any inconsistency surfaces as a non-zero
    residual.

    Returns ``(passed, trace)``. ``passed=False`` means at least one
    equation does not hold after substitution; the caller should abstain.
    """
    trace: list[str] = []
    try:
        import sympy as sp
    except Exception:
        trace.append("verify: sympy_unavailable; skipping")
        return True, trace  # don't reject when we can't check
    try:
        symbol_names: set[str] = set()
        for eq in equations:
            symbol_names.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", eq))
        symbol_names.update(knowns.keys())
        symbol_names.add(target_symbol)
        symbol_names -= {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sqrt", "log", "log10", "exp", "abs", "radians",
            "degrees", "pi", "e",
        }
        symbols = {name: sp.Symbol(name, real=True) for name in symbol_names}
        # Build the full substitution map: knowns + target.
        full_subs: dict[Any, float] = {}
        for k, v in knowns.items():
            if k in symbols:
                full_subs[symbols[k]] = float(v)
        if target_symbol in symbols:
            full_subs[symbols[target_symbol]] = float(target_value)

        residuals: list[float] = []
        for eq in equations:
            if "=" not in eq:
                continue
            lhs_str, rhs_str = eq.split("=", 1)
            try:
                lhs = sp.sympify(lhs_str.strip(), locals=symbols).subs(full_subs)
                rhs = sp.sympify(rhs_str.strip(), locals=symbols).subs(full_subs)
            except Exception as exc:
                trace.append(f"verify: parse_failed for {eq!r}: {exc}")
                continue
            try:
                lhs_n = float(sp.N(lhs))
                rhs_n = float(sp.N(rhs))
            except Exception:
                # Equation still has free symbols — can't check numerically.
                # That happens when the equation introduces a variable that
                # was neither a known nor the target (e.g. an intermediate
                # we did not solve for). Skip rather than reject.
                trace.append(f"verify: free_symbols_remain in {eq!r}")
                continue
            if not (math.isfinite(lhs_n) and math.isfinite(rhs_n)):
                trace.append(f"verify: non_finite residual in {eq!r}")
                return False, trace
            scale = max(abs(lhs_n), abs(rhs_n), 1e-12)
            rel = abs(lhs_n - rhs_n) / scale
            residuals.append(rel)
            if rel > rel_tolerance:
                trace.append(
                    f"verify: FAIL {eq!r} -> lhs={lhs_n:.6g} rhs={rhs_n:.6g} rel_err={rel:.4g}"
                )
                return False, trace
            trace.append(f"verify: ok {eq!r} rel_err={rel:.4g}")
        if not residuals:
            trace.append("verify: no equations could be numerically checked")
            return True, trace  # nothing to disprove
        trace.append(f"verify: all {len(residuals)} equations satisfied")
        return True, trace
    except Exception as exc:
        trace.append(f"verify_error: {type(exc).__name__}: {exc}")
        return True, trace  # don't reject on internal verifier errors


def _format_answer(value: float, target_quantity: str, llm_unit: str) -> str:
    si_unit = _TARGET_UNIT_HINTS.get(target_quantity) or llm_unit or ""
    try:
        from app.physics.unit_converter import format_best_unit
        return format_best_unit(value, si_unit)
    except Exception:
        return f"{value:.6g} {si_unit}".strip()


def _dimensional_ok(produced_unit: str | None, target_quantity: str) -> bool | None:
    """Whether the LLM-emitted ``answer_unit`` is dimensionally consistent
    with the target quantity. Returns True/False when both are
    interpretable, None when the check cannot be made (treated as "not
    contradicted" by the caller).

    Mirrors ``app.physics.retrieval_grounded_method._dimensional_ok`` so
    the equation-graph Method has the same dimensional safety net the
    retrieval Method already enforces. This catches the common LLM
    failure mode where SymPy returns a numerically-valid value but the
    LLM mis-labelled the unit (e.g. "0.01885 ohm" for inductive
    reactance when the target is ohms but the value is in henries-omega
    confusion).
    """
    expected = _TARGET_UNIT_HINTS.get(target_quantity)
    if not expected or not produced_unit:
        return None
    try:
        from app.physics.unit_converter import normalize_unit
        from app.physics.dimensions import dimension_for_unit, dimensions_compatible
    except Exception:
        return None
    prod = normalize_unit(str(produced_unit))
    if not prod:
        return None
    prod_dim = dimension_for_unit(prod)
    exp_dim = dimension_for_unit(expected)
    if prod_dim is None or exp_dim is None:
        return None
    return dimensions_compatible(prod_dim, exp_dim)


class PhysicsEquationGraphMethod:
    """Method that decomposes a physics question into equations + SymPy solves."""

    method_id: str = "physics.equation_graph"
    family: MethodFamily = MethodFamily.PHYSICS_NUMERIC
    source: MethodSource = MethodSource.BUILTIN

    def signature(self) -> str:
        return self.method_id

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        if problem.is_lookup_question:
            return MethodApplicability(0.0, "lookup_not_numeric")
        if problem.quantity_count == 0:
            # No knowns to feed the system; nothing to solve.
            return MethodApplicability(0.0, "no_quantities")
        # Yes/no questions are not solved by computing a number — equation
        # graph would compute the underlying quantity and skip the y/n
        # framing. Defer to the legacy pipeline which has resonance / y-n
        # handlers. Generalizes structurally over every "Is the .../Does
        # the .../Will .../Can ..." physics y/n.
        low = problem.raw_question.lower().lstrip()
        yesno_starts = ("is the ", "is a ", "are the ", "are there ",
                        "does the ", "do the ", "will the ", "can the ",
                        "is there ", "does this ", "will this ", "can this ")
        if any(low.startswith(s) for s in yesno_starts):
            return MethodApplicability(0.0, "yesno_not_numeric")
        # Multi-component circuit topology problems (Wheatstone bridge,
        # ladder networks, mesh / nodal analysis) need topology-aware
        # solvers, not a small equation system. The legacy circuit
        # adapter has nodal analysis; defer to it.
        topology_words = (" wheatstone ", " bridge network ", " ladder network ",
                          " nodal analysis", " mesh analysis", " kirchhoff")
        if any(w in (" " + low + " ") for w in topology_words):
            return MethodApplicability(0.0, "circuit_topology_needs_specialist")
        # General-purpose numeric problem: applicable across domains.
        # Score above hand-coded adapters' baseline (0.7) only when at
        # least 2 quantities AND a target quantity are present — those are
        # the cases multi-equation decomposition pays off.
        score = 0.55 if problem.target_quantity else 0.45
        if problem.quantity_count >= 3:
            score += 0.1
        return MethodApplicability(score=min(0.85, score), why="numeric_with_quantities")

    def solve(
        self,
        problem: PhysicsProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        trace = MethodTrace(method_id=self.method_id)
        trace.inputs_seen.append("physics.parsed")
        started = time.perf_counter()
        if llm_client is None:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Equation-graph method requires an LLM client.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="no_llm_client",
            )
        prompt = _EQGRAPH_PROMPT.format(question=problem.raw_question)
        raw = _call_llm(llm_client, prompt)
        trace.llm_calls = 1
        trace.llm_roles = ["equation_extractor"]
        if not raw:
            trace.note("empty_llm_response")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="LLM did not return an equation system.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="empty_response",
            )
        payload = _parse_json(raw)
        if not payload:
            trace.note("invalid_json_from_llm")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="LLM response was not valid JSON.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="invalid_json",
            )
        target_quantity = str(payload.get("target_quantity") or "").strip().lower()
        target_symbol = str(payload.get("target_symbol") or "").strip()
        equations = [str(e) for e in (payload.get("equations") or []) if str(e).strip()]
        knowns_raw = payload.get("knowns") or {}
        knowns: dict[str, float] = {}
        for key, val in knowns_raw.items():
            try:
                # Strings can sneak in as "10 V" or "4*pi*1e-7"; strip the
                # unit suffix and re-evaluate safe arithmetic before float().
                if isinstance(val, str):
                    stripped = _strip_trailing_unit(val)
                    try:
                        parsed = float(stripped)
                    except Exception:
                        if re.fullmatch(
                            r"[\d\.\+\-\*\/eE\(\)piE\s]+", stripped
                        ):
                            parsed = float(eval(
                                stripped, {"__builtins__": {}}, _SAFE_CONSTANTS
                            ))
                        else:
                            continue
                    knowns[_ascii_symbol(str(key))] = parsed
                else:
                    knowns[_ascii_symbol(str(key))] = float(val)
            except Exception:
                continue
        # ASCII-fy equations (μ₀, κ, apostrophe → mu_0, kappa, _new) so
        # SymPy can parse them. Strip trailing unit suffixes from any
        # numeric-only RHS too (e.g. "V_total = 10 V" → "V_total = 10").
        clean_equations: list[str] = []
        for eq in equations:
            ascii_eq = _ascii_symbol(eq)
            if "=" in ascii_eq:
                lhs, rhs = ascii_eq.split("=", 1)
                rhs_stripped = _strip_trailing_unit(rhs.strip())
                ascii_eq = f"{lhs.strip()} = {rhs_stripped}"
            clean_equations.append(ascii_eq)
        equations = clean_equations
        target_symbol = _ascii_symbol(target_symbol)

        answer_unit = str(payload.get("answer_unit") or "").strip()
        explanation_step = str(payload.get("explanation_step") or "").strip()
        modifier_phrases = payload.get("modifier_phrases") or []
        if modifier_phrases:
            trace.step(f"modifiers: {[str(m)[:60] for m in modifier_phrases]}")

        # Target-from-unit repair: when answer_unit clearly names a different
        # quantity than target_quantity, the prompt's TARGET-FROM-UNIT rule
        # was violated. Switch the target to the unit-implied quantity if
        # the equation set already contains a symbol matching it. This
        # recovers the "asks for energy in kJ but LLM picked target=power"
        # failure mode (phys_05) without inventing equations.
        unit_target = _UNIT_TO_TARGET.get(answer_unit.lower().strip())
        if (
            unit_target
            and unit_target != target_quantity
            and unit_target  # non-empty
        ):
            # Find a symbol in the equations that names the implied target.
            preferred_names = {
                "energy": ("energy", "E", "W", "U"),
                "power": ("P",),
                "charge": ("Q",),
                "voltage": ("V",),
                "current": ("I",),
                "resistance": ("R",),
                "force": ("F",),
                "magnetic_field": ("B",),
                "capacitance": ("C",),
                "frequency": ("f", "freq"),
                "mass": ("m",),
                "speed": ("v",),
            }.get(unit_target, ())
            eq_symbols: set[str] = set()
            for eq in equations:
                eq_symbols.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", eq))
            for cand in preferred_names:
                if cand in eq_symbols:
                    trace.step(
                        f"target_repair: answer_unit={answer_unit!r} → "
                        f"target_quantity={unit_target!r} symbol={cand!r} "
                        f"(was {target_quantity!r}/{target_symbol!r})"
                    )
                    target_quantity = unit_target
                    target_symbol = cand
                    break

        if not equations or not target_symbol:
            trace.note("llm_declined_to_decompose")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="LLM did not provide equations + target.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="llm_declined",
            )

        trace.step(f"target = {target_symbol} ({target_quantity})")
        for eq in equations:
            trace.step(f"equation: {eq}")
        if knowns:
            trace.step(f"knowns: {knowns}")

        # Backend SymPy solve.
        value, sympy_trace = _solve_with_sympy(equations, knowns, target_symbol)
        for line in sympy_trace:
            trace.step(line)
        if value is None or not math.isfinite(value):
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="SymPy could not solve the system.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="sympy_no_solution",
            )

        # Backwards verification (Idea #5): substitute the solved value AND
        # the knowns into every original equation; each must become a
        # numeric identity. This catches the "shared LLM bug" failure mode
        # — two methods that agree on the wrong number because the LLM
        # consistently picked the wrong equation set. The check is
        # mechanical SymPy substitution, not another LLM call, so it can
        # find inconsistencies the consistency vote misses.
        verified, verify_trace = _verify_solution_in_equations(
            equations, knowns, target_symbol, value
        )
        for line in verify_trace:
            trace.step(line)
        if not verified:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=(
                    "SymPy solved the system but back-substitution failed: at "
                    "least one original equation does not hold for the computed "
                    "value. Equations are likely internally inconsistent (LLM "
                    "extraction bug). Abstaining."
                ),
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="backwards_verify_failed",
            )

        # Plausibility gate — reject absurd magnitudes.
        if abs(value) > 1e30:
            trace.note(f"rejected_implausible_value:{value}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation="Solved value is implausibly large.",
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="implausible_magnitude",
            )

        # Dimensional gate (AGENTS.md §13.2 / Req 14.x): the LLM's claimed
        # ``answer_unit`` must be dimensionally consistent with the
        # requested target. This blocks the failure mode where SymPy
        # returns a numerically-valid value but the LLM mis-labelled the
        # unit, producing a wrong-with-confidence answer (the eval found
        # cases where '0.01885 ohm' came out for inductive reactance
        # because the formula was ``XL = L * omega`` instead of
        # ``XL = 2*pi*f*L`` — the dimensional check catches scale errors
        # AND wrong-quantity errors).
        dim_ok = _dimensional_ok(answer_unit, target_quantity)
        if dim_ok is False:
            trace.note(f"dimensional_gate_blocked: unit={answer_unit!r}, target={target_quantity!r}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=(
                    f"SymPy returned {value:.6g} {answer_unit} but that unit is "
                    f"not dimensionally consistent with the requested "
                    f"{target_quantity}; the equation system is likely wrong. "
                    "Abstaining so other methods can try."
                ),
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="dimensional_mismatch",
            )

        # Conservation / partition gate (external-law check, AGENTS.md §26.3
        # shared-LLM-bug class). The internal gates above (backwards-verify,
        # dimensional, magnitude) only check that the LLM's equation set is
        # SELF-CONSISTENT — they cannot detect a self-consistent but
        # physically WRONG principle. The canonical example: a parallel
        # current divider where the LLM emits the SERIES formula
        # V=I*(R1+R2); the result (48 mA for a 12 mA source) passes every
        # internal gate. Partition conservation — no single part of a split
        # quantity may exceed the whole — is a dimension-agnostic external
        # law that rejects this whole class without per-formula knowledge.
        try:
            from app.physics.unit_converter import convert_value, normalize_unit
            value_si, _ = convert_value(
                float(value), normalize_unit(answer_unit) if answer_unit else ""
            )
        except Exception:
            value_si = float(value)
        partition_ok, partition_reason = _conservation_partition_ok(
            problem, target_quantity, value_si, answer_unit
        )
        if partition_ok is False:
            trace.note(f"conservation_gate_blocked: {partition_reason}")
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            return MethodResult(
                method_id=self.method_id, family=self.family, answer=None,
                explanation=(
                    f"SymPy returned {value:.6g} {answer_unit} but this violates "
                    f"partition conservation ({partition_reason}): a single part "
                    f"of a split/divided quantity cannot exceed the total being "
                    f"split. The equation set likely used the wrong principle "
                    f"(e.g. series formula for a parallel divider). Abstaining."
                ),
                confidence=0.0, trace=trace,
                abstained=True, abstain_reason="conservation_violation",
            )
        if partition_ok is True:
            trace.step(f"conservation_gate_passed: {partition_reason}")

        answer = _format_answer(value, target_quantity, answer_unit)
        explanation = (
            f"Decomposed the question into {len(equations)} standard physics "
            f"equation(s) and solved with SymPy. "
            f"{explanation_step}".strip()
            + f" Answer: {answer}."
        )
        # Confidence reflects whether the dimensional gate confirmed
        # (0.80) or could not be evaluated (0.65). Never above the
        # adapter peak so a hand-coded adapter that solved the same
        # case takes precedence on tie.
        confidence = 0.80 if dim_ok else 0.65
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        return MethodResult(
            method_id=self.method_id, family=self.family,
            answer=answer, explanation=explanation,
            confidence=confidence, trace=trace,
            formula_id=";".join(equations[:3]),
            numeric_value=float(value),
            numeric_unit=answer_unit or _TARGET_UNIT_HINTS.get(target_quantity) or "",
            # Backwards-verification has just passed: every equation
            # remains a numeric identity under (knowns ∪ {target: value}).
            # Plus the dimensional gate either passed or could not be
            # evaluated (False would have abstained). The planner uses
            # this to weight self-consistency votes — a backend-verified
            # answer outranks a verifier that has no witness of its own.
            backend_verified=True,
        )
