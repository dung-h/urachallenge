"""Monotonic relation registry and SymPy partial-derivative sign analysis.

This module is the deterministic *authority* for the qualitative
direction-of-change of one physics quantity with respect to another (Task 20.2,
Requirement 14.1). It uses SymPy to take partial derivatives of canonical
formulas drawn from :mod:`app.physics.formulas` and reports the **sign** of the
derivative on the physical positive-real domain.

Design contract (AGENTS.md §20, design.md "Qualitative_Reasoner sub-component"):

* The registry is keyed by canonical formula identifiers; it is **never** keyed
  by question text. There is no per-question override.
* :func:`monotonic_sign` is a pure function: given a canonical SymPy expression
  for the target, the input symbol whose sign we want, and the symbols held
  constant, it returns one of ``"+"``, ``"-"``, ``"0"`` or ``"ambiguous"``.
* :func:`select_formula` picks the *simplest* canonical formula in this module's
  registry that relates a target quantity to an input quantity, returning
  ``None`` if no formula applies. The caller (``qualitative_reasoner``) is
  responsible for translating between the natural-language quantity name (e.g.
  "power", "current", "resistance") and the canonical SymPy symbol.
* The whole module operates on **magnitudes**: every free symbol carries the
  ``positive=True`` SymPy assumption so that on the physical positive domain the
  sign of any partial derivative of a smooth registry formula is constant. If
  it is not, we return ``"ambiguous"`` and the caller abstains.

This module is intentionally independent of any LLM surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import sympy as sp


__all__ = [
    "Sign",
    "RegistryFormula",
    "FORMULA_REGISTRY",
    "monotonic_sign",
    "select_formula",
    "list_formula_ids",
]


# ---------------------------------------------------------------------------
# Sign type and a helper for converting SymPy results into our string sign.
# ---------------------------------------------------------------------------

Sign = Literal["+", "-", "0", "ambiguous"]


def _classify_sign(expr: sp.Expr) -> Sign:
    """Classify a SymPy expression's sign on the positive-real domain.

    The expression is expected to be a partial derivative of a smooth registry
    formula, evaluated symbolically with all free symbols carrying the
    ``positive=True`` assumption. We rely on SymPy's ``is_positive`` /
    ``is_negative`` / ``is_zero`` queries first, then fall back to refining the
    expression under positive assumptions before declaring ambiguity.
    """

    simplified = sp.simplify(expr)

    if simplified == 0 or simplified.is_zero:
        return "0"
    if simplified.is_positive:
        return "+"
    if simplified.is_negative:
        return "-"

    # Re-declare positivity for any remaining free symbols and try again.
    positive_subs = {
        s: sp.Symbol(s.name, positive=True)
        for s in simplified.free_symbols
        if not s.is_positive
    }
    refined = sp.simplify(simplified.xreplace(positive_subs)) if positive_subs else simplified

    if refined.is_positive:
        return "+"
    if refined.is_negative:
        return "-"
    if refined == 0 or refined.is_zero:
        return "0"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Canonical formula registry (mirrors AGENTS.md §14 and ``app/physics/formulas.py``).
# Keys are stable formula identifiers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryFormula:
    """A canonical physics formula for monotonic-sign analysis.

    Attributes
    ----------
    formula_id:
        Stable identifier such as ``"P=V^2/R"``.
    expression_text:
        Human-readable formula expression (used by explanation worker).
    target:
        Canonical name of the target quantity (lowercase, e.g. ``"power"``).
    expression:
        SymPy expression giving the target as a function of named symbols.
    quantities:
        Mapping from canonical quantity name (e.g. ``"resistance"``) to its
        SymPy symbol (e.g. ``R``). All symbols carry ``positive=True``.
    """

    formula_id: str
    expression_text: str
    target: str
    expression: sp.Expr
    quantities: dict[str, sp.Symbol]


def _pos(name: str) -> sp.Symbol:
    """Declares a symbol positive."""
    return sp.Symbol(name, positive=True)


# Symbols (positive reals).
_V = _pos("V")
_I = _pos("I")
_R = _pos("R")
_P = _pos("P")
_C = _pos("C")
_Q = _pos("Q")
_E = _pos("E")  # generic energy symbol (capacitor energy, inductor energy)
_q1 = _pos("q1")
_q2 = _pos("q2")
_q = _pos("q")
_r = _pos("r")
_L = _pos("L")
_k = sp.Symbol("k", positive=True)


def _build_registry() -> dict[str, RegistryFormula]:
    """Return the canonical (formula_id -> RegistryFormula) registry.

    Only formulas explicitly required by Task 20.2 are listed. Each entry is
    independently exact arithmetic in SymPy: no LLM input.
    """

    entries: list[RegistryFormula] = [
        RegistryFormula(
            formula_id="V=I*R",
            expression_text="V = I * R",
            target="voltage",
            expression=_I * _R,
            quantities={"current": _I, "resistance": _R},
        ),
        RegistryFormula(
            formula_id="I=V/R",
            expression_text="I = V / R",
            target="current",
            expression=_V / _R,
            quantities={"voltage": _V, "resistance": _R},
        ),
        RegistryFormula(
            formula_id="R=V/I",
            expression_text="R = V / I",
            target="resistance",
            expression=_V / _I,
            quantities={"voltage": _V, "current": _I},
        ),
        RegistryFormula(
            formula_id="P=V*I",
            expression_text="P = V * I",
            target="power",
            expression=_V * _I,
            quantities={"voltage": _V, "current": _I},
        ),
        RegistryFormula(
            formula_id="P=I^2*R",
            expression_text="P = I^2 * R",
            target="power",
            expression=_I**2 * _R,
            quantities={"current": _I, "resistance": _R},
        ),
        RegistryFormula(
            formula_id="P=V^2/R",
            expression_text="P = V^2 / R",
            target="power",
            expression=_V**2 / _R,
            quantities={"voltage": _V, "resistance": _R},
        ),
        RegistryFormula(
            formula_id="Rseries=R1+R2",
            expression_text="R_total = R1 + R2",
            target="total_resistance",
            expression=_pos("R1") + _pos("R2"),
            quantities={
                "resistance_1": _pos("R1"),
                "resistance_2": _pos("R2"),
            },
        ),
        RegistryFormula(
            formula_id="Rparallel=R1*R2/(R1+R2)",
            expression_text="R_total = R1 * R2 / (R1 + R2)",
            target="total_resistance",
            expression=(_pos("R1") * _pos("R2")) / (_pos("R1") + _pos("R2")),
            quantities={
                "resistance_1": _pos("R1"),
                "resistance_2": _pos("R2"),
            },
        ),
        RegistryFormula(
            formula_id="Q=C*V",
            expression_text="Q = C * V",
            target="charge",
            expression=_C * _V,
            quantities={"capacitance": _C, "voltage": _V},
        ),
        RegistryFormula(
            formula_id="V=Q/C",
            expression_text="V = Q / C",
            target="voltage",
            expression=_Q / _C,
            quantities={"charge": _Q, "capacitance": _C},
        ),
        RegistryFormula(
            formula_id="Ecap=0.5*C*V^2",
            expression_text="E = 0.5 * C * V^2",
            target="capacitor_energy",
            expression=sp.Rational(1, 2) * _C * _V**2,
            quantities={"capacitance": _C, "voltage": _V},
        ),
        RegistryFormula(
            formula_id="Ecap_from_charge=Q^2/(2C)",
            expression_text="E = Q^2 / (2 * C)",
            target="capacitor_energy",
            expression=_Q**2 / (2 * _C),
            quantities={"charge": _Q, "capacitance": _C},
        ),
        RegistryFormula(
            formula_id="F=k*q1*q2/r^2",
            expression_text="F = k * q1 * q2 / r^2",
            target="force",
            expression=_k * _q1 * _q2 / _r**2,
            quantities={"charge_1": _q1, "charge_2": _q2, "distance": _r},
        ),
        RegistryFormula(
            formula_id="E=k*q/r^2",
            expression_text="E = k * q / r^2",
            target="electric_field",
            expression=_k * _q / _r**2,
            quantities={"charge": _q, "distance": _r},
        ),
        RegistryFormula(
            formula_id="U=q*V",
            expression_text="U = q * V",
            target="potential_energy",
            expression=_q * _V,
            quantities={"charge": _q, "voltage": _V},
        ),
        RegistryFormula(
            formula_id="Eind=0.5*L*I^2",
            expression_text="E = 0.5 * L * I^2",
            target="inductor_energy",
            expression=sp.Rational(1, 2) * _L * _I**2,
            quantities={"inductance": _L, "current": _I},
        ),
    ]

    return {entry.formula_id: entry for entry in entries}


FORMULA_REGISTRY: dict[str, RegistryFormula] = _build_registry()


def list_formula_ids() -> list[str]:
    """Return the canonical formula identifiers (deterministic ordering).

    Returns:
        A list of formula identifier strings.
    """
    return list(FORMULA_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Sign analysis and formula selection.
# ---------------------------------------------------------------------------


def monotonic_sign(
    target_expr: sp.Expr,
    input_symbol: sp.Symbol,
    fixed_symbols: Iterable[sp.Symbol] = (),
) -> Sign:
    """Compute the sign of ``∂target_expr / ∂input_symbol`` on the positive domain.

    Args:
        target_expr: SymPy expression giving the target quantity in terms of named symbols.
        input_symbol: The symbol with respect to which we differentiate.
        fixed_symbols: Symbols that should be treated as positive constants (held fixed).

    Returns:
        "+" if strictly positive, "-" if strictly negative, "0" if zero, and "ambiguous" otherwise.
    """

    if input_symbol not in target_expr.free_symbols:
        # Target does not depend on the input at all: derivative is identically 0.
        return "0"

    # All free symbols other than the input are held fixed; ensure positivity.
    fixed = {sym for sym in fixed_symbols if sym is not input_symbol}
    fixed |= {sym for sym in target_expr.free_symbols if sym is not input_symbol}

    positive_subs = {
        sym: sp.Symbol(sym.name, positive=True) for sym in fixed if not sym.is_positive
    }
    expr_pos = target_expr.xreplace(positive_subs)
    input_pos = (
        input_symbol
        if input_symbol.is_positive
        else sp.Symbol(input_symbol.name, positive=True)
    )
    expr_pos = expr_pos.xreplace({input_symbol: input_pos})

    derivative = sp.diff(expr_pos, input_pos)
    return _classify_sign(derivative)


def select_formula(
    target_quantity: str,
    input_quantity: str,
) -> RegistryFormula | None:
    """Pick the simplest registry formula relating ``target`` and ``input``.

    Args:
        target_quantity: Lowercase canonical name of target quantity (e.g. "voltage").
        input_quantity: Lowercase canonical name of input quantity (e.g. "resistance").

    Returns:
        The matched RegistryFormula, or None if no formula applies.
    """

    target_quantity = target_quantity.strip().lower()
    input_quantity = input_quantity.strip().lower()

    candidates: list[RegistryFormula] = []
    for entry in FORMULA_REGISTRY.values():
        if entry.target != target_quantity:
            continue
        if input_quantity not in entry.quantities:
            continue
        candidates.append(entry)

    if not candidates:
        return None

    candidates.sort(key=lambda f: (len(f.expression.free_symbols), f.formula_id))
    return candidates[0]

