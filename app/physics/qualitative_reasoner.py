"""Qualitative monotonic-reasoning core (Req 14.1, 14.2).

This module is the deterministic backend authority for "direction-of-change"
physics questions: given a formula relating a target quantity to an input
quantity (with all other inputs held constant), it returns the sign of the
target's response to the input.

Design notes (from `.kiro/specs/exact-challenge-optimization/design.md`,
Qualitative_Reasoner sub-component):

* The registry `MONOTONIC_REGISTRY` is the fast path. Entries are pre-computed
  by hand from the formulas in :mod:`app.physics.formulas`, so a registry
  lookup is `O(1)` and the signs are auditable.
* On registry miss, :func:`derive_sign` performs symbolic partial-derivative
  sign analysis on the formula's positive-real domain via SymPy. If the sign
  is not constant on the relevant domain, ``None`` is returned and the
  caller (the solver) must yield ``unknown`` (Req 14.2 / Req 11.1).
* The :func:`_normalize_qualitative_label` function is a *general* phrasing
  -> label rule set (regex/keyword classes). It is NEVER a per-question
  text -> answer mapping (AGENTS.md Β§20.1, Req 11.5, Req 13.1).
* Signs come from formula structure only β€” never from LLM output or from the
  question text (root-cause discipline, AGENTS.md Β§20).

The companion module :mod:`app.physics.qualitative_parser` extracts the
qualitative-question shape (input_var, change_direction, target_var,
all-else-constant guard) so the reasoner can be invoked safely. This module
does not depend on the parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import sympy as sp


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonotonicRelation:
    """Direction of change of ``target_var`` w.r.t. ``input_var`` for a formula.

    The ``sign`` is the sign of the partial derivative
    ``βˆ‚target_var / βˆ‚input_var`` on the formula's physical (positive-real)
    domain, holding every other input constant:

    * ``+1`` β€” target strictly increases when input increases.
    * ``-1`` β€” target strictly decreases when input increases.
    * ``0``  β€” target is independent of the input on the relevant domain.

    ``derivation`` is a short human-readable justification stored alongside
    the sign so the explanation worker can ground qualitative explanations
    in the recorded trace (Req 10.3 / Req 14.4).
    """

    formula_id: str
    input_var: str
    target_var: str
    sign: int  # +1 | -1 | 0
    derivation: str

    def __post_init__(self) -> None:
        if self.sign not in (-1, 0, 1):
            raise ValueError(
                f"MonotonicRelation.sign must be in {{-1, 0, 1}}, got {self.sign!r}"
            )


CanonicalLabel = Literal["increases", "decreases", "unchanged", "unknown"]


# ---------------------------------------------------------------------------
# Hand-computed registry of monotonic relationships
# ---------------------------------------------------------------------------
# Every entry below corresponds to one of the 13 named formulas listed in the
# AGENTS.md "Formula Registry Initial Scope" and surfaced in
# `app/physics/formulas.py`:
#
#   1.  V = I * R                           (ohms_law_v_ir)
#   2.  P = V * I                           (power_p_vi)
#   3.  P = I^2 * R                         (power_p_i2r)
#   4.  P = V^2 / R                         (power_p_v2r)
#   5.  R_total = R1 + R2 + ...             (series_resistance)
#   6.  1/R_total = sum(1/R_i)              (parallel_resistance)
#   7.  Q = C * V                           (capacitor_charge_q_cv)
#   8.  E = 0.5 * C * V^2                   (capacitor_energy_e_half_cv2)
#   9.  1/C_total = sum(1/C_i)              (series_capacitance)
#  10.  C_total = C1 + C2 + ...             (parallel_capacitance)
#  11.  F = k * q1 * q2 / r^2               (coulomb_force)
#  12.  E = k * q / r^2                     (electric_field_kq_r2)
#  13.  U = q * V                           (potential_energy_u_qv)
#
# For variadic formulas (series/parallel) we use a generic input-name
# ``R_i`` / ``C_i`` to mean "any one of the bank's components"; the
# physical domain is positive resistances / capacitances, so the sign of
# βˆ‚R_total/βˆ‚R_i is the same as the sign for any concrete component.

_REGISTRY_SOURCES: tuple[tuple[str, str, str, int, str], ...] = (
    # ---- Ohm's law: V = I * R ----
    ("ohms_law_v_ir", "I", "V", +1, "V = I*R; dV/dI = R > 0 on R > 0"),
    ("ohms_law_v_ir", "R", "V", +1, "V = I*R; dV/dR = I > 0 on I > 0"),
    # ---- Power: P = V * I ----
    ("power_p_vi", "V", "P", +1, "P = V*I; dP/dV = I > 0 on I > 0"),
    ("power_p_vi", "I", "P", +1, "P = V*I; dP/dI = V > 0 on V > 0"),
    # ---- Power: P = I^2 * R ----
    ("power_p_i2r", "I", "P", +1, "P = I^2*R; dP/dI = 2*I*R > 0 on I,R > 0"),
    ("power_p_i2r", "R", "P", +1, "P = I^2*R; dP/dR = I^2 > 0 on I > 0"),
    # ---- Power: P = V^2 / R ----
    ("power_p_v2r", "V", "P", +1, "P = V^2/R; dP/dV = 2*V/R > 0 on V,R > 0"),
    ("power_p_v2r", "R", "P", -1, "P = V^2/R; dP/dR = -V^2/R^2 < 0 on V,R > 0"),
    # ---- Series resistance: R_total = R1 + R2 + ... ----
    (
        "series_resistance",
        "R_i",
        "R_total",
        +1,
        "R_total = sum(R_i); dR_total/dR_i = 1 > 0",
    ),
    # ---- Parallel resistance: 1/R_total = sum(1/R_i) ----
    (
        "parallel_resistance",
        "R_i",
        "R_total",
        +1,
        "R_total = 1/sum(1/R_i); dR_total/dR_i = (R_total/R_i)^2 > 0",
    ),
    # ---- Capacitor charge: Q = C * V ----
    ("capacitor_charge_q_cv", "C", "Q", +1, "Q = C*V; dQ/dC = V > 0 on V > 0"),
    ("capacitor_charge_q_cv", "V", "Q", +1, "Q = C*V; dQ/dV = C > 0 on C > 0"),
    # ---- Capacitor energy: E = 0.5 * C * V^2 ----
    (
        "capacitor_energy_e_half_cv2",
        "C",
        "E",
        +1,
        "E = (1/2)*C*V^2; dE/dC = V^2/2 > 0 on V > 0",
    ),
    (
        "capacitor_energy_e_half_cv2",
        "V",
        "E",
        +1,
        "E = (1/2)*C*V^2; dE/dV = C*V > 0 on C,V > 0",
    ),
    # ---- Series capacitance: 1/C_total = sum(1/C_i) ----
    (
        "series_capacitance",
        "C_i",
        "C_total",
        +1,
        "C_total = 1/sum(1/C_i); dC_total/dC_i = (C_total/C_i)^2 > 0",
    ),
    # ---- Parallel capacitance: C_total = C1 + C2 + ... ----
    (
        "parallel_capacitance",
        "C_i",
        "C_total",
        +1,
        "C_total = sum(C_i); dC_total/dC_i = 1 > 0",
    ),
    # ---- Coulomb force: F = k * q1 * q2 / r^2 (positive magnitudes) ----
    (
        "coulomb_force",
        "q1",
        "F",
        +1,
        "F = k*q1*q2/r^2; dF/dq1 = k*q2/r^2 > 0 on q2,r > 0",
    ),
    (
        "coulomb_force",
        "q2",
        "F",
        +1,
        "F = k*q1*q2/r^2; dF/dq2 = k*q1/r^2 > 0 on q1,r > 0",
    ),
    (
        "coulomb_force",
        "r",
        "F",
        -1,
        "F = k*q1*q2/r^2; dF/dr = -2*k*q1*q2/r^3 < 0 on q1,q2,r > 0",
    ),
    # ---- Electric field of a point charge: E = k*q/r^2 ----
    (
        "electric_field_kq_r2",
        "q",
        "E",
        +1,
        "E = k*q/r^2; dE/dq = k/r^2 > 0 on r > 0",
    ),
    (
        "electric_field_kq_r2",
        "r",
        "E",
        -1,
        "E = k*q/r^2; dE/dr = -2*k*q/r^3 < 0 on q,r > 0",
    ),
    # ---- Potential energy: U = q * V ----
    ("potential_energy_u_qv", "q", "U", +1, "U = q*V; dU/dq = V > 0 on V > 0"),
    ("potential_energy_u_qv", "V", "U", +1, "U = q*V; dU/dV = q > 0 on q > 0"),
)


def _build_registry() -> dict[tuple[str, str, str], MonotonicRelation]:
    """Constructs the registry of MonotonicRelation objects from the sources."""
    registry: dict[tuple[str, str, str], MonotonicRelation] = {}
    for formula_id, input_var, target_var, sign, derivation in _REGISTRY_SOURCES:
        key = (formula_id, input_var, target_var)
        if key in registry:
            raise ValueError(f"duplicate registry entry for {key}")
        registry[key] = MonotonicRelation(
            formula_id=formula_id,
            input_var=input_var,
            target_var=target_var,
            sign=sign,
            derivation=derivation,
        )
    return registry


MONOTONIC_REGISTRY: dict[tuple[str, str, str], MonotonicRelation] = _build_registry()


# ---------------------------------------------------------------------------
# SymPy expressions for fallback partial-derivative sign analysis
# ---------------------------------------------------------------------------
# Each entry maps a formula_id to ``(target_var_name, expression)``. Symbols
# are declared positive so that SymPy can prove sign properties on the
# physical domain (`is_positive` / `is_negative`).
#
# These cover (a) every registry formula, so the property test can validate
# the registry by symbolic differentiation, and (b) several formulas that
# are *not* in the registry, so :func:`derive_sign` exercises the SymPy
# fallback path.


def _S(name: str) -> sp.Symbol:
    return sp.Symbol(name, positive=True)


_SYMPY_FORMULAS: dict[str, tuple[str, sp.Expr]] = {
    # --- registry formulas ---
    "ohms_law_v_ir": ("V", _S("I") * _S("R")),
    "power_p_vi": ("P", _S("V") * _S("I")),
    "power_p_i2r": ("P", _S("I") ** 2 * _S("R")),
    "power_p_v2r": ("P", _S("V") ** 2 / _S("R")),
    "series_resistance": ("R_total", _S("R1") + _S("R2")),
    "parallel_resistance": (
        "R_total",
        1 / (1 / _S("R1") + 1 / _S("R2")),
    ),
    "capacitor_charge_q_cv": ("Q", _S("C") * _S("V")),
    "capacitor_energy_e_half_cv2": (
        "E",
        sp.Rational(1, 2) * _S("C") * _S("V") ** 2,
    ),
    "series_capacitance": (
        "C_total",
        1 / (1 / _S("C1") + 1 / _S("C2")),
    ),
    "parallel_capacitance": ("C_total", _S("C1") + _S("C2")),
    "coulomb_force": (
        "F",
        _S("k") * _S("q1") * _S("q2") / _S("r") ** 2,
    ),
    "electric_field_kq_r2": ("E", _S("k") * _S("q") / _S("r") ** 2),
    "potential_energy_u_qv": ("U", _S("q") * _S("V")),
    # --- additional formulas exercised through SymPy fallback only ---
    # X_L = 2*pi*f*L
    "inductive_reactance": ("X_L", 2 * sp.pi * _S("f") * _S("L")),
    # X_C = 1 / (2*pi*f*C)
    "capacitive_reactance": ("X_C", 1 / (2 * sp.pi * _S("f") * _S("C"))),
    # E = 0.5 * L * I^2
    "inductor_energy": ("E", sp.Rational(1, 2) * _S("L") * _S("I") ** 2),
    # B = mu0 * (N/l) * I  (treat mu0 as a positive symbol; sign is preserved)
    "solenoid_B": (
        "B",
        _S("mu0") * (_S("N") / _S("l")) * _S("I"),
    ),
    # f = v / wavelength
    "wave_frequency": ("f", _S("v") / _S("wavelength")),
}


# Aliases for variadic input names ("R_i" -> "R1", "C_i" -> "C1") so that
# `derive_sign` and the property test can use the registry's qualitative
# input-var name on the SymPy expression's concrete-symbol form.
_INPUT_VAR_ALIASES: dict[tuple[str, str], str] = {
    ("series_resistance", "R_i"): "R1",
    ("parallel_resistance", "R_i"): "R1",
    ("series_capacitance", "C_i"): "C1",
    ("parallel_capacitance", "C_i"): "C1",
}


def _resolve_sym(expr: sp.Expr, name: str) -> sp.Symbol | None:
    """Finds a symbol by name in the expression's free symbols."""
    for s in expr.free_symbols:
        if getattr(s, "name", None) == name:
            return s  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Public API: derive_sign and lookup_formula
# ---------------------------------------------------------------------------


def derive_sign(
    formula_id: str, input_var: str, target_var: str
) -> MonotonicRelation | None:
    """Derive the sign of ∂target_var/∂input_var for ``formula_id``.

    Attempts a registry lookup first, then falls back to symbolic partial-derivative
    sign analysis using SymPy on the positive-real domain.

    Args:
        formula_id: The formula ID string.
        input_var: The input variable symbol.
        target_var: The target variable symbol.

    Returns:
        A MonotonicRelation containing the sign and derivation, or None.
    """

    key = (formula_id, input_var, target_var)
    cached = MONOTONIC_REGISTRY.get(key)
    if cached is not None:
        return cached

    expr_info = _SYMPY_FORMULAS.get(formula_id)
    if expr_info is None:
        return None
    target_name, expr = expr_info
    if target_name != target_var:
        return None

    sym_name = _INPUT_VAR_ALIASES.get((formula_id, input_var), input_var)
    target_sym = _resolve_sym(expr, sym_name)
    if target_sym is None:
        return None

    deriv = sp.simplify(sp.diff(expr, target_sym))

    if deriv.is_zero:
        sign = 0
    elif deriv.is_positive:
        sign = +1
    elif deriv.is_negative:
        sign = -1
    else:
        # Sign is not constant on the positive-real domain.
        return None

    return MonotonicRelation(
        formula_id=formula_id,
        input_var=input_var,
        target_var=target_var,
        sign=sign,
        derivation=f"d{target_var}/d{input_var} = {deriv}",
    )


# Pre-computed (target_var, input_var) -> formula_id index. The first
# matching registry entry wins; iteration order of `_REGISTRY_SOURCES`
# determines the canonical formula picked when multiple formulas share
# the same (target, input) pair.
def _build_lookup_index() -> dict[tuple[str, str], str]:
    """Constructs the lookup index mapping (target_var, input_var) to formula_id."""
    index: dict[tuple[str, str], str] = {}
    for formula_id, input_var, target_var, _sign, _deriv in _REGISTRY_SOURCES:
        index.setdefault((target_var, input_var), formula_id)
    return index


_LOOKUP_INDEX: dict[tuple[str, str], str] = _build_lookup_index()


def lookup_formula(target_var: str, input_var: str) -> str | None:
    """Finds a registry formula_id containing the monotonic relation.

    Args:
        target_var: The target variable symbol.
        input_var: The input variable symbol.

    Returns:
        The matching formula_id, or None if not found.
    """

    return _LOOKUP_INDEX.get((target_var, input_var))



# ---------------------------------------------------------------------------
# Canonical-label normalizer (Req 14.3, AGENTS.md Β§20.1)
# ---------------------------------------------------------------------------
# This module re-exports the canonical-label normalizer owned by
# :mod:`app.physics.qualitative_normalizer` (Task 20.3) so the qualitative
# reasoner consumes the same rule set as the scorer and the solver branch.
# The normalizer is GENERAL phrasing -> label rules; it is NEVER a
# per-question text -> answer mapping (AGENTS.md Β§20.1, Req 11.5, Req 13.1).
#
# The reasoner's downstream API treats only ``increases``,``decreases``,
# ``unchanged`` as accept labels and abstains on everything else (Req 14.4).
# To preserve that contract while still delegating the rule set to the
# central normalizer, we collapse the magnitude labels (``doubled`` β†’
# ``increases``, ``halved`` β†’ ``decreases``) before returning.
from app.physics.qualitative_normalizer import (
    DIRECTIONAL_EQUIVALENCE as _DIRECTIONAL_EQUIVALENCE,
    normalize_qualitative_label as _normalize_with_magnitudes,
)


def _normalize_qualitative_label(text: str) -> CanonicalLabel:
    """Map a free-text direction phrase to a canonical reasoner label.

    Returns one of ``"increases"``, ``"decreases"``, ``"unchanged"``, or
    ``"unknown"``. Delegates to
    :func:`app.physics.qualitative_normalizer.normalize_qualitative_label`
    and then collapses any magnitude label (``doubled``, ``halved``) onto
    its directional equivalent (Req 14.3). Extremum labels (``maximum`` /
    ``minimum``) collapse to ``"unknown"`` in the reasoner because the
    sign-based reasoner cannot produce them.
    """

    raw = _normalize_with_magnitudes(text)
    if raw in {"increases", "decreases", "unchanged", "unknown"}:
        return raw  # type: ignore[return-value]
    if raw in _DIRECTIONAL_EQUIVALENCE:
        return _DIRECTIONAL_EQUIVALENCE[raw]  # type: ignore[return-value]
    # Extrema and any future canonical labels not in the reasoner's contract
    # collapse to ``unknown`` so the reasoner abstains rather than emitting
    # an unsupported label.
    return "unknown"


__all__ = [
    "MonotonicRelation",
    "MONOTONIC_REGISTRY",
    "derive_sign",
    "lookup_formula",
    "_normalize_qualitative_label",
]
