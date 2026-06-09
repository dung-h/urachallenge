"""Backend-computed confidence scoring (Req 9).

This module is the single home for the ``confidence`` field. Confidence is
computed deterministically from measured backend pipeline signals — never
invented by an LLM (Req 9.2). The scorer is a pure function: identical signal
values always produce identical output (Req 9.6), and every input signal value
is recorded alongside the resulting confidence (Req 9.5).

Mapping (AGENTS.md §16, Req 9.3/9.4):

* deterministic solver success + valid units + no ambiguity + verified
  → ``>= 0.90``
* solver success with parsing ambiguity → ``~0.80``
* a verified answer that relied on an LLM proposal → strictly ``< 0.90``
* ``unknown`` → ``<= 0.30`` and strictly below any verified answer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ConfidenceSignals", "compute_confidence"]


@dataclass(frozen=True)
class ConfidenceSignals:
    """Measured backend signals consumed by :func:`compute_confidence`.

    CRITICAL (Req 9.2): there is intentionally NO field for an LLM-produced
    confidence value. The LLM never contributes a confidence number, directly
    or as an input signal. ``llm_fallback_used`` records only *whether* an LLM
    proposal participated in producing the answer, not any LLM-generated score.

    Physics-only signals (``formula_matched``, ``unit_valid``) accept ``None``
    when the case is not a physics case so the same structure serves both task
    families.
    """

    parser_success: bool
    formula_matched: bool  # physics
    solver_success: bool  # answer != "unknown"
    unit_valid: bool | None  # physics (None when not applicable)
    answer_verified: bool  # independent backend agreement reached
    premise_selection_score: float  # logic
    llm_fallback_used: bool
    json_valid: bool
    ambiguity_detected: bool


# Confidence anchor values. Kept as named constants so the deterministic
# mapping is auditable and the strict orderings required by Req 9.3/9.4 are
# obvious from the source.
_CONF_DETERMINISTIC_VERIFIED = 0.95  # deterministic success, valid units, no ambiguity
_CONF_DETERMINISTIC_AMBIGUOUS = 0.80  # solver success but parsing ambiguity
_CONF_LLM_VERIFIED = 0.70  # verified answer relying on an LLM proposal (< 0.90)
_CONF_LLM_AMBIGUOUS = 0.60  # LLM-dependent verified answer with ambiguity
_CONF_WEAK = 0.40  # an answer produced without backend verification
_CONF_UNKNOWN = 0.20  # abstention (<= 0.30, below any verified answer)


def _is_unknown(answer: str) -> bool:
    """True when the answer is an abstention."""
    return str(answer).strip().lower() == "unknown"


def compute_confidence(signals: ConfidenceSignals, answer: str) -> tuple[float, dict[str, Any]]:
    """Compute confidence from backend signals (pure, deterministic).

    Args:
        signals: Measured backend signals.
        answer: The generated answer string.

    Returns:
        A tuple of (confidence_score, recorded_signal_values).
    """

    # --- Abstention dominates: unknown is always <= 0.30 (Req 9.4, 11.2). ---
    if _is_unknown(answer):
        confidence = _CONF_UNKNOWN
    elif signals.answer_verified:
        # An accepted answer with independent backend agreement.
        if signals.llm_fallback_used:
            # Verified, but the answer relied on an LLM proposal: strictly
            # below the deterministic-success band (Req 9.3).
            confidence = (
                _CONF_LLM_AMBIGUOUS if signals.ambiguity_detected else _CONF_LLM_VERIFIED
            )
        else:
            # Deterministic, backend-verified answer.
            units_ok = signals.unit_valid is not False
            clean = (
                signals.parser_success
                and signals.solver_success
                and units_ok
                and not signals.ambiguity_detected
            )
            if clean:
                # Deterministic success + valid units + no ambiguity (Req 9.3).
                confidence = _CONF_DETERMINISTIC_VERIFIED
            else:
                # Solver success but parsing ambiguity / soft signal (~0.80).
                confidence = _CONF_DETERMINISTIC_AMBIGUOUS
    else:
        # A non-unknown answer that was not backend-verified. This is weaker
        # than any verified answer but still above an abstention.
        confidence = _CONF_WEAK

    confidence = max(0.0, min(1.0, float(confidence)))

    recorded: dict[str, Any] = asdict(signals)
    recorded["answer"] = str(answer)
    recorded["confidence"] = confidence
    return confidence, recorded
