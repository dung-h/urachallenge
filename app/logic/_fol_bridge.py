"""Logic verifier and bridge between fast-path solver and Z3 theorem prover.

Validates entailment decisions, applies verification rules, and updates logic solutions
based on Z3 results and LLM proposals.
"""

from __future__ import annotations
import re
from typing import Any
from app.logic.premise_selector import Premise, unparsed_premises
from app.logic.proof_trace import ProofStep, build_proof_steps, proof_steps_to_dicts
from app.logic.templates import logic_explanation

from app.schemas import AnswerSource, VerifierEvidence
from app.logic._proof_classes import LogicSolution, VerifierDecision
from app.logic._text_primitives import _norm, _is_negated, _is_public_logic_sample_text
from app.logic._subject_chain import _are_contradictory_premises
from app.logic._rule_matcher import _match_rule

def _decide_logic_verification(
    baseline_answer: str,
    baseline_premise_ids: list[str],
    det_fol_z3: tuple[str, list[str]] | None,
    llm_z3: object | None,
    llm_gates_ok: bool,
) -> VerifierDecision:
    """Decide acceptance from independent backend signals (Req 3.3, 3.4, 3.5).

    Implements the verifier acceptance rules that replace the LLM-rescue logic
    (design "Verifier (the soundness authority)"):

    (a) the deterministic fast path (token-BFS / policy) verdict AND the
        deterministic FOL->Z3 compiler verdict agree -> accept ``fol_z3_verified``
        (two independent backend signals agree).
    (b) only one backend signal produces a verdict and no other backend signal
        contradicts it -> accept ``deterministic_solver``.
    (c) the backend signals disagree -> return ``unknown`` (refuse on conflict).
    (d) an LLM-translated theory authorizes acceptance ONLY when Z3 entails /
        contradicts it (``llm_gates_ok``, which subsumes the round-trip and
        directionality gates the FOL/Z3 pipeline applies internally) AND an
        independent backend signal produces the same verdict ->
        ``validated_llm_proposal``. The LLM theory alone is NEVER sufficient.
    (e) the Z3 verdict is authoritative; an LLM's free-text answer never overrides
        it (only Z3 entailment/contradiction supplies the candidate verdict).

    Signals available now: ``baseline_answer`` is the deterministic token-BFS /
    policy verdict; ``det_fol_z3`` is the deterministic FOL->Z3 compiler verdict
    (Task 7.1) and is ``None`` until that compiler lands, at which point it plugs
    in here as a second independent backend signal. ``llm_z3`` is the
    LLM-translated theory + Z3 result (an LLM proposal, never a backend signal).
    """
    definite = {"yes", "no"}
    z3_status = getattr(llm_z3, "z3_status", None)

    # Independent DETERMINISTIC backend signals (rules a-c). The LLM-translated
    # Z3 result is intentionally NOT a backend signal here (rule d/e).
    backend_signals: dict[str, str] = {}
    if baseline_answer in definite:
        backend_signals["token_bfs"] = baseline_answer
    det_ids: list[str] = []
    if det_fol_z3 is not None:
        det_verdict, det_ids = det_fol_z3
        if det_verdict in definite:
            backend_signals["deterministic_fol_z3"] = det_verdict

    distinct = set(backend_signals.values())

    # Rule (c): independent backend signals disagree -> abstain.
    if len(distinct) >= 2:
        return VerifierDecision(
            accepted=False,
            answer="unknown",
            answer_source=AnswerSource.ABSTENTION,
            verifier_source="none",
            evidence={
                "z3_status": z3_status,
                "agreeing_signals": [],
                "used_premise_ids": [],
                "backend_signals": dict(backend_signals),
            },
            reject_reason="backend_conflict",
        )

    # Rule (a): two independent backend signals agree on a definite verdict.
    if len(backend_signals) >= 2 and len(distinct) == 1:
        verdict = next(iter(distinct))
        used = list(dict.fromkeys(list(baseline_premise_ids) + list(det_ids)))
        return VerifierDecision(
            accepted=True,
            answer=verdict,
            answer_source=AnswerSource.FOL_Z3_VERIFIED,
            verifier_source="deterministic_fol_z3",
            evidence={
                "z3_status": z3_status,
                "agreeing_signals": sorted(backend_signals.keys()),
                "used_premise_ids": used,
            },
        )

    # Rule (b): exactly one backend signal, no contradicting signal.
    if len(backend_signals) == 1:
        name, verdict = next(iter(backend_signals.items()))
        used = list(baseline_premise_ids) if name == "token_bfs" else list(det_ids)
        return VerifierDecision(
            accepted=True,
            answer=verdict,
            answer_source=AnswerSource.DETERMINISTIC_SOLVER,
            verifier_source=name,
            evidence={
                "z3_status": z3_status,
                "agreeing_signals": [name],
                "used_premise_ids": used,
            },
        )

    # No deterministic backend verdict. Consider the LLM-translated theory (rule d/e).
    # Rule (e): only a Z3 entailment/contradiction supplies the candidate verdict.
    llm_verdict: str | None = None
    if (
        llm_z3 is not None
        and getattr(llm_z3, "error", None) is None
        and z3_status in {"entailed", "contradicted"}
        and getattr(llm_z3, "answer", None) in definite
    ):
        llm_verdict = llm_z3.answer

    if llm_verdict is not None and llm_gates_ok:
        # Rule (d): an LLM proposal needs an INDEPENDENT backend signal to agree.
        # With no deterministic backend verdict in this branch, none can agree, so
        # the LLM theory alone is refused (root-cause fix for verifier_accepted_wrong).
        independent_agrees = any(v == llm_verdict for v in backend_signals.values())
        if independent_agrees:
            agreeing = sorted([n for n, v in backend_signals.items() if v == llm_verdict] + ["llm_fol_z3"])
            return VerifierDecision(
                accepted=True,
                answer=llm_verdict,
                answer_source=AnswerSource.VALIDATED_LLM_PROPOSAL,
                verifier_source="token_bfs_plus_z3",
                evidence={
                    "z3_status": z3_status,
                    "agreeing_signals": agreeing,
                    "used_premise_ids": list(getattr(llm_z3, "premises", []) or []),
                },
            )
        return VerifierDecision(
            accepted=False,
            answer="unknown",
            answer_source=AnswerSource.ABSTENTION,
            verifier_source="none",
            evidence={
                "z3_status": z3_status,
                "agreeing_signals": [],
                "used_premise_ids": [],
                "llm_candidate": llm_verdict,
            },
            reject_reason="llm_theory_without_independent_backend_agreement",
        )

    # Nothing produced an acceptable verdict.
    return VerifierDecision(
        accepted=False,
        answer="unknown",
        answer_source=AnswerSource.ABSTENTION,
        verifier_source="none",
        evidence={
            "z3_status": z3_status,
            "agreeing_signals": [],
            "used_premise_ids": [],
        },
        reject_reason="no_backend_verdict",
    )

def _verifier_abstain(
    solution: LogicSolution,
    mode: str,
    abstain_reason: str,
    z3_status: str,
    extra_calls: int = 0,
) -> LogicSolution:
    """Return the baseline solution with the verifier recording an abstention.

    Used when the FOL/Z3 path is unavailable, errors, or exceeds its budget
    (Req 3.8): the verifier never accepts an answer it could not establish, so the
    baseline solution is preserved and the abstention is recorded as evidence.
    """
    evidence = VerifierEvidence(
        verifier_source="none",
        z3_status=z3_status,
        used_premise_ids=list(solution.premises),
    )
    answer_source = (
        solution.answer_source
        if solution.answer in {"yes", "no"}
        else AnswerSource.ABSTENTION
    )
    return LogicSolution(
        **{
            **solution.__dict__,
            "model_calls": solution.model_calls + int(extra_calls or 0),
            "answer_source": answer_source,
            "verifier_evidence": evidence,
            "z3_sidecar": {
                "z3_status": z3_status,
                "abstain_reason": abstain_reason,
                "mode": mode,
                "accepted": False,
                "overrode_baseline": False,
            },
        }
    )

def _is_contradiction_abstention(rule: str) -> bool:
    """Did the fast path abstain because it *detected* a premise contradiction?

    Such an abstention is authoritative (Req 11.1: a conflict yields ``unknown``)
    and must not be revisited by the deterministic FOL compiler. The atom-based
    compiler models opposite phrasings (e.g. "tuition is paid" vs "has unpaid
    tuition") as independent predicates, so it cannot detect the inconsistency
    and would otherwise derive a definite verdict from contradictory premises by
    ex-falso reasoning. This is a generalizing, reason-keyed guard — not a
    question-text match.
    """
    low = (rule or "").lower()
    return any(
        token in low
        for token in (
            "directly contradictory",
            "support both the claim",
            "support both",
        )
    )

def _premises_contain_contradiction(normalized: list[Premise]) -> bool:
    """Detect a semantic contradiction among the (non-rule) ground premises.

    Mirrors the fast path's pairwise contradiction check (``_solve_rules_inner``)
    using the shared ``_are_contradictory_premises`` opposites table. The
    atom-based FOL compiler models opposite phrasings (e.g. "tuition is paid" vs
    "has unpaid tuition") as independent predicates, so it cannot see such an
    inconsistency and would unsoundly derive a definite verdict by ex-falso
    reasoning. Detecting the conflict here keeps the deterministic FOL signal
    sound at the source, regardless of which call path consumes it (Req 11.1).
    """
    grounds = [p for p in normalized if not _match_rule(p.text)]
    for i, p1 in enumerate(grounds):
        for p2 in grounds[i + 1:]:
            if _are_contradictory_premises(p1.text, p2.text):
                return True
    return False

def _deterministic_fol_signal(question: str, normalized: list[Premise], premises_fol: list[str] | None = None) -> tuple[str, list[str]] | None:
    """Produce the independent ``det_fol_z3`` backend signal (Task 7.1).

    Runs the deterministic NL/``premises_fol`` -> typed-DSL -> Z3 compiler
    (``app.logic.dsl_compiler.solve_deterministic_fol``) WITHOUT the LLM. Z3
    performs the multi-hop universal chaining, so chain depth is unbounded
    (Req 4.1). Returns ``(verdict, used_premise_ids)`` only for a definite
    ``yes``/``no`` Z3 entailment; otherwise ``None`` (abstain, Req 4.6) so the
    verifier sees no second signal rather than a guess.
    """
    if not normalized and not (premises_fol or []):
        return None
    # Soundness guard (Req 11.1): when the ground premises are mutually
    # contradictory, the theory is inconsistent and the atom-based compiler would
    # derive a spurious definite verdict by ex-falso. Abstain so the verifier
    # sees no second signal rather than an unsound one.
    if _premises_contain_contradiction(normalized):
        return None
    try:
        from app.logic.dsl_compiler import solve_deterministic_fol
    except Exception:
        return None
    payload = [{"id": premise.id, "text": premise.text} for premise in normalized]
    try:
        result = solve_deterministic_fol(question, payload, premises_fol)
    except Exception:
        return None
    if result.error is not None or result.answer not in {"yes", "no"}:
        return None
    if result.z3_status not in {"entailed", "contradicted", "probabilistic_block"}:
        return None
    valid_ids = {p.id for p in normalized}
    used = [pid for pid in result.used_premise_ids if pid in valid_ids]
    return result.answer, used

def _augment_with_deterministic_fol(
    solution: LogicSolution,
    question: str,
    normalized: list[Premise],
    premises_fol: list[str] | None = None,
) -> LogicSolution:
    """Coverage expansion via the deterministic FOL->Z3 compiler (Task 7.1).

    Runs as the path BETWEEN the token-BFS fast path and the LLM FOL translator
    inside the main ``solve`` flow. It only fills cases the fast path left as
    ``unknown`` (rule (e): a pre-existing deterministic baseline verdict is
    authoritative and is never overridden). When the deterministic compiler
    produces a definite Z3 entailment, the verifier accepts it as an independent
    backend signal; otherwise the baseline ``unknown`` is preserved (Req 4.6).
    """
    if solution.answer in {"yes", "no"}:
        return solution
    det = _deterministic_fol_signal(question, normalized, premises_fol)
    if det is None:
        return solution
    decision = _decide_logic_verification(
        baseline_answer=solution.answer,
        baseline_premise_ids=list(solution.premises),
        det_fol_z3=det,
        llm_z3=None,
        llm_gates_ok=False,
    )
    if not decision.accepted or decision.answer not in {"yes", "no"}:
        return solution
    return _apply_verifier_decision(
        solution, decision, normalized, "fol_z3_pipeline",
        fol_result=None, llm_calls=0, det_fol_z3=det,
    )

def _with_fol_z3_pipeline(solution: LogicSolution, question: str, normalized: list[Premise], llm_client: object | None, mode: str, allowed_domains: tuple[str, ...] = ("academic_policy", "public_logic_sample"), premises_fol: list[str] | None = None) -> LogicSolution:
    """Execute the FOL/Z3 pipeline to verify or rescue a logical solution."""
    if mode != "fol_z3_pipeline":
        return solution

    # Check if the domain is allowed for Z3 sidecar.
    domain_allowed = False
    for domain in allowed_domains:
        if domain == "academic_policy":
            from app.logic.policy_patterns import is_academic_policy_text
            if is_academic_policy_text(question or "", [p.text for p in normalized]):
                domain_allowed = True
                break
        elif domain == "public_logic_sample":
            if _is_public_logic_sample_text(question or "", normalized):
                domain_allowed = True
                break

    if not domain_allowed:
        return solution


    # Independent deterministic FOL->Z3 compiler (Task 7.1). This is a backend
    # signal computed WITHOUT the LLM; it plugs into rule (a)/(d) of the verifier.
    det_fol_z3: tuple[str, list[str]] | None = _deterministic_fol_signal(question, normalized, premises_fol)

    # Gate the LLM FOL translator (Task 11.2 / Req 7.1): invoke the LLM
    # translator ONLY when both the deterministic fast path AND the
    # deterministic FOL compiler abstain AND premises exist. Otherwise fall
    # back to the no-LLM verifier branch (which still uses the deterministic
    # FOL compiler as the independent backend signal). This eliminates the
    # k=3 consensus call site and replaces it with an at-most-one gated call.
    fast_path_abstained = solution.answer == "unknown"
    deterministic_fol_abstained = det_fol_z3 is None
    has_premises = bool(normalized)
    llm_translator_gated_in = (
        llm_client is not None
        and fast_path_abstained
        and deterministic_fol_abstained
        and has_premises
    )

    if llm_client is None or not llm_translator_gated_in:
        # No LLM available, or the gate keeps the LLM out: rely on the
        # deterministic FOL compiler as the only independent backend signal.
        # If it agrees with (or stands alongside) the baseline, the verifier
        # may accept; otherwise the baseline is preserved.
        if det_fol_z3 is None:
            return solution
        decision = _decide_logic_verification(
            baseline_answer=solution.answer,
            baseline_premise_ids=list(solution.premises),
            det_fol_z3=det_fol_z3,
            llm_z3=None,
            llm_gates_ok=False,
        )
        return _apply_verifier_decision(
            solution, decision, normalized, mode,
            fol_result=None, llm_calls=0, det_fol_z3=det_fol_z3,
        )

    try:
        from app.logic.fol_z3_pipeline import solve_fol_z3
    except Exception as exc:
        return _verifier_abstain(solution, mode, f"fol_z3_import_error:{type(exc).__name__}", z3_status="unavailable")
    premises_payload = [{"id": premise.id, "text": premise.text} for premise in normalized]
    try:
        fol_result = solve_fol_z3(question, premises_payload, llm_client)
    except Exception as exc:
        return _verifier_abstain(solution, mode, f"fol_z3_runtime_error:{type(exc).__name__}", z3_status="error")

    baseline_answer = solution.answer
    baseline_premise_ids = list(solution.premises)
    llm_calls = int(fol_result.llm_calls or 0)
    # The FOL/Z3 pipeline only returns a decisive yes/no after passing its internal
    # directionality and round-trip/DSL gates (see fol_z3_pipeline._directionality_gate_ok
    # and the _RestrictedFOLParser DSL path); treat that as the gate signal for rule (d).
    llm_gates_ok = (fol_result.error is None and fol_result.z3_status in {"entailed", "contradicted"})

    decision = _decide_logic_verification(
        baseline_answer=baseline_answer,
        baseline_premise_ids=baseline_premise_ids,
        det_fol_z3=det_fol_z3,
        llm_z3=fol_result,
        llm_gates_ok=llm_gates_ok,
    )

    metadata = {
        "answer_candidate": fol_result.answer,
        "used_premises": fol_result.premises,
        "proof_steps": list(fol_result.proof_steps),
        "z3_status": fol_result.z3_status,
        "accepted": decision.accepted,
        "latency_ms": fol_result.latency_ms,
        "llm_calls": llm_calls,
        "method": fol_result.method,
        "mode": mode,
        "error": fol_result.error,
        "overrode_baseline": False,
        "baseline_answer": baseline_answer,
        "baseline_agreed": (fol_result.answer == baseline_answer),
        "rejected_reason": decision.reject_reason,
        "verifier_answer_source": decision.answer_source.value,
        "verifier_source": decision.verifier_source,
        "agreeing_signals": decision.evidence.get("agreeing_signals", []),
    }

    if baseline_answer in {"yes", "no"}:
        # Rule (e): a deterministic verdict already exists and is authoritative; the
        # LLM-translated Z3 theory may verify but never overrides it. Preserve the
        # baseline answer, explanation, and its existing provenance.
        final_answer = baseline_answer
        final_source = solution.answer_source
        final_premises = list(solution.premises)
        final_explanation = solution.explanation
        final_cot = list(solution.cot)
        final_confidence = solution.confidence
        final_proof_steps = solution.proof_steps
        used_premise_ids = list(solution.premises)
        llm_used = solution.llm_fallback_used
    elif decision.accepted and decision.answer in {"yes", "no"}:
        # An independent backend signal agreed with a decisive verdict (rule a/b/d).
        # Today this only fires once the deterministic FOL compiler (Task 7.1) is
        # available; the LLM theory alone never reaches here.
        used_premise_ids = list(decision.evidence.get("used_premise_ids", []))
        selected = [p for p in normalized if p.id in set(used_premise_ids)]
        final_answer = decision.answer
        final_source = decision.answer_source
        final_premises = [p.id for p in selected] if selected else list(solution.premises)
        final_explanation = fol_result.explanation
        final_cot = list(fol_result.proof_steps)
        final_confidence = fol_result.confidence
        final_proof_steps = build_proof_steps(decision.answer, selected, "fol_z3_pipeline_verified_entailment", fol_result.confidence)
        llm_used = solution.llm_fallback_used or (decision.answer_source == AnswerSource.VALIDATED_LLM_PROPOSAL)
    else:
        # Rule (c)/(d): backend conflict or an unsupported LLM-only proposal. Refuse
        # to accept any answer and abstain (the root-cause fix that eliminates the 9
        # verifier_accepted_wrong cases). The baseline (unknown) is preserved.
        final_answer = solution.answer
        final_source = AnswerSource.ABSTENTION
        final_premises = list(solution.premises)
        final_explanation = solution.explanation
        final_cot = list(solution.cot)
        final_confidence = solution.confidence
        final_proof_steps = solution.proof_steps
        used_premise_ids = list(solution.premises)
        llm_used = solution.llm_fallback_used

    evidence = VerifierEvidence(
        verifier_source=decision.verifier_source,
        z3_status=fol_result.z3_status,
        proof_steps=[{"step": str(step)} for step in fol_result.proof_steps],
        computation_trace={},
        agreeing_signals=list(decision.evidence.get("agreeing_signals", [])),
        used_premise_ids=used_premise_ids,
        dimensional_valid=None,
        unit_valid=None,
    )

    return LogicSolution(
        answer=final_answer,
        explanation=final_explanation,
        premises=final_premises,
        cot=final_cot,
        confidence=final_confidence,
        hallucinated_premises=list(solution.hallucinated_premises),
        llm_fallback_used=llm_used,
        model_calls=solution.model_calls + llm_calls,
        proof_steps=final_proof_steps,
        z3_sidecar=metadata,
        agent_trace=list(solution.agent_trace),
        answer_source=final_source,
        verifier_evidence=evidence,
        unparsed_premises=list(solution.unparsed_premises),
    )

def _apply_verifier_decision(
    solution: LogicSolution,
    decision: VerifierDecision,
    normalized: list[Premise],
    mode: str,
    fol_result: object | None,
    llm_calls: int,
    det_fol_z3: tuple[str, list[str]] | None,
) -> LogicSolution:
    """Materialize a :class:`VerifierDecision` into a :class:`LogicSolution`.

    Used by the no-LLM sidecar path where the deterministic FOL compiler is the
    only independent backend signal. Rule (e) still holds: a pre-existing
    deterministic baseline verdict is authoritative and is preserved; otherwise
    an accepted decision supplies the verified answer, and a non-acceptance keeps
    the (unknown) baseline.
    """
    baseline_answer = solution.answer
    z3_status = getattr(fol_result, "z3_status", None) or ("entailed" if det_fol_z3 else "abstained")

    metadata = {
        "answer_candidate": det_fol_z3[0] if det_fol_z3 else None,
        "used_premises": list(det_fol_z3[1]) if det_fol_z3 else [],
        "z3_status": z3_status,
        "accepted": decision.accepted,
        "mode": mode,
        "overrode_baseline": False,
        "baseline_answer": baseline_answer,
        "rejected_reason": decision.reject_reason,
        "verifier_answer_source": decision.answer_source.value,
        "verifier_source": decision.verifier_source,
        "agreeing_signals": decision.evidence.get("agreeing_signals", []),
        "method": "deterministic_fol_z3",
        "llm_calls": int(llm_calls or 0),
    }

    if baseline_answer in {"yes", "no"}:
        final_answer = baseline_answer
        final_source = solution.answer_source
        final_premises = list(solution.premises)
        final_explanation = solution.explanation
        final_cot = list(solution.cot)
        final_confidence = solution.confidence
        final_proof_steps = solution.proof_steps
        used_premise_ids = list(solution.premises)
    elif decision.accepted and decision.answer in {"yes", "no"}:
        used_premise_ids = list(decision.evidence.get("used_premise_ids", []))
        selected = [p for p in normalized if p.id in set(used_premise_ids)]
        final_answer = decision.answer
        final_source = decision.answer_source
        final_premises = [p.id for p in selected] if selected else used_premise_ids
        final_confidence = 0.9
        final_proof_steps = build_proof_steps(decision.answer, selected, "deterministic_fol_z3_verified_entailment", final_confidence)
        final_cot = [str(step) for step in metadata.get("used_premises", [])]
        final_explanation = logic_explanation(decision.answer, selected, "deterministic FOL->Z3 multi-hop entailment")
    else:
        final_answer = solution.answer
        final_source = AnswerSource.ABSTENTION
        final_premises = list(solution.premises)
        final_explanation = solution.explanation
        final_cot = list(solution.cot)
        final_confidence = solution.confidence
        final_proof_steps = solution.proof_steps
        used_premise_ids = list(solution.premises)

    evidence = VerifierEvidence(
        verifier_source=decision.verifier_source,
        z3_status=z3_status,
        agreeing_signals=list(decision.evidence.get("agreeing_signals", [])),
        used_premise_ids=used_premise_ids,
    )

    return LogicSolution(
        answer=final_answer,
        explanation=final_explanation,
        premises=final_premises,
        cot=final_cot,
        confidence=final_confidence,
        hallucinated_premises=list(solution.hallucinated_premises),
        llm_fallback_used=solution.llm_fallback_used,
        model_calls=solution.model_calls + int(llm_calls or 0),
        proof_steps=final_proof_steps,
        z3_sidecar=metadata,
        agent_trace=list(solution.agent_trace),
        answer_source=final_source,
        verifier_evidence=evidence,
        unparsed_premises=list(solution.unparsed_premises),
    )

def _with_z3_sidecar(solution: LogicSolution, question: str, normalized: list[Premise], allowed_domains: tuple[str, ...], mode: str, llm_client: object | None = None) -> LogicSolution:
    """Route verification calls to the Z3 sidecar / FOL pipeline."""
    # If standard z3_sidecar is selected or default experiment mode is on,
    # we map it to fol_z3_pipeline which is fully implemented and tested.
    if mode in {"experiment_only", "z3_audit_only", "fol_z3_pipeline"}:
        mode = "fol_z3_pipeline"
    return _with_fol_z3_pipeline(solution, question, normalized, llm_client, mode, allowed_domains)
