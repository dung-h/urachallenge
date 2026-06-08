"""Data classes for representing logical solutions, verifier decisions, facts, and rules.

This module defines the core data transfer objects and structures used across the logic
reasoning package, bridging the parser, solvers, and verifiers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from app.logic.premise_selector import Premise
from app.logic.proof_trace import ProofStep
from app.schemas import AnswerSource, VerifierEvidence

@dataclass
class LogicSolution:
    """Holds the result and metadata of a logic solving execution.

    Contains the resolved answer, explanation, selected premises, chain of thought,
    confidence metrics, and detailed tracing information (such as Z3 details and
    individual proof steps) required by downstream components.
    """
    answer: str
    explanation: str
    premises: list[str]
    cot: list[str] = field(default_factory=list)
    confidence: float = 0.0
    hallucinated_premises: list[str] = field(default_factory=list)
    llm_fallback_used: bool = False
    model_calls: int = 0
    proof_steps: list[ProofStep] = field(default_factory=list)
    z3_sidecar: dict[str, object] | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    answer_source: AnswerSource = AnswerSource.ABSTENTION
    verifier_evidence: VerifierEvidence = field(default_factory=VerifierEvidence)
    # Retained premises that could not be parsed into a supported structure
    # (Req 6.1, 6.3). Each entry is a {"id", "text", "reason"} dict carried on
    # the solution so downstream components (router/trace/confidence) can see the
    # original premise text plus its failure reason instead of silently dropping
    # it. Empty when every premise parsed.
    unparsed_premises: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class VerifierDecision:
    """Outcome of the logic verifier's independent-backend-agreement check.

    The Verifier (AGENTS.md §20.5, design "Verifier" component) accepts an answer
    only when an independent backend computation agrees with it; otherwise it
    abstains. answer is the verdict the verifier authorizes (yes / no /
    unknown). answer_source records the provenance per Req 1.3/1.4.
    verifier_source names which backend(s) produced the agreement
    (token_bfs_plus_z3 | deterministic_fol_z3 | none). evidence
    carries the proof/agreement detail retained for harness scoring (Req 3.6).
    """

    accepted: bool
    answer: str
    answer_source: AnswerSource
    verifier_source: str
    evidence: dict
    reject_reason: str | None = None


@dataclass
class Fact:
    """Represents a logical fact extracted from a premise.

    Attributes:
        text: The normalized text representation of the fact.
        tokens: The set of content/predicate tokens extracted from the text.
        positive: True if the fact is positive, False if it is negated.
        premises: The list of source Premise objects from which this fact originated.
    """
    text: str
    tokens: set[str]
    positive: bool
    premises: list[Premise]


@dataclass
class Rule:
    """Represents an implication rule parsed from a premise.

    Attributes:
        premise: The source Premise object.
        antecedent_tokens: Tokens extracted from the antecedent clause.
        consequent_tokens: Tokens extracted from the consequent clause.
        antecedent_positive: True if the antecedent is positive, False if negated.
        consequent_positive: True if the consequent is positive, False if negated.
    """
    premise: Premise
    antecedent_tokens: set[str]
    consequent_tokens: set[str]
    antecedent_positive: bool = True
    consequent_positive: bool = True
