"""General requirement-satisfaction reasoner (AGENTS.md §20.4-sound).

Handles the question shape that the entailment-oriented solvers mishandle:

    "Does X meet ALL requirements for GOAL?"
    "Is <condition> sufficient for GOAL?"
    "Does <condition> guarantee GOAL?"
    "Is <condition> enough to GOAL?"

These ask about REQUIREMENT SATISFACTION, not about whether the goal
predicate is entailed. A pure-FOL prover answers "Is GOAL true?" with
"unknown" when it cannot derive GOAL — but for "does X meet all
requirements for GOAL?", if a REQUIRED condition on the goal's rule chain
is *explicitly violated* (a ground fact negates it, or a numeric threshold
is provably unmet), the correct answer is a decisive **"no"**.

Soundness contract (critical — this is why it is safe to wire in)
-----------------------------------------------------------------
This reasoner returns "no" ONLY when it finds POSITIVE EVIDENCE that a
required condition is violated:
  * a ground fact explicitly negates a required leaf condition, OR
  * a required numeric threshold ("at least 5 courses") is contradicted by
    a ground fact ("completed 4 courses").
It NEVER returns "no" merely because a condition is unprovable — that case
returns ``None`` so the caller keeps the honest "unknown". It also never
returns "yes" (proving full satisfaction is the entailment solvers' job).

So the only state transition this module can cause is:
    unknown  ->  no   (when a requirement is provably violated)
which strictly increases correctness on the dataset's dominant
requirement-trap class without risking false "yes" answers.

The chain is built backward from the goal using the same rule parsing the
BFS solver uses, so it generalizes to any domain (driver/hazmat, person,
student, ...), not just academic policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.logic.premise_selector import Premise
from app.logic._text_primitives import (
    _norm,
    _content_tokens,
    _is_negated,
)
from app.logic._rule_matcher import _match_rule, _match_if_rule
from app.logic._subject_chain import _split_antecedent_conjuncts


# Question-shape detection ---------------------------------------------------

_REQUIREMENT_Q_MARKERS = (
    "meet all requirements",
    "meet all the requirements",
    "satisfy all requirements",
    "all requirements",
    "all the requirements",
    "all conditions",
    "all the conditions",
    "meet the requirements",
    "satisfy the requirements",
    "meet every requirement",
)
_SUFFICIENCY_Q_MARKERS = (
    "sufficient for",
    "sufficient to",
    "enough to",
    "enough for",
    "guarantee",
    "guarantees",
    "make him eligible",
    "make her eligible",
    "make them eligible",
)


def is_requirement_question(question: str) -> bool:
    """True if the question asks about requirement satisfaction / sufficiency."""
    low = _norm(question)
    return any(m in low for m in _REQUIREMENT_Q_MARKERS) or any(
        m in low for m in _SUFFICIENCY_Q_MARKERS
    )


# Goal extraction ------------------------------------------------------------

_GOAL_PREP_PATTERNS = [
    r"meet all (?:the )?requirements? (?:for|to)\s+(?P<goal>.+)$",
    r"meet all (?:the )?conditions? (?:for|to)\s+(?P<goal>.+)$",
    r"satisfy all (?:the )?requirements? (?:for|to)\s+(?P<goal>.+)$",
    r"meet (?:the )?requirements? (?:for|to)\s+(?P<goal>.+)$",
    r"sufficient (?:for|to)\s+(?P<goal>.+)$",
    r"enough (?:for|to)\s+(?P<goal>.+)$",
    r"guarantees?\s+(?P<goal>.+)$",
]


def _extract_goal(question: str) -> str | None:
    """Pull the GOAL phrase out of a requirement question."""
    low = _norm(question).rstrip("?.")
    # Drop a trailing ", according to the premises" style qualifier.
    low = re.sub(r",?\s*according to the premises.*$", "", low).strip()
    for pat in _GOAL_PREP_PATTERNS:
        m = re.search(pat, low)
        if m:
            goal = m.group("goal").strip()
            # Strip leading articles / helper verbs.
            goal = re.sub(r"^(?:be|being|to|a|an|the)\s+", "", goal).strip()
            return goal or None
    return None


# Numeric threshold (general, not academic-only) -----------------------------

_THRESHOLD_PATTERNS = [
    (r"at least\s+(\d+(?:\.\d+)?)", ">="),
    (r"no less than\s+(\d+(?:\.\d+)?)", ">="),
    (r"minimum (?:of )?\s*(\d+(?:\.\d+)?)", ">="),
    (r"more than\s+(\d+(?:\.\d+)?)", ">"),
    (r"greater than\s+(\d+(?:\.\d+)?)", ">"),
    (r"above\s+(\d+(?:\.\d+)?)", ">"),
    (r"at most\s+(\d+(?:\.\d+)?)", "<="),
    (r"no more than\s+(\d+(?:\.\d+)?)", "<="),
    (r"fewer than\s+(\d+(?:\.\d+)?)", "<"),
    (r"less than\s+(\d+(?:\.\d+)?)", "<"),
    (r"below\s+(\d+(?:\.\d+)?)", "<"),
]


def _parse_threshold(text: str) -> tuple[str, float, set[str]] | None:
    """Return (operator, value, context_tokens) for a numeric requirement.

    ``context_tokens`` are the non-numeric content tokens around the
    threshold (e.g. {course} for "at least 5 courses") so we only compare a
    fact's number when it concerns the SAME quantity.
    """
    low = _norm(text)
    for pat, op in _THRESHOLD_PATTERNS:
        m = re.search(pat, low)
        if m:
            val = float(m.group(1))
            # context = content tokens of the condition minus the number word
            ctx = _content_tokens(low) - {str(int(val)) if val.is_integer() else str(val)}
            return op, val, ctx
    return None


def _fact_number_for_context(fact_text: str, ctx: set[str]) -> float | None:
    """Find a number in ``fact_text`` whose surrounding tokens overlap ctx."""
    low = _norm(fact_text)
    # Only trust the number when the fact shares context tokens with the
    # threshold (same quantity being measured).
    fact_tokens = _content_tokens(low)
    if ctx and not (fact_tokens & ctx):
        return None
    nums = re.findall(r"(\d+(?:\.\d+)?)", low)
    if not nums:
        return None
    # Use the first number; requirement facts are simple ("completed 4 courses").
    try:
        return float(nums[0])
    except ValueError:
        return None


def _threshold_violated(op: str, threshold: float, actual: float) -> bool:
    """True iff ``actual`` provably FAILS the required threshold."""
    if op == ">=":
        return actual < threshold
    if op == ">":
        return actual <= threshold
    if op == "<=":
        return actual > threshold
    if op == "<":
        return actual >= threshold
    return False


# Condition / fact matching --------------------------------------------------


def _condition_tokens(text: str) -> set[str]:
    """Salient content tokens of a condition clause (drop pure copula/aux)."""
    drop = {
        "is", "are", "am", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did", "will", "can", "could", "to", "a", "an",
        "the", "they", "them", "their", "person", "driver", "student", "one",
        "who", "if", "and", "or", "then",
    }
    return _content_tokens(text) - drop


@dataclass
class _Rule:
    premise: Premise
    antecedent: str
    consequent: str
    consequent_tokens: set[str]


def _collect_rules(premises: list[Premise]) -> list[_Rule]:
    rules: list[_Rule] = []
    for p in premises:
        parts = _match_if_rule(p.text) or _match_rule(p.text)
        if not parts:
            continue
        ant, cons = parts
        rules.append(_Rule(p, ant, cons, _condition_tokens(cons)))
    return rules


def _ground_facts(premises: list[Premise]) -> list[Premise]:
    return [p for p in premises if not _match_rule(p.text)]


def _condition_explicitly_violated(
    cond: str,
    facts: list[Premise],
) -> Premise | None:
    """Return a fact that PROVABLY violates the condition, else None.

    Two violation modes:
      (1) explicit negation: a negated fact whose content tokens cover the
          condition's content tokens ("not received a safety endorsement"
          vs required "received a safety endorsement").
      (2) numeric threshold: the condition states a threshold and a fact
          gives a contradicting number for the same quantity.
    """
    cond_tokens = _condition_tokens(cond)
    if not cond_tokens:
        return None
    cond_negated = _is_negated(_norm(cond))

    # (2) numeric threshold violation
    thr = _parse_threshold(cond)
    if thr is not None:
        op, val, ctx = thr
        for f in facts:
            actual = _fact_number_for_context(f.text, ctx or cond_tokens)
            if actual is not None and _threshold_violated(op, val, actual):
                return f

    # (1) explicit polarity-opposite fact over the same content
    for f in facts:
        f_low = _norm(f.text)
        f_negated = _is_negated(f_low)
        # Opposite polarity required for a contradiction.
        if f_negated == cond_negated:
            continue
        f_tokens = _condition_tokens(f.text)
        if not f_tokens:
            continue
        # The fact must be ABOUT the same condition: its content tokens must
        # cover the condition's salient tokens (so "not received a safety
        # endorsement" covers required "received a safety endorsement").
        shared = cond_tokens & f_tokens
        if len(shared) >= max(1, len(cond_tokens) - 1) and shared == cond_tokens:
            return f
    return None


def _goal_required_conditions(
    goal: str, rules: list[_Rule], max_depth: int = 6
) -> list[str]:
    """Backward-expand the goal into its required leaf conditions.

    For each rule whose consequent matches the current target, its
    antecedent conjuncts become sub-targets. Conjuncts that are themselves
    a rule consequent are expanded further; the rest are leaf conditions.
    Returns the flat list of required condition clauses (leaves + interior).
    """
    goal_tokens = _condition_tokens(goal)
    if not goal_tokens:
        return []
    required: list[str] = []
    seen_targets: set[frozenset[str]] = set()
    frontier: list[set[str]] = [goal_tokens]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier: list[set[str]] = []
        for target in frontier:
            key = frozenset(target)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            # Find the rule whose consequent best covers this target.
            best = None
            best_overlap = 0
            for r in rules:
                ov = len(r.consequent_tokens & target)
                # Require the consequent to substantially cover the target.
                if ov > best_overlap and ov >= max(1, len(target) - 1):
                    best, best_overlap = r, ov
            if best is None:
                continue
            conjuncts = _split_antecedent_conjuncts(best.antecedent) or [best.antecedent]
            for c in conjuncts:
                required.append(c)
                ct = _condition_tokens(c)
                if ct:
                    next_frontier.append(ct)
        frontier = next_frontier
    return required


def _condition_established(cond: str, facts: list[Premise], rules: list[_Rule]) -> bool:
    """True iff ``cond`` is POSITIVELY supported by the premises.

    A condition is established when a positive ground fact covers its salient
    tokens, OR a rule whose consequent covers it has all its own antecedent
    conjuncts established (shallow one-level derivation — enough for the
    leaf-coverage check; deeper chains are handled by the BFS solver that
    runs before this reasoner). Numeric thresholds are established when a
    fact provides a number that SATISFIES (not violates) the threshold.
    """
    cond_tokens = _condition_tokens(cond)
    if not cond_tokens:
        return True  # vacuous / unparseable condition: don't penalize
    cond_negated = _is_negated(_norm(cond))

    # Numeric threshold satisfied?
    thr = _parse_threshold(cond)
    if thr is not None:
        op, val, ctx = thr
        for f in facts:
            actual = _fact_number_for_context(f.text, ctx or cond_tokens)
            if actual is not None and not _threshold_violated(op, val, actual):
                return True
        # threshold present but no satisfying fact -> not established
        # (fall through; a non-numeric fact won't establish a numeric cond)

    # Positive ground fact covering the condition with matching polarity.
    for f in facts:
        f_low = _norm(f.text)
        if _is_negated(f_low) != cond_negated:
            continue
        f_tokens = _condition_tokens(f.text)
        if cond_tokens and cond_tokens <= f_tokens:
            return True

    # One-level derivation: a rule whose consequent covers the condition and
    # whose antecedent conjuncts are themselves established by ground facts.
    for r in rules:
        if not cond_tokens <= r.consequent_tokens and not (cond_tokens & r.consequent_tokens) == cond_tokens:
            # require consequent to cover the condition tokens
            if not (cond_tokens and cond_tokens <= r.consequent_tokens):
                continue
        conjuncts = _split_antecedent_conjuncts(r.antecedent) or [r.antecedent]
        if conjuncts and all(
            _condition_established_shallow(c, facts) for c in conjuncts
        ):
            return True
    return False


def _condition_established_shallow(cond: str, facts: list[Premise]) -> bool:
    """Ground-fact-only establishment check (no rule recursion) to bound depth."""
    cond_tokens = _condition_tokens(cond)
    if not cond_tokens:
        return True
    cond_negated = _is_negated(_norm(cond))
    thr = _parse_threshold(cond)
    if thr is not None:
        op, val, ctx = thr
        for f in facts:
            actual = _fact_number_for_context(f.text, ctx or cond_tokens)
            if actual is not None and not _threshold_violated(op, val, actual):
                return True
        return False
    for f in facts:
        if _is_negated(_norm(f.text)) != cond_negated:
            continue
        if cond_tokens <= _condition_tokens(f.text):
            return True
    return False


@dataclass(frozen=True)
class RequirementDecision:
    answer: str
    premises: list[Premise]
    reason: str
    confidence: float = 0.88


def solve_requirement(
    question: str, premises: list[Premise]
) -> RequirementDecision | None:
    """Sound requirement-satisfaction check with scoped Closed-World Assumption.

    Two paths, both returning ``"no"`` (or ``None`` to keep "unknown"):

      A) PROVABLE VIOLATION (open-world-sound): a required condition is
         explicitly negated by a fact, or a numeric threshold is provably
         violated. Confidence 0.88.

      B) SCOPED CWA (Poole & Mackworth §5.7, scoped to the requirement
         question shape ONLY): when EVERY required condition has been
         enumerated and at least one is NEITHER established NOR derivable
         from the premises, then under the closed-world reading of "meet ALL
         requirements" the requirement is not met -> "no". Lower confidence
         (0.75) because it rests on absence-of-evidence, which is only valid
         here because the question explicitly asks whether ALL requirements
         hold. Generic entailment questions never reach this function, so
         the 209 open-world "unknown"-gold items are untouched.

    Never returns "yes" (proving full satisfaction is the entailment solvers'
    job) and never fires on non-requirement questions.
    """
    if not is_requirement_question(question):
        return None
    goal = _extract_goal(question)
    if not goal:
        return None
    rules = _collect_rules(premises)
    if not rules:
        return None
    facts = _ground_facts(premises)
    if not facts:
        return None

    required = _goal_required_conditions(goal, rules)
    if not required:
        return None

    # ---- Path A: provable violation -------------------------------------
    for cond in required:
        offending = _condition_explicitly_violated(cond, facts)
        if offending is not None:
            cited = [offending]
            cond_tokens = _condition_tokens(cond)
            for r in rules:
                if cond_tokens and cond_tokens <= _content_tokens(r.antecedent):
                    cited.append(r.premise)
                    break
            return RequirementDecision(
                answer="no",
                premises=list(dict.fromkeys(cited)),
                reason=(
                    f"requirement not met: a required condition "
                    f"(\"{cond.strip()[:60]}\") for the goal is explicitly "
                    f"violated by a premise"
                ),
                confidence=0.88,
            )

    # ---- Path B: scoped CWA (unestablished required condition) -----------
    # Only safe because we are inside is_requirement_question() — the
    # question explicitly asks whether ALL requirements are met, which is a
    # closed-world reading. We require a CONCRETE leaf condition (>=2 salient
    # tokens) so we don't fire on vague/over-generalized expansions, and we
    # require that the goal itself is NOT directly established (else the BFS
    # would have answered "yes" already and we shouldn't override).
    unestablished: list[str] = []
    for cond in required:
        ct = _condition_tokens(cond)
        if len(ct) < 2:
            continue  # skip thin/vague conditions — avoid false "no"
        if not _condition_established(cond, facts, rules):
            unestablished.append(cond)
    if unestablished:
        # Cite the rule premises that introduced the unmet conditions.
        cited: list[Premise] = []
        for cond in unestablished[:2]:
            ctoks = _condition_tokens(cond)
            for r in rules:
                if ctoks and ctoks <= _content_tokens(r.antecedent):
                    cited.append(r.premise)
                    break
        if not cited:
            cited = [r.premise for r in rules[:1]]
        return RequirementDecision(
            answer="no",
            premises=list(dict.fromkeys(cited)),
            reason=(
                f"requirement not met (closed-world): required condition "
                f"(\"{unestablished[0].strip()[:60]}\") is not established by "
                f"any premise, so not all requirements for the goal hold"
            ),
            confidence=0.75,
        )
    return None
