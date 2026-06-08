"""Comparative-threshold rule resolution (Class-3 deep fix, Session 11o).

Handles rules whose antecedent is a COMPARATIVE THRESHOLD over an ordered
dimension, e.g.:

    "Lecturers with a degree higher than a Master's can teach undergraduate
     courses."
    "Employees with a salary above the median receive a bonus."

combined with:
    "Dr. John is a lecturer with a PhD."           (has-value fact)
    "A PhD is higher than a Master's degree."       (ordering premise)
    "A Master's degree is higher than a Bachelor's." (ordering premise)

The deterministic token-BFS cannot satisfy "degree higher than a Master's"
because the antecedent is a RELATION over the rank order, not a literal
predicate any ground fact carries. This module:

  1. extracts the rank-comparison graph (reuses _comparison_reasoner),
  2. finds threshold rules ("<class> with [a] [<attr>] higher/lower than
     <bound> <modal> <consequent>"),
  3. finds has-value facts ("<entity> ... with/has [a] <value>"),
  4. for each (rule, fact) where the entity is of the rule's class, checks
     the comparison graph: does <value> outrank (or under-rank) <bound>?
     If yes, the threshold is satisfied → emit a derived consequent fact
     "<entity> <consequent>".

Soundness: emits a derived fact ONLY when the ordering is PROVEN by the
graph (transitive reachability). If the ordering is unknown, nothing is
emitted (abstain-preserving, §20.4). Pure structural relation handling —
no per-question text.
"""

from __future__ import annotations

import re

from app.logic.premise_selector import Premise
from app.logic._proof_classes import Fact
from app.logic._text_primitives import _norm, _clean_content_tokens
from app.logic._comparison_reasoner import _extract_comparison_edges, _normalize_entity


# "<class> with [a] [<attr>] (higher|lower|greater|...) than [a] <bound> (modal) <consequent>"
_THRESHOLD_RE = re.compile(
    r"^(?P<cls>.+?)\s+with\s+(?:a\s+|an\s+)?(?:[a-z]+\s+)?"
    r"(?P<dir>higher|lower|greater|less|more|older|younger|taller|shorter|"
    r"larger|smaller|bigger|longer)\s+than\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<bound>.+?)\s+(?P<modal>can|are|is|may|will|must|qualify|qualifies|"
    r"receive|receives|get|gets)\s+(?P<conc>.+?)\.?$",
    re.I,
)

# "<entity> is [a] <class> with [a] <value>"  OR  "<entity> has [a] <value>"
_HASVAL_RE = re.compile(
    r"^(?P<subj>.+?)\s+(?:is\s+(?:a\s+|an\s+)?.*?\bwith\s+(?:a\s+|an\s+)?|"
    r"has\s+(?:a\s+|an\s+)?|holds\s+(?:a\s+|an\s+)?)(?P<value>.+?)\.?$",
    re.I,
)

# Directions that mean "subject outranks bound" (graph edge value -> bound).
_HIGHER_DIRS = {"higher", "greater", "more", "older", "taller", "larger",
                "bigger", "longer"}


def _bound_key(bound: str) -> str:
    """Normalized key for a bound phrase, tolerant of the 'degree' suffix."""
    b = _normalize_entity(bound)
    # Drop a trailing "degree" so "master's" and "master's degree" unify.
    b = re.sub(r"\s+degree$", "", b).strip()
    return b


def _node_key(node: str) -> str:
    n = _normalize_entity(node)
    return re.sub(r"\s+degree$", "", n).strip()


def resolve_threshold_rules(premises: list[Premise]) -> list[Fact]:
    """Return derived consequent Facts for satisfied comparative-threshold rules.

    Each returned Fact carries the consequent predicate tokens and is attributed
    to the rule premise + the has-value fact + the ordering premises that proved
    the comparison.
    """
    graphs = _extract_comparison_edges(premises)
    if not graphs:
        return []

    # Collect has-value facts: entity -> (value_key, premise)
    has_value: list[tuple[str, str, Premise]] = []
    for p in premises:
        low = _norm(p.text)
        if " higher than " in low or " lower than " in low or " than " in low:
            continue  # ordering/threshold premise, not a plain has-value fact
        m = _HASVAL_RE.match(low)
        if not m:
            continue
        subj = m.group("subj").strip()
        value = m.group("value").strip()
        # Reject when "value" is itself a clause (contains a verb-y connector).
        if any(w in value for w in (" can ", " are ", " is ", " who ", " that ", " with ")):
            continue
        has_value.append((subj, _node_key(value), p))

    if not has_value:
        return []

    derived: list[Fact] = []
    for p in premises:
        low = _norm(p.text)
        m = _THRESHOLD_RE.match(low)
        if not m:
            continue
        cls = m.group("cls").strip()
        direction = m.group("dir").lower()
        bound = _bound_key(m.group("bound"))
        conc = m.group("conc").strip()
        cls_tokens = _clean_content_tokens(cls)

        # Which graph dimension holds this bound? Pick any graph containing it.
        for dim, graph in graphs.items():
            node_keys = {_node_key(n): n for n in graph.nodes}
            if bound not in node_keys:
                continue
            bound_node = node_keys[bound]
            for subj, value_key, fact_p in has_value:
                if value_key not in node_keys:
                    continue
                value_node = node_keys[value_key]
                # Entity must belong to the rule's class (token overlap with
                # the has-value fact's text), to avoid cross-applying rules.
                fact_tokens = _clean_content_tokens(fact_p.text)
                if cls_tokens and not (cls_tokens & fact_tokens):
                    # class mention may be in the fact ("John is a lecturer..")
                    # require at least one class token present.
                    continue
                # Check ordering via the graph (transitive reachability).
                if direction in _HIGHER_DIRS:
                    ok = graph.reachable(value_node, bound_node) is not None
                else:
                    ok = graph.reachable(bound_node, value_node) is not None
                if not ok:
                    continue
                # Threshold satisfied → derive "<entity> <conc>".
                path_support = []
                chain = (graph.reachable(value_node, bound_node)
                         if direction in _HIGHER_DIRS
                         else graph.reachable(bound_node, value_node)) or []
                for a, b in zip(chain, chain[1:]):
                    sp = graph.support.get((a, b))
                    if sp:
                        path_support.append(sp)
                support = list(dict.fromkeys([p, fact_p] + path_support))
                conc_tokens = _clean_content_tokens(conc)
                if not conc_tokens:
                    continue
                derived.append(Fact(
                    text=f"{subj} {conc}",
                    tokens=conc_tokens | _clean_content_tokens(subj),
                    positive=True,
                    premises=support,
                ))
    return derived
