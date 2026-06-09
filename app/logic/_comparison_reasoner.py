"""Transitive comparison and ranking reasoner.

This module provides a deterministic solver for comparative-relation questions
that reduce to reachability / extremum queries over a partial order built from
premises (AGENTS.md §20: structural, no per-question text overrides):

  * Chained comparisons:   "A is taller than B", "B is taller than C"
                           → "Is A taller than C?"  → yes (transitive)
                           → "Is C taller than A?"  → no  (reverse edge)
  * Superlative queries:   "Who is tallest?" → the unique source of the order.
  * Transitive equality:   "A equals B", "B equals C" → "Is A equal to C?" → yes.

The reasoner extracts comparison edges with general regex over a small set of
comparative adjectives and their antonyms (taller/shorter, larger/smaller,
older/younger, etc.), builds a directed graph (X → Y means "X outranks Y" on
the stated dimension), and answers by graph reachability. Equality relations
build an undirected union-find structure.

It returns ``None`` (abstain) whenever the question or premises do not form a
clean comparison structure, so the caller falls through to other solvers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.logic.premise_selector import Premise
from app.logic._text_primitives import _norm


# Comparative adjective pairs: (greater_word, lesser_word). A premise
# "X is <greater> than Y" creates edge X -> Y (X outranks Y). A premise
# "X is <lesser> than Y" creates edge Y -> X. Antonyms are normalized to a
# canonical dimension so "taller"/"shorter" both map to the height order.
_COMPARATIVES: dict[str, tuple[str, int]] = {
    # word -> (dimension, direction)  direction +1 = subject outranks object
    "taller": ("height", +1),
    "shorter": ("height", -1),
    "larger": ("size", +1),
    "bigger": ("size", +1),
    "smaller": ("size", -1),
    "older": ("age", +1),
    "younger": ("age", -1),
    "faster": ("speed", +1),
    "slower": ("speed", -1),
    "heavier": ("weight", +1),
    "lighter": ("weight", -1),
    "stronger": ("strength", +1),
    "weaker": ("strength", -1),
    "richer": ("wealth", +1),
    "poorer": ("wealth", -1),
    "longer": ("length", +1),
    "greater": ("magnitude", +1),
    "less": ("magnitude", -1),
    "higher": ("rank", +1),
    "lower": ("rank", -1),
    "more expensive": ("cost", +1),
    "cheaper": ("cost", -1),
    "hotter": ("temperature", +1),
    "colder": ("temperature", -1),
    "warmer": ("temperature", +1),
    "cooler": ("temperature", -1),
}

# Superlative -> (dimension, want_max). "tallest" asks for the height maximum.
_SUPERLATIVES: dict[str, tuple[str, bool]] = {
    "tallest": ("height", True),
    "shortest": ("height", False),
    "largest": ("size", True),
    "biggest": ("size", True),
    "smallest": ("size", False),
    "oldest": ("age", True),
    "youngest": ("age", False),
    "fastest": ("speed", True),
    "slowest": ("speed", False),
    "heaviest": ("weight", True),
    "lightest": ("weight", False),
    "strongest": ("strength", True),
    "weakest": ("strength", False),
    "richest": ("wealth", True),
    "poorest": ("wealth", False),
    "longest": ("length", True),
    "shortest_length": ("length", False),
    "greatest": ("magnitude", True),
    "hottest": ("temperature", True),
    "coldest": ("temperature", False),
}


@dataclass
class _ComparisonGraph:
    """A directed comparison graph for one dimension.

    edges[x] is the set of y such that x outranks y (x > y) on this dimension.
    """

    dimension: str
    edges: dict[str, set[str]] = field(default_factory=dict)
    nodes: set[str] = field(default_factory=set)
    support: dict[tuple[str, str], Premise] = field(default_factory=dict)

    def add_edge(self, greater: str, lesser: str, premise: Premise) -> None:
        """Record that ``greater`` outranks ``lesser`` on this dimension."""
        self.nodes.add(greater)
        self.nodes.add(lesser)
        self.edges.setdefault(greater, set()).add(lesser)
        self.support[(greater, lesser)] = premise

    def reachable(self, start: str, goal: str) -> list[str] | None:
        """Return the chain of edges proving start > goal, or None.

        BFS over the directed graph. The returned list is the node path
        ``[start, ..., goal]`` so the caller can cite the supporting premises.
        """
        if start not in self.nodes or goal not in self.nodes:
            return None
        from collections import deque

        queue: deque[list[str]] = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            tail = path[-1]
            if tail == goal:
                return path
            for nxt in self.edges.get(tail, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None


_ENTITY_RE = r"([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)?|object\s+[A-Za-z0-9]+|[a-z]+)"


def _normalize_entity(name: str) -> str:
    """Normalize an entity token for matching (lowercase, strip articles).

    Also strips a leading generic descriptor noun ("object", "item", "person",
    "thing", "element") so "object X" in a question matches a bare "X" in the
    premises. General structural normalization, not a per-question text match.
    """
    n = name.strip().lower()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(
        r"^(?:object|item|person|thing|element|box|block|number|value|point)\s+",
        "",
        n,
    )
    return n.strip()


def _extract_comparison_edges(
    premises: list[Premise],
) -> dict[str, _ComparisonGraph]:
    """Build comparison graphs keyed by dimension from the premises.

    Recognizes "X is <comparative> than Y" for each comparative word. Antonyms
    flip the edge direction so all edges in a dimension point greater -> lesser.
    """
    graphs: dict[str, _ComparisonGraph] = {}
    # Build a regex alternation of comparative phrases (longest first).
    words = sorted(_COMPARATIVES.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in words)
    pattern = re.compile(
        rf"(.+?)\s+(?:is|are|was|were)\s+(?:much\s+|far\s+|a\s+lot\s+)?({alt})\s+than\s+(.+)",
        re.I,
    )
    for premise in premises:
        text = _norm(premise.text).rstrip(".")
        m = pattern.search(text)
        if not m:
            continue
        left = _normalize_entity(m.group(1))
        word = m.group(2).lower()
        right = _normalize_entity(m.group(3))
        if not left or not right:
            continue
        dimension, direction = _COMPARATIVES[word]
        graph = graphs.setdefault(dimension, _ComparisonGraph(dimension))
        if direction > 0:
            graph.add_edge(left, right, premise)  # left > right
        else:
            graph.add_edge(right, left, premise)  # right > left
    return graphs


def _extract_equality_groups(
    premises: list[Premise],
) -> tuple[dict[str, str], dict[tuple[str, str], Premise]]:
    """Union-find over "X equals Y" / "X is equal to Y" relations.

    Returns (parent_map, support_map). Two entities are equal iff they share a
    union-find root.
    """
    parent: dict[str, str] = {}
    support: dict[tuple[str, str], Premise] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pattern = re.compile(
        r"(.+?)\s+(?:is\s+)?equals?\s+(?:to\s+)?(.+)",
        re.I,
    )
    for premise in premises:
        text = _norm(premise.text).rstrip(".")
        m = pattern.search(text)
        if not m:
            continue
        left = _normalize_entity(m.group(1))
        right = _normalize_entity(m.group(2))
        if not left or not right:
            continue
        union(left, right)
        support[(left, right)] = premise
    return parent, support


def _extract_temporal_edges(premises: list[Premise]) -> _ComparisonGraph:
    """Build a temporal precedence graph from "X happened before Y" premises.

    Edge X -> Y means X precedes Y in time. "X after Y" adds Y -> X. Handles
    "X happened/occurred/came before/after Y" and bare "X before Y". Reachability
    then answers transitive "did X happen before Z?" queries.
    """
    graph = _ComparisonGraph("time")
    # "<X> (happened|occurred|came|was|is|...) before <Y>"  -> X precedes Y
    before_re = re.compile(
        r"^(.+?)\s+(?:happened|occurred|occured|came|took\s+place|was|is|comes?)?\s*before\s+(.+)$",
        re.I,
    )
    after_re = re.compile(
        r"^(.+?)\s+(?:happened|occurred|occured|came|took\s+place|was|is|comes?)?\s*after\s+(.+)$",
        re.I,
    )
    for premise in premises:
        text = _norm(premise.text).rstrip(".")
        m = before_re.search(text)
        if m:
            x = _normalize_temporal_entity(m.group(1))
            y = _normalize_temporal_entity(m.group(2))
            if x and y:
                graph.add_edge(x, y, premise)  # x precedes y
            continue
        m = after_re.search(text)
        if m:
            x = _normalize_temporal_entity(m.group(1))
            y = _normalize_temporal_entity(m.group(2))
            if x and y:
                graph.add_edge(y, x, premise)  # y precedes x
    return graph


def _normalize_temporal_entity(name: str) -> str:
    """Normalize a temporal entity ("event A", "the meeting") to a match key."""
    n = name.strip().lower()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(r"^(?:event|events|step|phase|stage|task)\s+", "", n)
    return n.strip()


def _question_temporal(question: str) -> tuple[str, str, bool] | None:
    """Parse "Did X happen before/after Y?" → (X, Y, is_before)."""
    low = _norm(question).rstrip("?.")
    m = re.search(
        r"did\s+(.+?)\s+(?:happen|occur|occured|come|take\s+place)?\s*(before|after)\s+(.+)$",
        low,
        re.I,
    )
    if not m:
        return None
    x = _normalize_temporal_entity(m.group(1))
    y = _normalize_temporal_entity(m.group(3))
    is_before = m.group(2).lower() == "before"
    if not x or not y:
        return None
    return x, y, is_before


def _question_comparison(question: str) -> tuple[str, str, str] | None:
    """Parse "Is X <comparative> than Y?" → (subject, comparative_word, object)."""
    words = sorted(_COMPARATIVES.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in words)
    m = re.search(
        rf"is\s+(.+?)\s+({alt})\s+than\s+(.+?)\??$",
        _norm(question).rstrip("?."),
        re.I,
    )
    if not m:
        return None
    return _normalize_entity(m.group(1)), m.group(2).lower(), _normalize_entity(m.group(3))


def _question_superlative(question: str) -> tuple[str, bool] | None:
    """Parse "Who/which is <superlative>?" → (dimension, want_max)."""
    low = _norm(question).rstrip("?.")
    for word, (dimension, want_max) in _SUPERLATIVES.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            return dimension, want_max
    return None


def _question_equality(question: str) -> tuple[str, str] | None:
    """Parse "Is X equal to Y?" → (X, Y)."""
    m = re.search(
        r"is\s+(.+?)\s+(?:equal\s+to|equals|the\s+same\s+as)\s+(.+?)\??$",
        _norm(question).rstrip("?."),
        re.I,
    )
    if not m:
        return None
    return _normalize_entity(m.group(1)), _normalize_entity(m.group(2))


def solve_comparison(
    question: str, premises: list[Premise]
) -> tuple[str, list[Premise], str] | None:
    """Solve a transitive comparison / ranking / equality question.

    Returns ``(answer, support_premises, reason)`` or ``None`` to abstain.
    Answers: "yes" / "no" for relational queries, the entity name for
    superlatives, or ``None`` when the structure is absent or indeterminate.
    """
    # 1. Equality transitivity: "Is A equal to C?"
    eq_q = _question_equality(question)
    if eq_q is not None:
        parent, support = _extract_equality_groups(premises)
        a, b = eq_q

        def find(x: str) -> str:
            seen = set()
            while parent.get(x, x) != x and x not in seen:
                seen.add(x)
                x = parent[x]
            return x

        if a in parent and b in parent and find(a) == find(b):
            return "yes", list(dict.fromkeys(support.values())), "transitive equality (A=B, B=C ⊢ A=C)"
        # If both entities are known but in different groups, we cannot assert.
        return None

    # 1b. Temporal precedence: "Did X happen before/after Y?"
    temp_q = _question_temporal(question)
    if temp_q is not None:
        x, y, is_before = temp_q
        graph = _extract_temporal_edges(premises)
        if not graph.nodes:
            return None
        # For "before": does x precede y (x -> ... -> y)? For "after": does y precede x?
        a, b = (x, y) if is_before else (y, x)
        forward = graph.reachable(a, b)
        if forward is not None:
            return "yes", _path_support(graph, forward), f"transitive temporal order ({' -> '.join(forward)})"
        backward = graph.reachable(b, a)
        if backward is not None:
            return "no", _path_support(graph, backward), f"reverse temporal order ({' -> '.join(backward)})"
        return None

    # 2. Relational comparison: "Is X taller than Y?"
    comp_q = _question_comparison(question)
    if comp_q is not None:
        subj, word, obj = comp_q
        dimension, direction = _COMPARATIVES[word]
        graphs = _extract_comparison_edges(premises)
        graph = graphs.get(dimension)
        if graph is None:
            return None
        # Normalize query direction to "does A outrank B?".
        a, b = (subj, obj) if direction > 0 else (obj, subj)
        forward = graph.reachable(a, b)
        if forward is not None:
            support = _path_support(graph, forward)
            return "yes", support, f"transitive comparison on {dimension} ({' > '.join(forward)})"
        backward = graph.reachable(b, a)
        if backward is not None:
            support = _path_support(graph, backward)
            return "no", support, f"reverse comparison on {dimension} ({' > '.join(backward)})"
        return None

    # 3. Superlative: "Who is tallest?"
    sup_q = _question_superlative(question)
    if sup_q is not None:
        dimension, want_max = sup_q
        graphs = _extract_comparison_edges(premises)
        graph = graphs.get(dimension)
        if graph is None or not graph.nodes:
            return None
        winner = _find_extremum(graph, want_max)
        if winner is None:
            return None
        # Cite all edges as support (the full chain establishing the order).
        support = list(dict.fromkeys(graph.support.values()))
        return winner.title(), support, f"extremum on {dimension} (unique {'max' if want_max else 'min'})"

    return None


def _path_support(graph: _ComparisonGraph, path: list[str]) -> list[Premise]:
    """Collect the premises backing each consecutive edge in a reachability path."""
    out: list[Premise] = []
    for i in range(len(path) - 1):
        premise = graph.support.get((path[i], path[i + 1]))
        if premise is not None and premise not in out:
            out.append(premise)
    return out


def _find_extremum(graph: _ComparisonGraph, want_max: bool) -> str | None:
    """Find the unique maximum (source) or minimum (sink) of the comparison order.

    For a max query, the answer is the unique node that outranks (reaches) every
    other node. For a min query, the unique node reachable from every other node.
    Returns None when no unique extremum exists (ambiguous order).
    """
    nodes = list(graph.nodes)
    if not nodes:
        return None

    def reaches_all(start: str) -> bool:
        # The start node must reach every other node via directed edges.
        from collections import deque

        visited = {start}
        queue: deque[str] = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in graph.edges.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited >= set(nodes)

    if want_max:
        candidates = [n for n in nodes if reaches_all(n)]
        return candidates[0] if len(candidates) == 1 else None

    # Min: build reverse reachability — the node reachable from all others.
    reverse = _ComparisonGraph(graph.dimension)
    for greater, lessers in graph.edges.items():
        for lesser in lessers:
            reverse.add_edge(lesser, greater, graph.support[(greater, lesser)])

    def reaches_all_reverse(start: str) -> bool:
        from collections import deque

        visited = {start}
        queue: deque[str] = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in reverse.edges.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited >= set(nodes)

    candidates = [n for n in nodes if reaches_all_reverse(n)]
    return candidates[0] if len(candidates) == 1 else None
