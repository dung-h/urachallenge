from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Premise:
    id: str
    text: str


STOPWORDS = {"the", "a", "an", "is", "are", "does", "do", "if", "then", "all", "some", "no", "of", "in", "on", "to", "and", "or"}


def normalize_premises(premises: list[str]) -> list[Premise]:
    normalized: list[Premise] = []
    for idx, premise in enumerate(premises, start=1):
        text = premise.strip()
        match = re.match(r"^(P\d+)\s*:\s*(.+)$", text, re.I)
        if match:
            normalized.append(Premise(match.group(1).upper(), match.group(2).strip()))
        else:
            normalized.append(Premise(f"P{idx}", text))
    return normalized


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1}


def select_premises(question: str, premises: list[Premise], limit: int | None = None) -> list[Premise]:
    q_tokens = tokens(question)
    scored = []
    for premise in premises:
        p_tokens = tokens(premise.text)
        score = len(q_tokens & p_tokens)
        if any(t in premise.text.lower() for t in ["if", "all", "no", "some"]):
            score += 0.5
        scored.append((score, premise))
    selected = [p for score, p in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    if not selected:
        selected = premises[:]
    return selected[:limit] if limit else selected


def hallucinated_premises(cited_ids: set[str], premises: list[Premise]) -> set[str]:
    allowed = {p.id.upper() for p in premises}
    return {pid.upper() for pid in cited_ids} - allowed
