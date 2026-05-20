from __future__ import annotations

from app.logic.premise_selector import Premise


def logic_explanation(answer: str, selected: list[Premise], rule: str) -> str:
    evidence = "; ".join(f"{p.id}: {p.text}" for p in selected)
    return f"Answer is {answer}. Rule used: {rule}. Evidence: {evidence}."
