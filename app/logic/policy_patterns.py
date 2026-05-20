from __future__ import annotations

import re


ACADEMIC_KEYWORDS = {
    "scholarship", "gpa", "cpa", "credits", "disciplinary", "academic warning", "register",
    "registration", "prerequisite", "exam", "attendance", "absence", "retake", "improvement",
    "graduation", "graduate", "financial aid", "tuition", "fee hold", "bursary", "exchange",
    "course", "advisor approval", "medical certificate",
}

STOPWORDS = {"the", "a", "an", "is", "are", "for", "to", "may", "does", "has", "have", "with", "student", "students", "who", "all", "no"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def tokens(text: str) -> set[str]:
    normalized = set()
    for token in re.findall(r"[a-z0-9]+", norm(text)):
        if token in STOPWORDS or len(token) <= 1:
            continue
        if token in {"meets", "met"}:
            token = "meet"
        elif token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        normalized.add(token)
    return normalized


def is_academic_policy_text(question: str, premises: list[str]) -> bool:
    text = norm(question + " " + " ".join(premises))
    return any(keyword in text for keyword in ACADEMIC_KEYWORDS)


def overlap(left: str, right: str) -> int:
    return len(tokens(left) & tokens(right))


def strip_student_prefix(text: str) -> str:
    low = norm(text)
    low = re.sub(r"^(all|no)\s+students?\s+", "", low)
    low = re.sub(r"^students?\s+", "", low)
    return low
