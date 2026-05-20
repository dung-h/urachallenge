from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    auto = "auto"
    physics = "physics"
    logic = "logic"


def normalize_answer_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[`*_\"']", "", text)
    text = re.sub(r"\s+", " ", text)
    option = re.match(r"^(?:option\s*)?([a-e])(?:[\).:]|$)", text)
    if option:
        return option.group(1).upper()
    if any(token in text for token in ["cannot be determined", "cannot determine", "not enough", "unknown", "undetermined", "cannot conclude", "not necessarily", "insufficient"]):
        return "unknown"
    if text in {"true", "yes", "y"} or text.startswith("yes"):
        return "yes"
    if text in {"false", "no", "n"} or text.startswith("no"):
        return "no"
    return text


class PhysicsInput(BaseModel):
    question: str = Field(min_length=1)


class LogicInput(BaseModel):
    question: str = Field(min_length=1)
    premises: list[str] = Field(default_factory=list)


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    premises: list[str] = Field(default_factory=list)
    premises_fol: list[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.auto
    choices: list[str] = Field(default_factory=list)
    allow_llm_fallback: bool = False
    request_id: str | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        return value.strip()

    @field_validator("request_id")
    @classmethod
    def strip_request_id(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("premises", "choices")
    @classmethod
    def strip_lists(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if str(v).strip()]


class QAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    premises: list[str] = Field(default_factory=list)
    cot: list[str] = Field(default_factory=list)
    fol: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    task_type: Literal["physics", "logic", "unknown"] = "unknown"
    raw_json_validity: bool | None = None
    repaired_json_validity: bool | None = None

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        normalized = normalize_answer_label(value)
        if normalized in {"yes", "no", "unknown"} or re.fullmatch(r"[A-E]", normalized):
            return normalized
        return str(value).strip()

    @model_validator(mode="after")
    def ensure_confidence_range(self) -> "QAResponse":
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        return self


class TraceRecord(BaseModel):
    step: str
    detail: str
