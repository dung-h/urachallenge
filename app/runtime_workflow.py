from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.guardrails import GuardrailResult, guardrail_prompt_text
from app.schemas import QARequest, QAResponse, TaskType


PHYSICS_HINTS = {
    "voltage", "current", "resistance", "resistor", "ohm", "power", "capacitor", "capacitance",
    "charge", "electric field", "force", "joule", "watt", "microfarad", "coulomb",
    "lc circuit", "rlc", "inductance", "inductor", "mh", "uf", "frequency", "angular frequency",
    "rad/s", "resonant", "resonance",
}


@dataclass(frozen=True)
class NormalizedRequest:
    original: QARequest
    question: str
    premises: list[str]
    choices: list[str]
    guardrail: GuardrailResult
    warnings: list[str] = field(default_factory=list)
    embedded_logic_extracted: bool = False
    embedded_premise_count: int = 0
    embedded_choice_count: int = 0

    def as_qa_request(self) -> QARequest:
        return self.original.model_copy(
            update={
                "question": self.question,
                "premises": self.premises,
                "choices": self.choices,
            }
        )


@dataclass(frozen=True)
class OrchestrationPlan:
    task_type: str | None
    route_reason: str
    confidence: float = 0.0
    use_search: bool = False
    use_llm_reasoner: bool = False
    use_explanation_rewrite: bool = False
    rescue_unknown: bool = False
    search_queries: list[str] = field(default_factory=list)
    physics_hint: dict[str, Any] = field(default_factory=dict)
    logic_hint: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "route_reason": self.route_reason,
            "confidence": self.confidence,
            "use_search": self.use_search,
            "use_llm_reasoner": self.use_llm_reasoner,
            "use_explanation_rewrite": self.use_explanation_rewrite,
            "rescue_unknown": self.rescue_unknown,
            "search_queries": list(self.search_queries),
            "physics_hint": dict(self.physics_hint),
            "logic_hint": dict(self.logic_hint),
            "source": self.source,
            "raw": dict(self.raw),
        }

    def task_enum(self) -> TaskType | None:
        if self.task_type in {TaskType.physics.value, TaskType.logic.value, TaskType.auto.value}:
            return TaskType(self.task_type)
        return None


class RuntimeTrace(BaseModel):
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_type: str
    solver_used: str | None = None
    input_question_original: str
    input_question_normalized: str
    normalized_question: str
    normalized_premises: list[str] = Field(default_factory=list)
    normalized_choices: list[str] = Field(default_factory=list)
    normalization_warnings: list[str] = Field(default_factory=list)
    input_guardrail_noise_detected: bool = False
    input_guardrail_noise_markers: list[str] = Field(default_factory=list)
    input_guardrail_removed_segments: list[str] = Field(default_factory=list)
    formula_id: str | None = None
    selected_premises: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_factors: dict[str, Any] = Field(default_factory=dict)
    physics_variables: dict[str, Any] = Field(default_factory=dict)
    proof_steps: list[dict[str, Any]] = Field(default_factory=list)
    proof_step_validity: bool | None = None
    proof_step_errors: list[str] = Field(default_factory=list)
    orchestration_plan: dict[str, Any] | None = None
    physics_problem_frame: dict[str, Any] | None = None
    physics_search_trace: list[dict[str, Any]] = Field(default_factory=list)
    physics_agent_session_id: str | None = None
    physics_agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    physics_agent_events: list[dict[str, Any]] = Field(default_factory=list)
    logic_agent_session_id: str | None = None
    logic_agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    logic_agent_events: list[dict[str, Any]] = Field(default_factory=list)
    explanation_trace: dict[str, Any] | None = None
    explanation_rewrite_accepted: bool = False
    explanation_rewrite_rejected: bool = False
    explanation_rewrite_validation_errors: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_accepted: bool = False
    fallback_rejected_reason: str | None = None
    model_calls: int = 0
    llm_client: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    llm_trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    answer: str
    explanation: str
    cot: list[str] = Field(default_factory=list)
    raw_json_validity: bool | None = None
    repaired_json_validity: bool | None = None


def looks_like_logic_prompt(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search(r"\bif\b.+\bthen\b", low, re.S)
        or "based only on the rules" in low
        or "rules state" in low
        or re.search(r"(?m)^\s*(?:all|no|some)\b", text, re.I)
        or re.search(r"(?m)^\s*(?:does|is|are|must)\b.+\?\s*$", text, re.I)
        or re.search(r"\b[A-E]\)\s+", text)
    )


def _strip_wrapping_quotes(text: str) -> str:
    return text.strip().strip("\"'“”‘’").strip()


def _strip_logic_line_prefix(text: str) -> tuple[str, str | None]:
    match = re.match(r"^(rule|premise|fact|observation|question)\s*:\s*(.*)$", text, flags=re.I)
    if not match:
        return text, None
    return match.group(2).strip(), match.group(1).lower()


def _looks_like_rule_premise(text: str) -> bool:
    return bool(
        re.match(r"^(?:if\b.+\bthen\b|all\b|no\b|some\b)", text, flags=re.I)
        or re.search(r"\b(?:requires?|need|needs|must have|only if|unless)\b", text, re.I)
        or re.search(r"\bstudents?\s+who\b.+\bmay\s+(?:register|receive|enroll|graduate|apply|take)\b", text, re.I)
    )


def _looks_like_fact_premise(text: str) -> bool:
    low = text.lower().rstrip(".")
    if low.endswith("?"):
        return False
    return bool(
        re.match(r"^.+?\s+(?:is|are|was|were)\s+(?:a |an |the )?.+$", low)
        or re.match(r"^.+?\s+(?:studies|registers|rings|fails|turns on|receives .+|has .+|can .+|completed .+|earned .+)$", low)
    )


def _looks_like_question(text: str) -> bool:
    return bool(re.match(r"^(?:does|is|are|did|must|may|will|would|which|what|who|when|where|can|could|should)\b.+\?\s*$", text, flags=re.I))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _extract_from_lines(question: str, premises: list[str], choices: list[str]) -> tuple[str, list[str], list[str], list[str], bool]:
    if premises:
        return question, premises, choices, [], False
    if (
        "\n" not in question
        and not re.search(r"\bp\d+\s*:", question, re.I)
        and not re.search(r"\b(?:question|rule|premise|fact|observation)\s*:", question, re.I)
        and not re.search(r"\b[A-E]\)\s+", question)
    ):
        return question, premises, choices, [], False

    extracted_premises: list[str] = []
    extracted_choices = list(choices)
    had_choices = bool(choices)
    question_lines: list[str] = []
    choice_lines: list[str] = []
    warnings: list[str] = []
    collecting_unlabeled_choices = False

    for raw_line in question.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean_line = _strip_wrapping_quotes(line)
        clean_line, prefix = _strip_logic_line_prefix(clean_line)
        if not clean_line:
            continue
        line_low = clean_line.lower()
        if prefix == "question":
            question_lines.append(clean_line)
            continue
        if prefix in {"rule", "premise", "fact", "observation"}:
            extracted_premises.append(clean_line.rstrip("."))
            continue
        premise_id_match = re.match(r"^(p\d+)\s*:\s*(.+)$", clean_line, re.I)
        if premise_id_match:
            extracted_premises.append(f"{premise_id_match.group(1).upper()}: {premise_id_match.group(2).strip().rstrip('.')}")
            continue
        if _looks_like_question(clean_line):
            question_lines.append(clean_line)
            continue
        if "which of the following" in line_low:
            collecting_unlabeled_choices = True
            question_lines.append(clean_line)
            continue
        numbered_match = re.match(r"^\d+[\.)]\s*(.+)$", clean_line)
        if numbered_match:
            candidate = _strip_wrapping_quotes(numbered_match.group(1)).rstrip(".")
            if candidate.endswith("?"):
                question_lines.append(candidate)
            elif collecting_unlabeled_choices and not had_choices:
                extracted_choices.append(candidate)
                choice_lines.append(f"{chr(ord('A') + len(extracted_choices) - 1)}) {candidate}")
            else:
                extracted_premises.append(candidate)
            continue
        choice_match = re.match(r"^([A-E])\)\s*(.+)$", clean_line, re.I)
        if choice_match:
            if not had_choices:
                extracted_choices.append(choice_match.group(2).strip())
                choice_lines.append(f"{choice_match.group(1).upper()}) {choice_match.group(2).strip()}")
            continue
        if collecting_unlabeled_choices and not had_choices and re.match(r"^if\b", clean_line, re.I):
            extracted_choices.append(clean_line)
            choice_lines.append(f"{chr(ord('A') + len(extracted_choices) - 1)}) {clean_line}")
            continue
        if _looks_like_rule_premise(clean_line) or (extracted_premises and _looks_like_fact_premise(clean_line)):
            extracted_premises.append(clean_line.rstrip("."))
            continue
        question_lines.append(clean_line)

    if not extracted_premises:
        return question, premises, choices, warnings, False

    compact_question = " ".join(question_lines)
    if choice_lines:
        compact_question = " ".join([compact_question, *choice_lines]).strip()
    warnings.append("embedded_logic_lines_extracted")
    return compact_question or question, extracted_premises, extracted_choices, warnings, True


def _extract_from_paragraph(question: str, premises: list[str], choices: list[str]) -> tuple[str, list[str], list[str], list[str], bool]:
    if premises or "\n" in question:
        return question, premises, choices, [], False
    low = question.lower()
    has_physics_signal = any(hint in low for hint in PHYSICS_HINTS) or bool(re.search(r"\b[0-9.]+\s*(v|a|ohm|ω|w|f|c|j|n)\b", low))
    has_strong_logic_signal = (
        bool(re.search(r"\bif\b.+\bthen\b", low))
        or bool(re.search(r"\b(?:eligible|requires?|need|needs|must have|only if|unless)\b", low))
        or bool(re.search(r"\bstudents?\s+who\b", low))
        or bool(re.search(r"\bmay\s+(?:register|receive|enroll|graduate|apply|take)\b", low))
    )
    has_logic_signal = has_strong_logic_signal or (bool(re.search(r"\b(?:all|no|some)\b", low)) and not has_physics_signal)
    if not has_logic_signal:
        return question, premises, choices, [], False
    sentences = _split_sentences(question)
    if len(sentences) < 2:
        return question, premises, choices, [], False

    question_index: int | None = None
    for index in range(len(sentences) - 1, -1, -1):
        if _looks_like_question(sentences[index]):
            question_index = index
            break
    if question_index is None:
        return question, premises, choices, [], False

    extracted: list[str] = []
    remainder: list[str] = []
    for index, sentence in enumerate(sentences):
        clean = sentence.strip().rstrip(".")
        if index == question_index:
            continue
        if _looks_like_rule_premise(clean) or _looks_like_fact_premise(clean):
            extracted.append(clean)
        else:
            remainder.append(sentence.strip())

    if not extracted:
        return question, premises, choices, [], False
    compact_question = " ".join([*remainder, sentences[question_index]]).strip()
    return compact_question or sentences[question_index], extracted, choices, ["paragraph_logic_premises_extracted"], True


class InputNormalizer:
    def normalize(self, request: QARequest) -> NormalizedRequest:
        guardrail = guardrail_prompt_text(request.question)
        question = guardrail.normalized_text
        premises = list(request.premises)
        choices = list(request.choices)
        warnings: list[str] = []

        question, premises, choices, line_warnings, line_changed = _extract_from_lines(question, premises, choices)
        warnings.extend(line_warnings)
        if not line_changed:
            question, premises, choices, paragraph_warnings, paragraph_changed = _extract_from_paragraph(question, premises, choices)
            warnings.extend(paragraph_warnings)
        else:
            paragraph_changed = False

        changed = line_changed or paragraph_changed
        return NormalizedRequest(
            original=request,
            question=question,
            premises=premises,
            choices=choices,
            guardrail=guardrail,
            warnings=warnings,
            embedded_logic_extracted=changed,
            embedded_premise_count=len(premises) if changed else 0,
            embedded_choice_count=len(choices) if changed else 0,
        )


class TaskRouter:
    def route(self, normalized: NormalizedRequest) -> TaskType:
        if normalized.original.task_type == TaskType.physics and looks_like_logic_prompt(normalized.question):
            return TaskType.logic
        if normalized.original.task_type != TaskType.auto:
            return normalized.original.task_type
        if normalized.premises:
            return TaskType.logic
        low = normalized.question.lower()
        if looks_like_logic_prompt(normalized.question):
            return TaskType.logic
        if any(hint in low for hint in PHYSICS_HINTS) or re.search(r"\b[0-9.]+\s*(v|a|ohm|ω|w|f|c|j|n)\b", low):
            return TaskType.physics
        return TaskType.logic


class LLMOrchestrator:
    def _heuristic_plan(self, normalized: NormalizedRequest) -> OrchestrationPlan:
        routed = TaskRouter().route(normalized)
        route_reason = "heuristic_router"
        if normalized.premises:
            route_reason = "heuristic_logic_premises"
        elif routed == TaskType.physics:
            route_reason = "heuristic_physics_signals"
        return OrchestrationPlan(
            task_type=routed.value,
            route_reason=route_reason,
            confidence=0.35 if routed == TaskType.physics else 0.3,
            use_search=routed == TaskType.physics,
            use_llm_reasoner=routed == TaskType.logic and bool(normalized.premises),
            use_explanation_rewrite=False,
            rescue_unknown=True,
            search_queries=[],
            physics_hint={},
            logic_hint={},
            source="heuristic",
            raw={},
        )

    def _normalize_plan(self, suggestion: dict[str, Any], fallback: OrchestrationPlan) -> OrchestrationPlan:
        def _bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                value = value.strip().lower()
                if value in {"1", "true", "yes", "y", "on"}:
                    return True
                if value in {"0", "false", "no", "n", "off"}:
                    return False
            if value is None:
                return default
            return bool(value)

        def _task(value: Any) -> str | None:
            if not isinstance(value, str):
                return None
            cleaned = value.strip().lower()
            if cleaned in {TaskType.physics.value, TaskType.logic.value, TaskType.auto.value}:
                return cleaned
            return None

        task_type = _task(suggestion.get("task_type")) or fallback.task_type
        route_reason = str(suggestion.get("route_reason") or suggestion.get("reason_short") or fallback.route_reason).strip()
        confidence = suggestion.get("confidence")
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = fallback.confidence
        search_queries = suggestion.get("search_queries")
        if not isinstance(search_queries, list):
            search_queries = []
        physics_hint = suggestion.get("physics_hint")
        logic_hint = suggestion.get("logic_hint")
        return OrchestrationPlan(
            task_type=task_type,
            route_reason=route_reason or fallback.route_reason,
            confidence=max(0.0, min(1.0, confidence_value)),
            use_search=_bool(suggestion.get("use_search"), fallback.use_search),
            use_llm_reasoner=_bool(suggestion.get("use_llm_reasoner"), fallback.use_llm_reasoner),
            use_explanation_rewrite=_bool(suggestion.get("use_explanation_rewrite"), fallback.use_explanation_rewrite),
            rescue_unknown=_bool(suggestion.get("rescue_unknown"), fallback.rescue_unknown),
            search_queries=[str(item).strip() for item in search_queries if str(item).strip()],
            physics_hint=physics_hint if isinstance(physics_hint, dict) else dict(fallback.physics_hint),
            logic_hint=logic_hint if isinstance(logic_hint, dict) else dict(fallback.logic_hint),
            source="llm",
            raw=suggestion,
        )

    def _build_payload(self, normalized: NormalizedRequest, fallback: OrchestrationPlan) -> dict[str, Any]:
        return {
            "question": normalized.question,
            "original_question": normalized.guardrail.original_text,
            "normalized_question": normalized.guardrail.normalized_text,
            "premises": list(normalized.premises),
            "choices": list(normalized.choices),
            "task_hint": normalized.original.task_type.value,
            "heuristic_plan": fallback.as_dict(),
            "guardrail_noise_detected": normalized.guardrail.noise_detected,
            "guardrail_markers": list(normalized.guardrail.noise_markers),
            "embedded_logic_extracted": normalized.embedded_logic_extracted,
            "physics_signals": {
                "has_physics_hints": bool(any(hint in normalized.question.lower() for hint in PHYSICS_HINTS)),
                "contains_units": bool(re.search(r"\b[0-9.]+\s*(v|a|ohm|ω|w|f|c|j|n)\b", normalized.question.lower())),
            },
            "logic_signals": {
                "has_premises": bool(normalized.premises),
                "looks_like_logic": looks_like_logic_prompt(normalized.question),
            },
            "available_actions": [
                "route_logic",
                "route_physics",
                "enable_search",
                "rescue_unknown",
                "rewrite_explanation",
            ],
        }

    def plan(self, normalized: NormalizedRequest, llm_client: Any | None = None) -> OrchestrationPlan:
        import os
        fallback = self._heuristic_plan(normalized)
        orchestrate = getattr(llm_client, "orchestrate", None) if llm_client is not None else None
        if not callable(orchestrate):
            return fallback

        allow_fallback = os.environ.get("URA_ALLOW_HEURISTIC_FALLBACK") == "1"

        try:
            suggestion = orchestrate(self._build_payload(normalized, fallback))
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"LLM Orchestration server call failed: {exc}") from exc
            fallback_plan = OrchestrationPlan(
                task_type=fallback.task_type,
                route_reason=f"heuristic_fallback_from_llm_error: {exc}",
                confidence=fallback.confidence,
                use_search=fallback.use_search,
                use_llm_reasoner=fallback.use_llm_reasoner,
                use_explanation_rewrite=fallback.use_explanation_rewrite,
                rescue_unknown=fallback.rescue_unknown,
                search_queries=fallback.search_queries,
                physics_hint=fallback.physics_hint,
                logic_hint=fallback.logic_hint,
                source="heuristic_fallback",
                raw={"error": str(exc)},
            )
            return fallback_plan

        # Retry once for JSON validation error if the server is live (status == "ok")
        if suggestion is None:
            last_trace = llm_client.call_traces[-1] if hasattr(llm_client, "call_traces") and llm_client.call_traces else None
            if last_trace and last_trace.get("status") == "ok":
                try:
                    suggestion = orchestrate(self._build_payload(normalized, fallback))
                except Exception:
                    suggestion = None

        if suggestion is None:
            last_trace = llm_client.call_traces[-1] if hasattr(llm_client, "call_traces") and llm_client.call_traces else None
            if last_trace and last_trace.get("status") == "error":
                err_msg = last_trace.get("error") or "Unknown connection error"
                if not allow_fallback:
                    raise ConnectionError(f"vLLM/OpenAI endpoint connection failed: {err_msg}")
                fallback_plan = OrchestrationPlan(
                    task_type=fallback.task_type,
                    route_reason=f"heuristic_fallback_from_connection_error: {err_msg}",
                    confidence=fallback.confidence,
                    use_search=fallback.use_search,
                    use_llm_reasoner=fallback.use_llm_reasoner,
                    use_explanation_rewrite=fallback.use_explanation_rewrite,
                    rescue_unknown=fallback.rescue_unknown,
                    search_queries=fallback.search_queries,
                    physics_hint=fallback.physics_hint,
                    logic_hint=fallback.logic_hint,
                    source="heuristic_fallback",
                    raw={"error": err_msg},
                )
                return fallback_plan

            # Safe deterministic route with warning for invalid JSON
            return OrchestrationPlan(
                task_type=fallback.task_type,
                route_reason="heuristic_fallback_from_invalid_json",
                confidence=fallback.confidence,
                use_search=fallback.use_search,
                use_llm_reasoner=fallback.use_llm_reasoner,
                use_explanation_rewrite=fallback.use_explanation_rewrite,
                rescue_unknown=fallback.rescue_unknown,
                search_queries=fallback.search_queries,
                physics_hint=fallback.physics_hint,
                logic_hint=fallback.logic_hint,
                source="heuristic_after_invalid_json",
                raw={"error": "LLM Orchestrator returned empty or invalid JSON suggestion"},
            )

        if not isinstance(suggestion, dict):
            return OrchestrationPlan(
                task_type=fallback.task_type,
                route_reason="heuristic_fallback_from_invalid_dict",
                confidence=fallback.confidence,
                use_search=fallback.use_search,
                use_llm_reasoner=fallback.use_llm_reasoner,
                use_explanation_rewrite=fallback.use_explanation_rewrite,
                rescue_unknown=fallback.rescue_unknown,
                search_queries=fallback.search_queries,
                physics_hint=fallback.physics_hint,
                logic_hint=fallback.logic_hint,
                source="heuristic_after_invalid_json",
                raw={"error": "LLM Orchestrator suggestion was not a dictionary"},
            )

        return self._normalize_plan(suggestion, fallback)


def build_runtime_trace(
    request_id: str,
    normalized: NormalizedRequest,
    response: QAResponse,
    metadata: dict[str, Any],
) -> RuntimeTrace:
    return RuntimeTrace(
        request_id=request_id,
        task_type=response.task_type,
        solver_used=metadata.get("solver_used"),
        input_question_original=normalized.guardrail.original_text,
        input_question_normalized=normalized.guardrail.normalized_text,
        normalized_question=normalized.question,
        normalized_premises=list(normalized.premises),
        normalized_choices=list(normalized.choices),
        normalization_warnings=list(normalized.warnings),
        input_guardrail_noise_detected=normalized.guardrail.noise_detected,
        input_guardrail_noise_markers=list(normalized.guardrail.noise_markers),
        input_guardrail_removed_segments=list(normalized.guardrail.removed_segments),
        formula_id=response.fol,
        selected_premises=list(response.premises),
        confidence=response.confidence,
        confidence_factors=dict(metadata.get("confidence_factors") or {}),
        physics_variables=dict(metadata.get("physics_variables") or {}),
        proof_steps=list(metadata.get("proof_steps") or []),
        proof_step_validity=metadata.get("proof_step_validity"),
        proof_step_errors=list(metadata.get("proof_step_errors") or []),
        orchestration_plan=metadata.get("orchestration_plan"),
        physics_problem_frame=metadata.get("physics_problem_frame"),
        physics_search_trace=list(metadata.get("physics_search_trace") or []),
        physics_agent_trace=list(metadata.get("physics_agent_trace") or []),
        physics_agent_events=list(metadata.get("physics_agent_events") or []),
        logic_agent_trace=list(metadata.get("logic_agent_trace") or []),
        logic_agent_events=list(metadata.get("logic_agent_events") or []),
        explanation_trace=metadata.get("explanation_trace"),
        explanation_rewrite_accepted=bool(metadata.get("explanation_rewrite_accepted", False)),
        explanation_rewrite_rejected=bool(metadata.get("explanation_rewrite_rejected", False)),
        explanation_rewrite_validation_errors=list(metadata.get("explanation_rewrite_validation_errors") or []),
        fallback_used=bool(metadata.get("fallback_used", False)),
        fallback_accepted=bool(metadata.get("fallback_accepted", False)),
        fallback_rejected_reason=metadata.get("fallback_rejected_reason"),
        model_calls=int(metadata.get("model_calls", 0) or 0),
        llm_client=dict(metadata.get("llm_client") or {"enabled": False}),
        llm_trace=list(metadata.get("llm_trace") or []),
        latency_ms=float(metadata.get("latency_ms", 0.0) or 0.0),
        answer=response.answer,
        explanation=response.explanation,
        cot=list(response.cot),
        raw_json_validity=response.raw_json_validity,
        repaired_json_validity=response.repaired_json_validity,
    )
