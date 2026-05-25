from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GENERIC_UNKNOWN_PHRASES = (
    "not enough information",
    "do not provide enough information",
    "do not form a complete chain",
    "not form a complete chain",
    "the premises are relevant, but",
    "the selected premises are relevant, but",
    "no deterministic entailment rule matched",
    "no deterministic formula matched",
    "missing condition or supporting rule",
    "a missing condition or supporting rule is still needed",
    "the answer is unknown because",
    "the question could not be translated safely",
    "insufficient evidence",
)

GENERIC_UNKNOWN_NOTES = (
    "no deterministic entailment rule matched",
    "selected premises are relevant, but they do not form a complete chain to the conclusion",
    "the premises are relevant, but they do not provide enough information to prove the conclusion",
    "missing academic warning threshold fact",
    "missing required condition",
    "unknown due to insufficient evidence",
)

GENERIC_PHYSICS_UNKNOWN_PHRASES = (
    "no deterministic formula matched",
    "could not map the given quantities",
    "the answer is unknown because",
    "could not reduce the question",
)

DEFAULT_SOURCE_ROOTS = (
    Path("datasets/eval"),
    Path("datasets/synthetic"),
    Path("outputs/traces/production_like"),
)


@dataclass(frozen=True)
class SuspiciousCase:
    source_path: Path
    request_id: str
    task_type: str
    answer: str
    score: int
    reasons: list[str]
    trace: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = dict(self.trace)
        record.update(
            {
                "source_path": str(self.source_path),
                "suspicion_score": self.score,
                "suspicion_reasons": list(self.reasons),
            }
        )
        return record


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _safe_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "disagreement"


def iter_json_records(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                yield from iter_json_records([child])
            for child in sorted(path.rglob("*.jsonl")):
                yield from iter_json_records([child])
            continue
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yield path, json.loads(line)
        elif path.suffix.lower() == ".json":
            yield path, json.loads(path.read_text(encoding="utf-8"))


def canonical_question(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    for line in raw.splitlines():
        match = re.match(r"^question\s*:\s*(.+)$", line.strip(), flags=re.I)
        if match:
            return _norm(match.group(1))
    first_line = raw.splitlines()[0]
    return _norm(first_line)


def extract_embedded_logic_prompt(text: Any) -> tuple[str, list[str]]:
    raw = str(text or "").strip()
    if not raw:
        return "", []
    question_lines: list[str] = []
    premises: list[str] = []
    for line in raw.splitlines():
        clean = line.strip()
        if not clean:
            continue
        match = re.match(r"^question\s*:\s*(.+)$", clean, flags=re.I)
        if match:
            question_lines.append(match.group(1).strip())
            continue
        match = re.match(r"^(p\d+)\s*:\s*(.+)$", clean, flags=re.I)
        if match:
            premises.append(f"{match.group(1).upper()}: {match.group(2).strip()}")
            continue
        if clean.lower().startswith(("is ", "does ", "are ", "may ", "can ", "what ", "which ")):
            question_lines.append(clean)
    question = " ".join(question_lines).strip() if question_lines else raw.splitlines()[0].strip()
    return question, premises


def _looks_like_question_line(text: str) -> bool:
    return bool(re.match(r"^(?:does|is|are|did|must|which|what)\b.+\?\s*$", text, flags=re.I))


def extract_natural_language_logic_prompt(text: Any) -> tuple[str, list[str]]:
    raw = str(text or "").strip()
    if not raw:
        return "", []

    embedded_question, embedded_premises = extract_embedded_logic_prompt(raw)
    if embedded_premises:
        return embedded_question, embedded_premises

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return "", []

    if len(lines) == 1:
        chunks = re.split(r"(?<=[.!?])\s+", lines[0])
    else:
        chunks = []
        for line in lines:
            chunks.extend(re.split(r"(?<=[.!?])\s+", line))

    chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not chunks:
        return "", []

    question_parts: list[str] = []
    premises: list[str] = []
    for chunk in chunks:
        clean = chunk.strip().strip("\"'“”‘’").strip()
        if not clean:
            continue
        if _looks_like_question_line(clean) or clean.endswith("?"):
            question_parts.append(clean.rstrip())
            continue
        cleaned = clean.rstrip(".")
        premises.append(cleaned)

    if question_parts and premises:
        return " ".join(question_parts).strip(), premises
    return "", []


def _proof_notes(trace: dict[str, Any]) -> str:
    notes: list[str] = []
    for step in trace.get("proof_steps") or []:
        if isinstance(step, dict):
            note = str(step.get("notes") or "").strip()
            if note:
                notes.append(note)
    return " ".join(notes).strip().lower()


def _explanation_text(trace: dict[str, Any]) -> str:
    return _norm(trace.get("explanation") or "")


def _reason(score_parts: list[tuple[int, str]], score: int, reason: str) -> None:
    score_parts.append((score, reason))


def score_trace(trace: dict[str, Any]) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    answer = _norm(trace.get("answer"))
    task_type = _norm(trace.get("task_type"))
    explanation = _explanation_text(trace)
    notes = _proof_notes(trace)
    proof_valid = trace.get("proof_step_validity")
    proof_errors = trace.get("proof_step_errors") or []
    selected_premises = trace.get("selected_premises") or []
    fallback_reason = _norm(trace.get("fallback_rejected_reason"))

    if proof_valid is False:
        score += 35
        reasons.append("invalid_proof_steps")
    if proof_errors:
        score += min(20 + 5 * len(proof_errors), 30)
        reasons.append("proof_step_errors")
    if fallback_reason:
        if "validation_failed" in fallback_reason or "error" in fallback_reason:
            score += 20
            reasons.append(f"fallback_reject:{fallback_reason}")
        elif "no_json" in fallback_reason or "no_proposal" in fallback_reason:
            score += 10
            reasons.append(f"fallback_reject:{fallback_reason}")

    if trace.get("explanation_rewrite_rejected"):
        score += 15
        reasons.append("explanation_rewrite_rejected")
    if trace.get("explanation_rewrite_accepted"):
        score += 5
        reasons.append("llm_explanation_accepted")

    if task_type == "logic":
        if not selected_premises:
            score += 25
            reasons.append("logic_without_selected_premises")
        if answer == "unknown":
            if any(phrase in explanation for phrase in GENERIC_UNKNOWN_PHRASES):
                score += 20
                reasons.append("generic_unknown_explanation")
            if any(phrase in notes for phrase in GENERIC_UNKNOWN_NOTES):
                score += 20
                reasons.append("generic_unknown_proof_note")
            if not proof_errors and not proof_valid and not selected_premises:
                score += 10
                reasons.append("unknown_without_trace_support")
        elif any(phrase in explanation for phrase in GENERIC_UNKNOWN_PHRASES):
            score += 15
            reasons.append("answer_explanation_mismatch")
    elif task_type == "physics":
        formula_id = trace.get("formula_id")
        if answer == "unknown":
            if any(phrase in explanation for phrase in GENERIC_PHYSICS_UNKNOWN_PHRASES):
                score += 20
                reasons.append("generic_physics_unknown_explanation")
            if not formula_id:
                score += 10
                reasons.append("physics_without_formula_id")
        else:
            if not formula_id:
                score += 10
                reasons.append("physics_answer_without_formula_id")

    if trace.get("model_calls", 0):
        score += 5
        reasons.append("model_calls_present")
    if trace.get("solver_used") and _norm(trace.get("solver_used")) not in {"deterministic", "physics", "logic"}:
        score += 5
        reasons.append(f"non_default_solver:{trace.get('solver_used')}")

    return score, reasons


def collect_suspicious_cases(paths: Iterable[Path], min_score: int = 20) -> list[SuspiciousCase]:
    cases: list[SuspiciousCase] = []
    for source_path, trace in iter_json_records(paths):
        score, reasons = score_trace(trace)
        if score < min_score:
            continue
        request_id = str(trace.get("request_id") or source_path.stem or "unknown")
        cases.append(
            SuspiciousCase(
                source_path=source_path,
                request_id=request_id,
                task_type=str(trace.get("task_type") or "unknown"),
                answer=str(trace.get("answer") or "unknown"),
                score=score,
                reasons=reasons,
                trace=trace,
            )
        )
    cases.sort(key=lambda item: (item.score, item.request_id), reverse=True)
    return cases


def _tokenize_keywords(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", _norm(text)):
        if len(token) < 3:
            continue
        if token in {"the", "and", "for", "with", "that", "this", "from", "rule", "used", "because", "answer"}:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _load_source_index(source_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not source_root.exists():
        return index
    for path in sorted(source_root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                q = canonical_question(row.get("question") or row.get("input_question_original") or row.get("prompt"))
                t = _norm(row.get("task_type") or "unknown")
                if q and (q, t) not in index:
                    index[(q, t)] = row
    return index


def _extract_logic_prompt_from_llm_trace(trace: dict[str, Any]) -> tuple[str, list[str], str]:
    for entry in trace.get("llm_trace") or []:
        if not isinstance(entry, dict):
            continue
        user_prompt = str(entry.get("user_prompt") or "").strip()
        if not user_prompt:
            continue
        question, premises = extract_embedded_logic_prompt(user_prompt)
        if question or premises:
            return question, premises, "llm_trace.user_prompt"
    return "", [], ""


def enrich_trace_for_skeleton(
    trace: dict[str, Any],
    source_root: Path | None = None,
    source_index: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    enriched = dict(trace)
    reconstruction_sources: list[str] = []
    question = str(trace.get("question") or trace.get("input_question_original") or trace.get("input_question_normalized") or "").strip()
    task_type = str(trace.get("task_type") or "unknown")

    premises = list(trace.get("premises") or [])
    if not premises and task_type == "logic":
        embedded_question, embedded_premises = extract_embedded_logic_prompt(trace.get("input_question_original") or "")
        if embedded_premises:
            question = embedded_question or question
            premises = embedded_premises
            reconstruction_sources.append("input_question_original")

    if task_type == "logic" and (not question or not premises):
        for source_text, source_label in (
            (trace.get("input_question_original"), "input_question_original"),
            (trace.get("question"), "question"),
            (trace.get("input_question_normalized"), "input_question_normalized"),
        ):
            parsed_question, parsed_premises = extract_natural_language_logic_prompt(source_text)
            if parsed_question or parsed_premises:
                reconstruction_sources.append(f"parsed:{source_label}")
            if parsed_question and not question:
                question = parsed_question
            if parsed_premises and not premises:
                premises = parsed_premises
            if question and premises:
                break

    if task_type == "logic" and (not question or not premises):
        llm_question, llm_premises, llm_source = _extract_logic_prompt_from_llm_trace(trace)
        if llm_source:
            reconstruction_sources.append(llm_source)
        if llm_question and not question:
            question = llm_question
        if llm_premises and not premises:
            premises = llm_premises

    source_row = None
    if source_index is None and source_root is not None and source_root.exists():
        source_index = _load_source_index(source_root)
    if source_index:
        key = (canonical_question(question), _norm(task_type))
        source_row = source_index.get(key)

    if source_row:
        question = str(source_row.get("question") or question).strip() or question
        if not premises:
            premises = list(source_row.get("premises") or [])
        enriched["expected_answer"] = source_row.get("expected_answer") or trace.get("answer")
        enriched["source_row"] = source_row
        reconstruction_sources.append("source_dataset")
    else:
        enriched["expected_answer"] = trace.get("answer")

    if not premises and task_type == "logic":
        premises = list(trace.get("selected_premise_texts") or [])

    enriched["question_for_test"] = question
    enriched["premises_for_test"] = premises
    enriched["reconstruction_sources"] = reconstruction_sources
    skip_reasons: list[str] = []
    if not question:
        skip_reasons.append("missing_question")
    if task_type == "logic" and not premises:
        skip_reasons.append("unparsed_premises")
    enriched["skip_reasons"] = skip_reasons
    return enriched


def _assertion_keywords(trace: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    notes = _proof_notes(trace)
    explanation = _explanation_text(trace)
    premises = [str(premise) for premise in trace.get("premises_for_test") or []]
    question_tokens = set(_tokenize_keywords(str(trace.get("question_for_test") or "")))
    for premise in premises:
        for word in _tokenize_keywords(premise):
            if word in question_tokens:
                continue
            if word in {"student", "students", "eligible", "scholarship", "register", "answer", "question"}:
                continue
            if word not in keywords:
                keywords.append(word)
    if len(keywords) < 2:
        for source in [notes, explanation]:
            for word in _tokenize_keywords(source):
                if word not in keywords:
                    keywords.append(word)
    if trace.get("answer") == "unknown" and "missing" not in keywords:
        keywords.append("missing")
    if trace.get("task_type") == "physics" and trace.get("formula_id"):
        formula_id = str(trace.get("formula_id"))
        for token in _tokenize_keywords(formula_id.replace("_", " ")):
            if token not in keywords:
                keywords.append(token)
    return keywords[:4]


def _preview_text(text: Any, limit: int = 96) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "…"


def _trace_question_preview(trace: dict[str, Any]) -> str:
    question = str(trace.get("question_for_test") or trace.get("question") or trace.get("input_question_original") or "").strip()
    if question:
        return question
    for entry in trace.get("llm_trace") or []:
        if isinstance(entry, dict):
            user_prompt = str(entry.get("user_prompt") or "").strip()
            if user_prompt:
                question, _ = extract_embedded_logic_prompt(user_prompt)
                if question:
                    return question
                return _preview_text(user_prompt)
    return ""


def _trace_premises_preview(trace: dict[str, Any]) -> str:
    premises = [str(premise).strip() for premise in (trace.get("premises_for_test") or trace.get("selected_premises") or []) if str(premise).strip()]
    if premises:
        return " | ".join(_preview_text(premise, 72) for premise in premises[:3])
    return ""


def render_pytest_skeleton(records: list[dict[str, Any]], source_root: Path | None = None) -> str:
    accepted_records, _ = build_generation_plan(records, source_root=source_root)
    lines = [
        "from app.logic.solver import solve as solve_logic",
        "from app.physics.solver import solve as solve_physics",
        "",
    ]
    for idx, enriched in enumerate(accepted_records, start=1):
        task_type = str(enriched.get("task_type") or "unknown")
        question = str(enriched.get("question_for_test") or "").strip()
        premises = list(enriched.get("premises_for_test") or [])
        expected_answer = str(enriched.get("expected_answer") or enriched.get("answer") or "unknown")
        keywords = _assertion_keywords(enriched)
        reasons = ", ".join(str(reason) for reason in enriched.get("suspicion_reasons") or [])
        func_name = f"test_disagreement_{_safe_id(str(enriched.get('request_id') or idx))}"
        solver_name = "solve_logic" if task_type == "logic" else "solve_physics"

        lines.append(f"def {func_name}() -> None:")
        lines.append(f"    # Suspicion score: {enriched.get('suspicion_score', '?')}")
        reconstruction_sources = ", ".join(str(source) for source in enriched.get("reconstruction_sources") or [])
        if reconstruction_sources:
            lines.append(f"    # Reconstructed from: {reconstruction_sources}")
        if reasons:
            lines.append(f"    # Reasons: {reasons}")
        lines.append(f"    result = {solver_name}(")
        lines.append(f"        {question!r},")
        if task_type == "logic":
            lines.append("        [")
            for premise in premises:
                lines.append(f"            {premise!r},")
            lines.append("        ],")
        lines.append("    )")
        lines.append(f"    assert result.answer == {expected_answer!r}")
        if task_type == "physics" and enriched.get("formula_id"):
            lines.append(f"    assert result.fol == {str(enriched.get('formula_id'))!r}")
        for keyword in keywords[:3]:
            lines.append(f"    assert {keyword!r} in result.explanation.lower()")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_generation_plan(
    records: list[dict[str, Any]],
    source_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_index = _load_source_index(source_root) if source_root is not None else None
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for trace in records:
        enriched = enrich_trace_for_skeleton(trace, source_root=source_root, source_index=source_index)
        if enriched.get("skip_reasons"):
            skipped.append(enriched)
        else:
            accepted.append(enriched)
    return accepted, skipped


def render_generation_report(
    records: list[dict[str, Any]],
    source_root: Path | None = None,
) -> str:
    accepted, skipped = build_generation_plan(records, source_root=source_root)
    lines = [
        "# Disagreement Generation Report",
        "",
        f"- Input records: {len(records)}",
        f"- Accepted skeletons: {len(accepted)}",
        f"- Skipped records: {len(skipped)}",
        "",
    ]
    if accepted:
        lines.extend([
            "## Accepted",
            "",
            "| request_id | task | score | question | premises preview | reconstructed from |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for record in accepted:
            lines.append(
                f"| {record.get('request_id', '')} | {record.get('task_type', '')} | {record.get('suspicion_score', '')} | "
                f"{_preview_text(_trace_question_preview(record)) or '-'} | "
                f"{_preview_text(_trace_premises_preview(record), 120) or '-'} | "
                f"{', '.join(str(item) for item in record.get('reconstruction_sources') or []) or '-'} |"
            )
        lines.append("")
    if skipped:
        lines.extend([
            "## Skipped",
            "",
            "| request_id | task | score | question preview | skip reasons | reconstruction sources |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for record in skipped:
            lines.append(
                f"| {record.get('request_id', '')} | {record.get('task_type', '')} | {record.get('suspicion_score', '')} | "
                f"{_preview_text(_trace_question_preview(record)) or '-'} | "
                f"{', '.join(str(item) for item in record.get('skip_reasons') or []) or '-'} | "
                f"{', '.join(str(item) for item in record.get('reconstruction_sources') or []) or '-'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
