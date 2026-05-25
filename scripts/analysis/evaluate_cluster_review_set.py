from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.router import predict_with_metadata
from app.schemas import QARequest, TaskType, normalize_answer_label

DEFAULT_REVIEW_SET = ROOT / "reports" / "eval_case_cluster_review_set.jsonl"
DEFAULT_RESULTS = ROOT / "reports" / "eval_case_cluster_review_eval.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "eval_case_cluster_review_eval.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task(value: str | None) -> TaskType:
    try:
        return TaskType(value or "auto")
    except Exception:
        return TaskType.auto


def _answer_match(expected: str, predicted: str) -> bool:
    exp = normalize_answer_label(expected)
    pred = normalize_answer_label(predicted)
    if exp == pred:
        return True
    if exp in {"yes", "no", "unknown"} or pred in {"yes", "no", "unknown"}:
        return False
    expected_number = _first_number(exp)
    predicted_number = _first_number(pred)
    predicted_si = _first_number_si(pred)
    if expected_number is not None and predicted_number is not None:
        tolerance = max(1e-6, abs(expected_number) * 1e-6)
        if abs(expected_number - predicted_number) <= tolerance:
            return True
    if expected_number is not None and predicted_si is not None:
        tolerance = max(1e-9, abs(expected_number) * 1e-6)
        if abs(expected_number - predicted_si) <= tolerance:
            return True
    return exp.lower() in pred.lower() or pred.lower() in exp.lower()


def _first_number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", str(text), re.I)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _first_number_si(text: str) -> float | None:
    match = re.search(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*"
        r"(μc|µc|uc|mc|c|μf|µf|uf|mf|f|kv|mv|v|ma|a|kohm|kω|ohm|ω|mw|w|mj|j|mt|t)\b",
        str(text),
        re.I,
    )
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    unit = match.group(2).lower()
    factors = {
        "μc": 1e-6,
        "µc": 1e-6,
        "uc": 1e-6,
        "mc": 1e-3,
        "c": 1.0,
        "μf": 1e-6,
        "µf": 1e-6,
        "uf": 1e-6,
        "mf": 1e-3,
        "f": 1.0,
        "kv": 1e3,
        "mv": 1e-3,
        "v": 1.0,
        "ma": 1e-3,
        "a": 1.0,
        "kohm": 1e3,
        "kω": 1e3,
        "ohm": 1.0,
        "ω": 1.0,
        "mw": 1e-3,
        "w": 1.0,
        "mj": 1e-3,
        "j": 1.0,
        "mt": 1e-3,
        "t": 1.0,
    }
    return value * factors.get(unit, 1.0)


def _explanation_flags(row: dict[str, Any], explanation: str, response: Any) -> list[str]:
    flags: list[str] = []
    text = explanation.strip()
    low = text.lower()
    if len(text) < 30:
        flags.append("too_short")
    if ".." in text and "..." not in text:
        flags.append("double_punctuation")
    if "no deterministic formula matched the question. means" in low:
        flags.append("awkward_unknown_physics_wording")
    if response.task_type == "logic" and row.get("premises") and not response.premises:
        flags.append("no_selected_premises")
    if response.task_type == "logic" and response.answer == "unknown":
        if not any(token in low for token in ["missing", "not establish", "not prove", "contradict", "insufficient", "not enough", "unknown"]):
            flags.append("unknown_logic_not_specific")
    if response.task_type == "logic" and response.answer != "unknown":
        if "evidence:" not in low and not response.premises:
            flags.append("logic_no_evidence")
    if response.task_type == "physics":
        if response.answer != "unknown" and not response.fol:
            flags.append("physics_answer_without_formula")
        if response.answer != "unknown" and not any(token in low for token in ["used", "formula", "computed", "="]):
            flags.append("physics_no_computation_trace")
    return flags


def _run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        response, metadata = predict_with_metadata(
            QARequest(
                question=row["question"],
                premises=row.get("premises") or [],
                task_type=_task(row.get("task_type")),
            )
        )
        flags = _explanation_flags(row, response.explanation, response)
        expected = str(row.get("answer") or "")
        results.append(
            {
                "cluster_key": row.get("cluster_key"),
                "cluster_label": row.get("cluster_label"),
                "id": row.get("id"),
                "source": row.get("source"),
                "question": row.get("question"),
                "expected_answer": expected,
                "predicted_answer": response.answer,
                "answer_match": _answer_match(expected, response.answer),
                "task_type": response.task_type,
                "confidence": response.confidence,
                "premises_used": response.premises,
                "formula": response.fol,
                "explanation": response.explanation,
                "explanation_flags": flags,
                "normalization_warnings": metadata.get("normalization_warnings", []),
            }
        )
    return results


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    matches = sum(1 for row in rows if row["answer_match"])
    flagged = [row for row in rows if row["explanation_flags"]]
    answer_misses = [row for row in rows if not row["answer_match"]]
    by_flag: dict[str, int] = {}
    for row in flagged:
        for flag in row["explanation_flags"]:
            by_flag[flag] = by_flag.get(flag, 0) + 1

    lines = [
        "# Eval Case Cluster Review Eval",
        "",
        f"- Cases: `{total}`",
        f"- Answer matches: `{matches}/{total}`",
        f"- Explanation-flagged cases: `{len(flagged)}/{total}`",
        "",
        "## Explanation Flags",
        "",
    ]
    if by_flag:
        for flag, count in sorted(by_flag.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{flag}`: `{count}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Answer Misses", ""])
    if answer_misses:
        for row in answer_misses:
            lines.extend(
                [
                    f"### `{row['id']}` / `{row['cluster_label']}`",
                    "",
                    f"- Expected: `{row['expected_answer']}`",
                    f"- Predicted: `{row['predicted_answer']}`",
                    f"- Source: `{row['source']}`",
                    f"- Question: {row['question']}",
                    f"- Explanation: {row['explanation']}",
                    "",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Explanation Issues", ""])
    if flagged:
        for row in flagged:
            lines.extend(
                [
                    f"### `{row['id']}` / `{row['cluster_label']}`",
                    "",
                    f"- Flags: `{', '.join(row['explanation_flags'])}`",
                    f"- Expected/Predicted: `{row['expected_answer']}` / `{row['predicted_answer']}`",
                    f"- Question: {row['question']}",
                    f"- Explanation: {row['explanation']}",
                    "",
                ]
            )
    else:
        lines.append("- None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clustered representative cases with answer and explanation proxies.")
    parser.add_argument("--input", type=Path, default=DEFAULT_REVIEW_SET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    results = _run(rows)
    _write_results(args.results, results)
    _write_report(args.report, results)
    print(
        json.dumps(
            {
                "cases": len(results),
                "answer_matches": sum(1 for row in results if row["answer_match"]),
                "explanation_flagged": sum(1 for row in results if row["explanation_flags"]),
                "results": str(args.results),
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
