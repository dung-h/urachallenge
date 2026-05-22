from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.scorers import gold_answer, score_response
from app.router import predict_response
from app.schemas import QARequest


DEFAULT_DATASETS = [
    "datasets/eval/exact_style_academic_policy_mock.jsonl",
    "datasets/eval/adversarial_logic.jsonl",
    "datasets/eval/hardcase_academic_policy_qualitative.jsonl",
    "datasets/eval/exact_style_physics_mock.jsonl",
    "datasets/eval/adversarial_physics.jsonl",
    "datasets/eval/hardcase_physics_qualitative.jsonl",
    "datasets/eval/hardcase_unknown_refusal.jsonl",
    "datasets/eval/phase_12_academic_policy_failures_regression.jsonl",
    "datasets/eval/phase_16_safety_failures_regression.jsonl",
    "datasets/eval/phase_24_alt_approach_eval.jsonl",
    "datasets/eval/phase_27_symbolic_model_search.jsonl",
    "datasets/eval/phase_9_failures_regression.jsonl",
    "datasets/eval/regression_from_errors.jsonl",
    "datasets/eval/schema_robustness.jsonl",
    "datasets/synthetic/mini_benchmark.jsonl",
]


def _iter_rows(paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if gold_answer(row) is None:
                continue
            yield path, row


def _bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "answer_correct": 0,
        "explanation_consistent": 0,
        "hallucinated_premise": 0,
        "latency_ms": 0.0,
        "premise_precision_sum": 0.0,
        "premise_precision_n": 0,
        "premise_recall_sum": 0.0,
        "premise_recall_n": 0,
        "formula_correct": 0,
        "formula_n": 0,
        "numeric_correct": 0,
        "numeric_n": 0,
        "unit_correct": 0,
        "unit_n": 0,
    }


def _add(bucket: dict[str, Any], score) -> None:
    bucket["rows"] += 1
    bucket["answer_correct"] += int(score.answer_correct)
    bucket["explanation_consistent"] += int(score.explanation_consistent)
    bucket["hallucinated_premise"] += int(score.hallucinated_premise)
    bucket["latency_ms"] += score.latency_ms
    if score.premise_precision is not None:
        bucket["premise_precision_sum"] += score.premise_precision
        bucket["premise_precision_n"] += 1
    if score.premise_recall is not None:
        bucket["premise_recall_sum"] += score.premise_recall
        bucket["premise_recall_n"] += 1
    if score.formula_correct is not None:
        bucket["formula_correct"] += int(score.formula_correct)
        bucket["formula_n"] += 1
    if score.numeric_correct is not None:
        bucket["numeric_correct"] += int(score.numeric_correct)
        bucket["numeric_n"] += 1
    if score.unit_correct is not None:
        bucket["unit_correct"] += int(score.unit_correct)
        bucket["unit_n"] += 1


def _rate(num: float, den: float) -> float | None:
    return None if not den else num / den


def _summarize(bucket: dict[str, Any]) -> dict[str, Any]:
    rows = bucket["rows"]
    return {
        "rows": rows,
        "answer_accuracy": _rate(bucket["answer_correct"], rows),
        "explanation_consistency": _rate(bucket["explanation_consistent"], rows),
        "hallucinated_premise_rate": _rate(bucket["hallucinated_premise"], rows),
        "avg_latency_ms": _rate(bucket["latency_ms"], rows),
        "premise_precision": _rate(bucket["premise_precision_sum"], bucket["premise_precision_n"]),
        "premise_recall": _rate(bucket["premise_recall_sum"], bucket["premise_recall_n"]),
        "formula_accuracy": _rate(bucket["formula_correct"], bucket["formula_n"]),
        "numeric_accuracy": _rate(bucket["numeric_correct"], bucket["numeric_n"]),
        "unit_accuracy": _rate(bucket["unit_correct"], bucket["unit_n"]),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_report(summary_path: Path, error_path: Path, summary: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    lines = [
        "# Quality Eval Summary",
        "",
        f"Date/time: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Aggregate",
        "",
        "| Group | Rows | Answer Acc | Explain Consistency | Hallucinated Premise | Avg ms | Premise P | Premise R | Formula Acc | Numeric Acc | Unit Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, metrics in sorted(summary["aggregate"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(metrics["rows"]),
                    _fmt(metrics["answer_accuracy"]),
                    _fmt(metrics["explanation_consistency"]),
                    _fmt(metrics["hallucinated_premise_rate"]),
                    _fmt(metrics["avg_latency_ms"]),
                    _fmt(metrics["premise_precision"]),
                    _fmt(metrics["premise_recall"]),
                    _fmt(metrics["formula_accuracy"]),
                    _fmt(metrics["numeric_accuracy"]),
                    _fmt(metrics["unit_accuracy"]),
                ]
            )
            + " |"
        )
    lines += ["", "## By Dataset", ""]
    lines.append("| Dataset | Rows | Answer Acc | Explain Consistency | Avg ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for group, metrics in sorted(summary["by_dataset"].items()):
        lines.append(
            f"| {group} | {metrics['rows']} | {_fmt(metrics['answer_accuracy'])} | "
            f"{_fmt(metrics['explanation_consistency'])} | {_fmt(metrics['avg_latency_ms'])} |"
        )
    lines += ["", "## Top Failure Categories", ""]
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category, count in summary["failure_categories"]:
        lines.append(f"| {category} | {count} |")
    lines += ["", "Error cases: `" + str(error_path) + "`", ""]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    with error_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")


def run(paths: list[Path], report_dir: Path, write_reports: bool = True) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = defaultdict(_bucket)
    by_dataset: dict[str, dict[str, Any]] = defaultdict(_bucket)
    failure_categories: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []

    for path, row in _iter_rows(paths):
        task = str(row.get("task_type") or "auto")
        request_task = task if task in {"auto", "logic", "physics"} else "auto"
        request = QARequest(
            question=row.get("question") or row.get("prompt") or "",
            premises=row.get("premises") or [],
            choices=row.get("choices") or [],
            task_type=request_task,
            allow_llm_fallback=False,
        )
        started = time.perf_counter()
        response = predict_response(request)
        latency_ms = (time.perf_counter() - started) * 1000
        score = score_response(row, response, latency_ms)
        dataset_key = f"{path.name}:{task}"
        for bucket in (aggregate[task], by_dataset[dataset_key]):
            _add(bucket, score)
        if not score.answer_correct or not score.explanation_consistent or score.hallucinated_premise:
            category = str(row.get("category") or row.get("reasoning_type") or row.get("difficulty") or row.get("failure_type") or "uncat")
            failure_categories[f"{task}:{category}"] += 1
            errors.append(
                {
                    "dataset": str(path),
                    "id": row.get("id"),
                    "task_type": task,
                    "category": category,
                    "gold_answer": gold_answer(row),
                    "answer": response.answer,
                    "score": score.to_dict(),
                    "fol": response.fol,
                    "premises": response.premises,
                    "explanation": response.explanation,
                }
            )

    summary = {
        "aggregate": {key: _summarize(value) for key, value in aggregate.items()},
        "by_dataset": {key: _summarize(value) for key, value in by_dataset.items()},
        "failure_categories": failure_categories.most_common(30),
    }
    if write_reports:
        report_dir.mkdir(parents=True, exist_ok=True)
        _write_report(
            report_dir / "quality_eval_summary.md",
            report_dir / "quality_eval_error_cases.jsonl",
            summary,
            errors,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets", help="JSONL dataset path. Repeatable.")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    paths = [Path(p) for p in (args.datasets or DEFAULT_DATASETS)]
    summary = run(paths, Path(args.report_dir), write_reports=not args.no_write)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
