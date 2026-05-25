from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "datasets" / "eval"
DEFAULT_REPORT = ROOT / "reports" / "eval_case_clusters.md"
DEFAULT_REPRESENTATIVES = ROOT / "reports" / "eval_case_cluster_representatives.jsonl"


@dataclass(frozen=True)
class Case:
    source: str
    row_index: int
    case_id: str
    task_type: str
    question: str
    premises: list[str]
    answer: str
    cluster_key: str
    cluster_label: str
    metadata: dict[str, Any]


def _answer(obj: dict[str, Any]) -> str:
    for key in ("expected_answer", "gold_answer", "gold", "answer"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return "unknown"


def _text(obj: dict[str, Any]) -> str:
    parts = [str(obj.get("question") or ""), " ".join(str(p) for p in obj.get("premises") or [])]
    return " ".join(parts).lower()


def _logic_label(obj: dict[str, Any]) -> str:
    text = _text(obj)
    if obj.get("reasoning_type"):
        return f"reasoning:{obj['reasoning_type']}"
    if obj.get("category"):
        return f"category:{obj['category']}"
    if "contradict" in text or re.search(r"\bnot\b.+\bnot\b", text):
        return "logic:contradiction_or_conflict"
    if any(token in text for token in ["only if", "requires", "required", "prerequisite", "must have"]):
        return "logic:necessary_or_required_condition"
    if "if " in text and " then " in text:
        return "logic:conditional_rule"
    if re.search(r"\ball\b.+\bare\b", text) or re.search(r"\bno\b.+\bare\b", text):
        return "logic:universal_rule"
    if any(token in text for token in ["scholarship", "eligible", "gpa", "credits", "warning", "register"]):
        return "logic:academic_policy"
    if re.search(r"\b[A-E]\)\s+", str(obj.get("question") or "")):
        return "logic:mcq"
    return "logic:general"


def _physics_label(obj: dict[str, Any]) -> str:
    if obj.get("formula_id"):
        return f"formula:{obj['formula_id']}"
    if obj.get("category"):
        return f"category:{obj['category']}"
    text = _text(obj)
    if "series" in text and "parallel" in text:
        return "physics:mixed_series_parallel"
    if "transformer" in text:
        return "physics:transformer"
    if "capacitor" in text and "energy" in text:
        return "physics:capacitor_energy"
    if "capacitor" in text:
        return "physics:capacitance_or_charge"
    if "solenoid" in text or "magnetic" in text:
        return "physics:magnetism"
    if any(token in text for token in ["current", "voltage", "resistance", "ohm", "power"]):
        return "physics:ohm_power"
    if "unknown" in _answer(obj).lower():
        return "physics:unknown_or_unsupported"
    return "physics:general"


def _cluster_label(obj: dict[str, Any]) -> str:
    task = str(obj.get("task_type") or "auto").lower()
    if task == "physics" or (task == "auto" and any(k in _text(obj) for k in ["ohm", "voltage", "current", "capacitor", "transformer"])):
        return _physics_label(obj)
    if task == "logic" or obj.get("premises"):
        return _logic_label(obj)
    return f"auto:{obj.get('category') or 'general'}"


def _difficulty(obj: dict[str, Any]) -> str:
    return str(obj.get("difficulty") or obj.get("reason_for_hardness") or obj.get("expected_behavior") or "unspecified")


def _load_cases(paths: list[Path]) -> list[Case]:
    cases: list[Case] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                task = str(obj.get("task_type") or "auto").lower()
                label = _cluster_label(obj)
                answer = _answer(obj)
                source = str(path.relative_to(ROOT))
                cases.append(
                    Case(
                        source=source,
                        row_index=row_index,
                        case_id=str(obj.get("id") or f"{path.stem}:{row_index}"),
                        task_type=task,
                        question=str(obj.get("question") or ""),
                        premises=[str(p) for p in obj.get("premises") or []],
                        answer=answer,
                        cluster_key=f"{task}|{label}|answer:{answer.lower()}",
                        cluster_label=label,
                        metadata={
                            "difficulty": _difficulty(obj),
                            "category": obj.get("category"),
                            "reasoning_type": obj.get("reasoning_type"),
                            "formula_id": obj.get("formula_id"),
                            "source": source,
                        },
                    )
                )
    return cases


def _representatives(cases: list[Case], per_cluster: int) -> dict[str, list[Case]]:
    clusters: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        clusters[case.cluster_key].append(case)
    selected: dict[str, list[Case]] = {}
    for key, rows in clusters.items():
        rows_sorted = sorted(
            rows,
            key=lambda c: (
                0 if any(token in c.metadata["difficulty"].lower() for token in ["hard", "failure", "trap", "unsupported"]) else 1,
                len(c.premises),
                len(c.question),
                c.source,
                c.case_id,
            ),
        )
        selected[key] = rows_sorted[:per_cluster]
    return dict(sorted(selected.items(), key=lambda item: (-len(clusters[item[0]]), item[0])))


def _write_jsonl(path: Path, reps: dict[str, list[Case]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, rows in reps.items():
            for case in rows:
                handle.write(
                    json.dumps(
                        {
                            "cluster_key": key,
                            "cluster_label": case.cluster_label,
                            "id": case.case_id,
                            "source": case.source,
                            "row_index": case.row_index,
                            "task_type": case.task_type,
                            "answer": case.answer,
                            "question": case.question,
                            "premises": case.premises,
                            "metadata": case.metadata,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def _write_report(path: Path, cases: list[Case], reps: dict[str, list[Case]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_sizes = Counter(case.cluster_key for case in cases)
    by_task = Counter(case.task_type for case in cases)
    lines = [
        "# Eval Case Clusters",
        "",
        f"- Total cases: {len(cases)}",
        f"- Clusters: {len(cluster_sizes)}",
        "- By task: " + ", ".join(f"{task}={count}" for task, count in sorted(by_task.items())),
        "",
        "## Representative Clusters",
        "",
    ]
    for key, rows in reps.items():
        size = cluster_sizes[key]
        first = rows[0]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- Size: {size}",
                f"- Label: {first.cluster_label}",
                f"- Suggested representatives: {len(rows)}",
                "",
            ]
        )
        for case in rows:
            premise_preview = " | ".join(case.premises[:2])
            if len(premise_preview) > 240:
                premise_preview = premise_preview[:237] + "..."
            lines.extend(
                [
                    f"- `{case.case_id}` from `{case.source}:{case.row_index}`",
                    f"  - answer: `{case.answer}`",
                    f"  - question: {case.question}",
                ]
            )
            if premise_preview:
                lines.append(f"  - premises: {premise_preview}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster eval cases and select representatives for manual explanation review.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--per-cluster", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--representatives", type=Path, default=DEFAULT_REPRESENTATIVES)
    args = parser.parse_args()

    paths = sorted(args.dataset_dir.rglob("*.jsonl"))
    cases = _load_cases(paths)
    reps = _representatives(cases, max(1, args.per_cluster))
    _write_report(args.report, cases, reps)
    _write_jsonl(args.representatives, reps)
    print(json.dumps({"cases": len(cases), "clusters": len(reps), "report": str(args.report), "representatives": str(args.representatives)}, indent=2))


if __name__ == "__main__":
    main()
