from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FOCUS_DATASETS = [
    "datasets/eval/hardcase_academic_policy_qualitative.jsonl",
    "datasets/eval/hardcase_physics_qualitative.jsonl",
    "datasets/eval/phase_12_academic_policy_failures_regression.jsonl",
    "datasets/eval/phase_16_safety_failures_regression.jsonl",
    "datasets/eval/phase_24_alt_approach_eval.jsonl",
    "datasets/eval/phase_27_symbolic_model_search.jsonl",
]

DEFAULT_REGRESSION_TESTS = [
    "tests/test_router.py",
    "tests/test_policy_unknown_handling.py",
    "tests/test_invalid_inference_traps.py",
    "tests/test_phase_6_fallback.py",
    "tests/test_physics_reverse_formulas.py",
    "tests/test_academic_policy_reasoner.py",
    "tests/test_logic_proof_steps.py",
    "tests/test_predict_public_cot.py",
]


@dataclass
class LayerResult:
    name: str
    passed: bool
    command: str
    summary: str
    details: dict[str, Any]


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def _load_json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip() or "command failed")
    return json.loads(result.stdout)


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def layer1_smoke(pytest_args: list[str]) -> LayerResult:
    cmd = [PYTHON, "-m", "pytest", "-s", "-q", *pytest_args]
    proc = _run_command(cmd)
    passed = proc.returncode == 0
    summary = "pytest passed" if passed else "pytest failed"
    details = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    return LayerResult("layer_1_smoke", passed, " ".join(cmd), summary, details)


def layer2_quality_gates(datasets: list[str] | None = None) -> LayerResult:
    cmd = [PYTHON, str(ROOT / "scripts" / "run_quality_eval.py"), "--no-write"]
    for dataset in datasets or []:
        cmd.extend(["--dataset", dataset])
    proc = _run_command(cmd)
    try:
        summary = _load_json_stdout(proc)
    except Exception as exc:
        return LayerResult(
            "layer_2_quality_gates",
            False,
            " ".join(cmd),
            f"quality eval failed: {exc}",
            {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        )

    aggregate = summary.get("aggregate", {})
    logic = aggregate.get("logic", {})
    physics = aggregate.get("physics", {})
    checks = [
        ("logic_answer_accuracy", logic.get("answer_accuracy", 0.0) >= 0.88),
        ("physics_answer_accuracy", physics.get("answer_accuracy", 0.0) >= 0.95),
        ("explanation_consistency", min(
            [metric.get("explanation_consistency", 0.0) for metric in aggregate.values() if isinstance(metric, dict)] or [0.0]
        ) >= 0.90),
        ("hallucinated_premise_rate", max(
            [metric.get("hallucinated_premise_rate", 0.0) for metric in aggregate.values() if isinstance(metric, dict)] or [0.0]
        ) <= 0.0),
        ("logic_latency_ms", logic.get("avg_latency_ms", 999.0) < 25.0),
        ("physics_latency_ms", physics.get("avg_latency_ms", 999.0) < 25.0),
    ]
    passed = all(ok for _name, ok in checks)
    summary_text = ", ".join(f"{name}={'ok' if ok else 'fail'}" for name, ok in checks)
    details = {
        "returncode": proc.returncode,
        "summary": summary,
        "checks": checks,
    }
    return LayerResult("layer_2_quality_gates", passed, " ".join(cmd), summary_text, details)


def layer3_failure_drilldown(datasets: list[str]) -> LayerResult:
    cmd = [PYTHON, str(ROOT / "scripts" / "run_quality_eval.py"), "--no-write"]
    for dataset in datasets:
        cmd.extend(["--dataset", dataset])
    proc = _run_command(cmd)
    try:
        summary = _load_json_stdout(proc)
    except Exception as exc:
        return LayerResult(
            "layer_3_failure_drilldown",
            False,
            " ".join(cmd),
            f"drilldown failed: {exc}",
            {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        )
    top = summary.get("failure_categories", [])[:10]
    by_dataset = summary.get("by_dataset", {})
    focus_summary = {
        dataset: {
            key: value
            for key, value in by_dataset.items()
            if key.startswith(f"{Path(dataset).name}:")
        }
        for dataset in datasets
    }
    passed = True
    details = {
        "top_failures": top,
        "focus_summary": focus_summary,
    }
    summary_text = ", ".join(f"{cat}={count}" for cat, count in top[:5]) if top else "no failures"
    return LayerResult("layer_3_failure_drilldown", passed, " ".join(cmd), summary_text, details)


def layer4_trace_audit() -> LayerResult:
    from app.router import predict_response
    from app.schemas import QARequest, TaskType

    cases = [
        {
            "name": "logic_yes",
            "request": QARequest(
                task_type=TaskType.logic,
                question="Does Kim receive academic warning?",
                premises=[
                    "P1: Students with GPA below 2.0 receive academic warning.",
                    "P2: Kim has GPA 1.8.",
                ],
            ),
            "expected_answer": "yes",
            "must_contain": ["P1", "P2", "academic warning"],
        },
        {
            "name": "logic_unknown",
            "request": QARequest(
                task_type=TaskType.logic,
                question="Does Nia receive academic warning?",
                premises=[
                    "P1: Students with GPA below 2.0 or fewer than 10 credits receive academic warning.",
                    "P2: Nia has GPA 2.4.",
                    "P3: Nia's credit record is not available.",
                ],
            ),
            "expected_answer": "unknown",
            "must_contain": ["unknown", "missing"],
        },
        {
            "name": "physics_yes",
            "request": QARequest(
                task_type=TaskType.physics,
                question="A capacitor of 2 microfarad is connected to 5 V. What is the charge?",
            ),
            "expected_answer": "10 μC",
            "must_contain": ["Q = C * V", "Python", "C=2e-06"],
        },
        {
            "name": "physics_unknown",
            "request": QARequest(
                task_type=TaskType.physics,
                question="A circuit has a 10 V source, a 5 ohm resistor, and an open switch. What current flows?",
            ),
            "expected_answer": "unknown",
            "must_contain": ["open switch", "unknown"],
        },
    ]

    failures: list[dict[str, Any]] = []
    passed = True
    for case in cases:
        response = predict_response(case["request"])
        explanation = str(response.explanation or "")
        if response.answer != case["expected_answer"]:
            passed = False
            failures.append(
                {
                    "case": case["name"],
                    "problem": "answer_mismatch",
                    "expected": case["expected_answer"],
                    "actual": response.answer,
                    "explanation": explanation,
                }
            )
            continue
        missing = [token for token in case["must_contain"] if token.lower() not in explanation.lower() and token.upper() not in explanation.upper()]
        if missing:
            passed = False
            failures.append(
                {
                    "case": case["name"],
                    "problem": "trace_or_explanation_missing_tokens",
                    "missing": missing,
                    "actual": response.answer,
                    "explanation": explanation,
                    "premises": response.premises,
                    "fol": response.fol,
                }
            )

    summary = "trace audit passed" if passed else f"{len(failures)} trace audit failure(s)"
    details = {"failures": failures, "cases": len(cases)}
    return LayerResult("layer_4_trace_audit", passed, "predict_response audit", summary, details)


def _write_report(report_path: Path, layers: list[LayerResult]) -> None:
    lines = [
        "# Four-Layer Quality Test Report",
        "",
        f"Date/time: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Layer | Pass | Summary |",
        "| --- | --- | --- |",
    ]
    for layer in layers:
        lines.append(f"| {layer.name} | {'yes' if layer.passed else 'no'} | {layer.summary} |")
    lines += ["", "## Details", ""]
    for layer in layers:
        lines.append(f"### {layer.name}")
        lines.append("")
        lines.append(f"- Command: `{layer.command}`")
        lines.append(f"- Pass: {'yes' if layer.passed else 'no'}")
        lines.append(f"- Summary: {layer.summary}")
        if layer.details:
            lines.append("- Details:")
            lines.append("```json")
            lines.append(json.dumps(layer.details, indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the four-layer quality test workflow.")
    parser.add_argument("--report-path", default="reports/four_layer_quality_report.md")
    parser.add_argument("--focus-dataset", action="append", dest="focus_datasets")
    parser.add_argument("--pytest-arg", action="append", dest="pytest_args")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    focus_datasets = args.focus_datasets or DEFAULT_FOCUS_DATASETS
    pytest_args = args.pytest_args or []

    layers = [
        layer1_smoke(pytest_args),
        layer2_quality_gates(),
        layer3_failure_drilldown(focus_datasets),
        layer4_trace_audit(),
    ]

    for idx, layer in enumerate(layers, start=1):
        status = "PASS" if layer.passed else "FAIL"
        print(f"[{idx}] {layer.name}: {status} - {layer.summary}")

    if not args.skip_report:
        report_path = ROOT / args.report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report(report_path, layers)
        print(f"[report] {report_path}")

    if all(layer.passed for layer in layers):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
