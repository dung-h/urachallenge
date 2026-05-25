from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "Physics_Problems_Text_Only.csv"
DEFAULT_REPORT = ROOT / "outputs" / "physics_api_eval.csv"
DEFAULT_CONFIG = ROOT / "configs" / "physics_api_eval.yaml"

UNIT_FACTORS = {
    "": ("dimensionless", 1.0),
    "dimensionless": ("dimensionless", 1.0),
    "%": ("%", 1.0),
    "degree": ("degree", 1.0),
    "degrees": ("degree", 1.0),
    "j": ("J", 1.0),
    "mj": ("J", 1e-3),
    "uj": ("J", 1e-6),
    "μj": ("J", 1e-6),
    "µj": ("J", 1e-6),
    "f": ("F", 1.0),
    "mf": ("F", 1e-3),
    "uf": ("F", 1e-6),
    "μf": ("F", 1e-6),
    "µf": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "pf": ("F", 1e-12),
    "c": ("C", 1.0),
    "mc": ("C", 1e-3),
    "uc": ("C", 1e-6),
    "μc": ("C", 1e-6),
    "µc": ("C", 1e-6),
    "n": ("N", 1.0),
    "mn": ("N", 1e-3),
    "v": ("V", 1.0),
    "mv": ("V", 1e-3),
    "kv": ("V", 1e3),
    "a": ("A", 1.0),
    "ma": ("A", 1e-3),
    "w": ("W", 1.0),
    "mw": ("W", 1e-3),
    "kw": ("W", 1e3),
    "h": ("H", 1.0),
    "mh": ("H", 1e-3),
    "uh": ("H", 1e-6),
    "μh": ("H", 1e-6),
    "µh": ("H", 1e-6),
    "hz": ("Hz", 1.0),
    "khz": ("Hz", 1e3),
    "mhz": ("Hz", 1e6),
    "ohm": ("ohm", 1.0),
    "ohms": ("ohm", 1.0),
    "Ω": ("ohm", 1.0),
    "ω": ("ohm", 1.0),
    "t": ("T", 1.0),
    "mt": ("T", 1e-3),
    "wb": ("Wb", 1.0),
    "mwb": ("Wb", 1e-3),
}


@dataclass(frozen=True)
class EvalCase:
    source_id: str
    question: str
    expected_answer: str
    expected_unit: str


@dataclass(frozen=True)
class EvalResult:
    source_id: str
    question: str
    expected: str
    predicted: str
    pass_numeric: bool
    latency_ms: float
    status_code: int | None
    error: str
    request_id: str


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def clean_unit(unit: str) -> str:
    return (
        clean_text(unit)
        .replace("Âµ", "μ")
        .replace("micro", "μ")
        .replace("Ohm", "ohm")
        .replace("ohms", "ohm")
    )


def parse_number(text: str) -> float | None:
    text = clean_text(text)
    sqrt_match = re.search(r"([-+]?\d*\.?\d+)\s*\\sqrt\{?([-+]?\d*\.?\d+)\}?", text)
    multiplier = 1.0
    if sqrt_match:
        multiplier = float(sqrt_match.group(1)) * math.sqrt(float(sqrt_match.group(2)))
        tail = text[sqrt_match.end() :]
    else:
        match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text, flags=re.I)
        if not match:
            return None
        multiplier = float(match.group(0))
        tail = text[match.end() :]

    sci = re.search(r"(?:x|×|\*)\s*10\s*(?:\^)?\s*([-+]?\d+)", tail, flags=re.I)
    if sci:
        multiplier *= 10 ** int(sci.group(1))
    return multiplier


def parse_value_and_unit(answer: str, fallback_unit: str) -> tuple[float, str] | None:
    value = parse_number(answer)
    if value is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?(?:\s*(?:x|×|\*)\s*10\s*(?:\^)?\s*[-+]?\d+)?", answer, flags=re.I)
    unit = fallback_unit
    if match:
        suffix = answer[match.end() :].strip()
        if suffix:
            unit = suffix.split()[0].strip(";,.")
    return value, clean_unit(unit)


def to_si(value: float, unit: str) -> tuple[float, str] | None:
    unit = clean_unit(unit)
    key = unit.lower()
    if key not in UNIT_FACTORS:
        return None
    si_unit, factor = UNIT_FACTORS[key]
    return value * factor, si_unit


def numeric_match(expected_answer: str, expected_unit: str, predicted_answer: str, rel_tol: float) -> bool:
    expected = parse_value_and_unit(expected_answer, expected_unit)
    predicted = parse_value_and_unit(predicted_answer, expected_unit)
    if expected is None or predicted is None:
        return False
    expected_si = to_si(*expected)
    predicted_si = to_si(*predicted)
    if expected_si is None or predicted_si is None:
        return False
    expected_value, expected_si_unit = expected_si
    predicted_value, predicted_si_unit = predicted_si
    if expected_si_unit != predicted_si_unit:
        return False
    return math.isclose(predicted_value, expected_value, rel_tol=rel_tol, abs_tol=1e-12)


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    for row in rows:
        question = clean_text(row.get("question", ""))
        if not question:
            continue
        cases.append(
            EvalCase(
                source_id=clean_text(row.get("id", "")),
                question=question,
                expected_answer=clean_text(row.get("answer", "")),
                expected_unit=clean_text(row.get("unit", "")),
            )
        )
    return cases


def select_cases(cases: list[EvalCase], limit: int, mode: str, seed: int) -> list[EvalCase]:
    if limit <= 0 or limit >= len(cases):
        return list(cases)
    if mode == "random":
        rng = random.Random(seed)
        return rng.sample(cases, limit)
    return cases[:limit]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((p / 100) * len(ordered)) - 1))
    return ordered[index]


def call_predict(
    client: httpx.Client,
    api_url: str,
    case: EvalCase,
    timeout: float,
    request_options: dict[str, Any],
) -> tuple[str, int | None, str, str]:
    payload = {"question": case.question}
    payload.update(request_options)
    try:
        response = client.post(
            f"{api_url.rstrip('/')}/predict",
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:
        return "", None, f"request_error:{type(exc).__name__}:{exc}", ""

    request_id = response.headers.get("x-request-id", "")
    if response.status_code != 200:
        return "", response.status_code, response.text[:500], request_id
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return "", response.status_code, "non_json_response", request_id
    return clean_text(payload.get("answer", "")), response.status_code, "", request_id


def run_eval(args: argparse.Namespace) -> list[EvalResult]:
    cases = select_cases(load_cases(args.dataset), args.limit, args.mode, args.seed)
    if not cases:
        raise SystemExit("No cases selected.")

    results: list[EvalResult] = []
    with httpx.Client() as client:
        health = client.get(f"{args.api_url.rstrip('/')}/health", timeout=args.timeout)
        health.raise_for_status()
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            predicted, status_code, error, request_id = call_predict(
                client,
                args.api_url,
                case,
                args.timeout,
                args.request_options,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            ok = False if error else numeric_match(case.expected_answer, case.expected_unit, predicted, args.rel_tol)
            result = EvalResult(
                source_id=case.source_id,
                question=case.question,
                expected=f"{case.expected_answer} {case.expected_unit}".strip(),
                predicted=predicted,
                pass_numeric=ok,
                latency_ms=latency_ms,
                status_code=status_code,
                error=error,
                request_id=request_id,
            )
            results.append(result)
            if args.verbose or index % args.progress_every == 0 or index == len(cases):
                mark = "PASS" if ok else "FAIL"
                safe_print(f"[{index}/{len(cases)}] {mark} {case.source_id} latency={latency_ms:.1f}ms predicted={predicted!r}")
    return results


def write_report(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "pass_numeric",
                "latency_ms",
                "status_code",
                "request_id",
                "expected",
                "predicted",
                "error",
                "question",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_id": result.source_id,
                    "pass_numeric": result.pass_numeric,
                    "latency_ms": f"{result.latency_ms:.3f}",
                    "status_code": result.status_code,
                    "request_id": result.request_id,
                    "expected": result.expected,
                    "predicted": result.predicted,
                    "error": result.error,
                    "question": result.question,
                }
            )


def build_summary(results: list[EvalResult], report_path: Path) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    passed = sum(1 for result in results if result.pass_numeric)
    failed = len(results) - passed
    return {
        "cases": len(results),
        "pass": passed,
        "fail": failed,
        "accuracy": passed / len(results) if results else 0.0,
        "accuracy_percent": (passed / len(results) * 100) if results else 0.0,
        "latency_ms": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
        "report": str(report_path),
    }


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, Any], report_path: Path, summary_path: Path) -> None:
    latency = summary["latency_ms"]
    safe_print("\nSummary")
    safe_print(
        f"cases={summary['cases']} pass={summary['pass']} fail={summary['fail']} "
        f"accuracy={summary['accuracy_percent']:.2f}%"
    )
    safe_print(
        "latency_ms "
        f"avg={latency['avg']:.1f} "
        f"p50={latency['p50']:.1f} "
        f"p95={latency['p95']:.1f} "
        f"max={latency['max']:.1f}"
    )
    safe_print(f"report={report_path}")
    safe_print(f"summary={summary_path}")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("physics_api_eval", data) or {}


def _cfg(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    pre_args, _unknown = pre_parser.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Evaluate the original physics CSV through the running /predict API.",
        parents=[pre_parser],
    )
    parser.set_defaults(config_path=pre_args.config)
    parser.add_argument("--api-url", default=str(_cfg(config, "api_url", "http://127.0.0.1:8000")), help="Base URL of the running FastAPI app.")
    parser.add_argument("--dataset", type=Path, default=_path(_cfg(config, "dataset", DEFAULT_DATASET)), help="Physics CSV path.")
    parser.add_argument("--limit", type=int, default=int(_cfg(config, "limit", 0)), help="Number of samples to run. Use 0 for all rows.")
    parser.add_argument("--mode", choices=["first", "random"], default=str(_cfg(config, "mode", "first")), help="Sample selection mode when --limit is set.")
    parser.add_argument("--seed", type=int, default=int(_cfg(config, "seed", 42)), help="Random seed for --mode random.")
    parser.add_argument("--timeout", type=float, default=float(_cfg(config, "timeout", 30.0)), help="Per-request timeout in seconds.")
    parser.add_argument("--rel-tol", type=float, default=float(_cfg(config, "rel_tol", 0.03)), help="Relative tolerance for numeric answer matching.")
    parser.add_argument("--output", type=Path, default=_path(_cfg(config, "output", DEFAULT_REPORT)), help="CSV report output path.")
    parser.add_argument("--summary-output", type=Path, default=None if _cfg(config, "summary_output", None) is None else _path(_cfg(config, "summary_output", None)), help="JSON summary output path. Defaults to <output>.summary.json.")
    parser.add_argument("--progress-every", type=int, default=int(_cfg(config, "progress_every", 25)), help="Print progress every N cases.")
    parser.add_argument("--verbose", action="store_true", help="Print every case.")
    parser.set_defaults(request_options=dict(_cfg(config, "request", {})))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not isinstance(args.request_options, dict):
        raise SystemExit("Config field 'request' must be a mapping.")
    results = run_eval(args)
    write_report(results, args.output)
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary = build_summary(results, args.output)
    write_summary(summary, summary_path)
    print_summary(summary, args.output, summary_path)
    return 0 if all(result.pass_numeric for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
