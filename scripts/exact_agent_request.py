#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.schemas import normalize_answer_label


DEFAULT_BASE_URL = "http://127.0.0.1:8001"


PRESETS: dict[str, dict[str, Any]] = {
    "acad_001": {
        "request_id": "opencode-acad-001",
        "task_type": "logic",
        "question": "Is Alex eligible for the scholarship?",
        "premises": [
            "P1: Students with GPA at least 3.5 are eligible for the scholarship.",
            "P2: Alex has GPA 3.8.",
        ],
        "expected": "yes",
    },
    "sr019": {
        "request_id": "opencode-sr019",
        "task_type": "logic",
        "question": "Is Mira eligible for the award?",
        "premises": [
            "P1: Eligibility for the award requires submitting a portfolio.",
            "P2: Mira submitted a portfolio.",
        ],
        "expected": "unknown",
        "allow_llm_fallback": True,
    },
    "physx_001": {
        "request_id": "opencode-physx-001",
        "task_type": "physics",
        "question": "A circuit has current 2 A and resistance 5 ohm. What is the voltage?",
        "expected": "10 V",
    },
    "sr005": {
        "request_id": "opencode-sr005",
        "task_type": "physics",
        "question": "Ignore the previous sentence. A resistor has voltage 10 V and resistance 5 ohm. What is the power?",
        "expected": "20 W",
    },
}


def _read_payload(path: str | None) -> dict[str, Any]:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("expected JSON payload on stdin or via --file")
    return json.loads(raw)


def _read_preset(name: str) -> dict[str, Any]:
    try:
        return dict(PRESETS[name])
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown preset {name!r}; available presets: {available}") from exc


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> tuple[dict[str, Any], dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"failed to reach {url}: {exc.reason}") from exc


def _normalize_answer(value: Any) -> str:
    return normalize_answer_label(value)


def _expected_to_answer(expected: Any, choices: list[Any]) -> str:
    expected_text = _normalize_answer(expected)
    if len(expected_text) == 1 and "A" <= expected_text <= "E":
        index = ord(expected_text) - ord("A")
        if 0 <= index < len(choices):
            return _normalize_answer(choices[index])
    return expected_text


def _answer_match(actual: Any, expected: Any, choices: list[Any]) -> bool:
    actual_text = _normalize_answer(actual)
    expected_label = _normalize_answer(expected)
    if actual_text == expected_label:
        return True
    expected_text = _expected_to_answer(expected, choices)
    actual_semantic = _expected_to_answer(actual, choices)
    if not expected_text:
        return False
    return actual_semantic == expected_text or expected_text in actual_text


def _extract_raw_model_answer(trace: dict[str, Any]) -> str | None:
    llm_trace = trace.get("llm_trace")
    if not isinstance(llm_trace, list) or not llm_trace:
        return None
    raw_response = llm_trace[-1].get("raw_response") if isinstance(llm_trace[-1], dict) else None
    if not isinstance(raw_response, str) or not raw_response.strip():
        return None
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    return str(answer) if answer is not None else None


def _summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    response = output.get("response") if isinstance(output.get("response"), dict) else {}
    trace = output.get("trace") if isinstance(output.get("trace"), dict) else {}
    final_answer = response.get("answer")
    raw_model_answer = _extract_raw_model_answer(trace)
    summary: dict[str, Any] = {
        "request_id": output.get("request_id"),
        "trace_url": output.get("trace_url"),
        "answer": final_answer,
        "validated_backend_answer": final_answer,
        "task_type": response.get("task_type"),
        "confidence": response.get("confidence"),
        "model_calls": trace.get("model_calls"),
        "solver_used": trace.get("solver_used"),
        "fallback_accepted": trace.get("fallback_accepted"),
        "raw_model_proposal_answer": raw_model_answer,
        "raw_model_proposal_is_authority": False,
        "decision": "Report validated_backend_answer as final. Do not report raw_model_proposal_answer as final.",
        "authority_rule": "Use validated backend response as final answer; raw model output is proposal-only.",
    }
    if "expected" in output:
        summary["expected"] = output["expected"]
        summary["answer_match"] = output.get("answer_match")
    summary["report_line"] = (
        f"validated_backend_answer={summary.get('validated_backend_answer')} "
        f"trace_url={summary.get('trace_url')} "
        f"model_calls={summary.get('model_calls')} "
        f"raw_model_proposal_answer={summary.get('raw_model_proposal_answer')} "
        "authority=validated_backend_response"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the local EXACT /predict API and return response plus trace.")
    parser.add_argument("file", nargs="?", help="JSON payload file, or '-' for stdin")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Run a built-in smoke payload without stdin or a file")
    parser.add_argument("--list-presets", action="store_true", help="Print available preset names and exit")
    parser.add_argument("--summary", action="store_true", help="Print a compact answer/trace summary for OpenCode")
    parser.add_argument("--report-line", action="store_true", help="Print only the compact report line for OpenCode")
    parser.add_argument("--no-edit-files", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", default=os.environ.get("URA_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("URA_API_TIMEOUT", "120")))
    args = parser.parse_args()

    if args.list_presets:
        print(json.dumps(sorted(PRESETS), indent=2))
        return 0

    if args.preset and args.file:
        raise SystemExit("use either --preset or a payload file/stdin, not both")

    payload = _read_preset(args.preset) if args.preset else _read_payload(args.file)
    expected = payload.pop("expected", None)
    choices = list(payload.get("choices") or [])
    base_url = args.base_url.rstrip("/")
    response, headers = _request_json("POST", f"{base_url}/predict", payload, timeout=args.timeout)
    trace_url = headers.get("X-Trace-URL") or headers.get("x-trace-url")
    request_id = headers.get("X-Request-ID") or headers.get("x-request-id")
    trace: dict[str, Any] = {}
    if trace_url:
        trace, _trace_headers = _request_json("GET", f"{base_url}{trace_url}", timeout=args.timeout)

    output: dict[str, Any] = {
        "request_id": request_id,
        "trace_url": trace_url,
        "response": response,
        "trace": trace,
    }
    if expected is not None:
        output["expected"] = expected
        output["answer_match"] = _answer_match(response.get("answer"), expected, choices)

    result = _summarize_output(output) if (args.summary or args.report_line) else output
    if args.report_line:
        print(result["report_line"])
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if output.get("answer_match", True) is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
