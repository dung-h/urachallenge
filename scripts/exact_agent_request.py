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


def _read_payload(path: str | None) -> dict[str, Any]:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("expected JSON payload on stdin or via --file")
    return json.loads(raw)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the local EXACT /predict API and return response plus trace.")
    parser.add_argument("file", nargs="?", help="JSON payload file, or '-' for stdin")
    parser.add_argument("--base-url", default=os.environ.get("URA_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("URA_API_TIMEOUT", "120")))
    args = parser.parse_args()

    payload = _read_payload(args.file)
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

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output.get("answer_match", True) is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
