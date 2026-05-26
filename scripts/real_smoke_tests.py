from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from app.main import app


def run() -> None:
    client = TestClient(app)
    cases = [
        {
            "name": "logic",
            "payload": {
                "question": "All birds are animals. Sparrows are birds. Is a sparrow an animal?",
                "premises": ["All birds are animals.", "Sparrows are birds."],
                "task_type": "logic",
            },
        },
        {
            "name": "physics",
            "payload": {
                "question": "A 12 V battery drives a 3 ohm resistor. What current flows?",
                "task_type": "physics",
            },
        },
    ]

    results: list[dict[str, object]] = []
    for case in cases:
        started = time.perf_counter()
        response = client.post("/predict", json=case["payload"])
        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text}
        results.append(
            {
                "case": case["name"],
                "status_code": response.status_code,
                "response": body,
                "trace_url": response.headers.get("X-Trace-URL"),
                "request_id": response.headers.get("X-Request-ID"),
                "wall_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
