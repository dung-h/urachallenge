"""In-process eval comparing the legacy router pipeline against the new
method-centric planner on the same hard_eval_v2 cases.

Runs each case TWICE (legacy then planner) so model variance is removed and
only routing differences contribute to the delta. Output:
    reports/planner_vs_legacy_summary.md
    reports/planner_vs_legacy_cases.jsonl

Usage (from /mnt/d/URA_challenge):
    source .venv/bin/activate
    python scripts/eval_planner_vs_legacy.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env so URA_LLM_BASE_URL/MODEL come from there.
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Disable persistence to keep this run repeatable.
os.environ["URA_METHODS_PERSISTENCE"] = "0"
# Enable the qualitative parser (production sets this in the API server
# launch script). Required for PhysicsQualitativeMethod to fire and for the
# legacy `_solve_qualitative` branch to run inside `solve_physics`. Without
# it, qualitative questions ("If R increases, what happens to I?") fall
# through to numeric adapters / retrieval and fail.
os.environ.setdefault("URA_ENABLE_QUALITATIVE_PARSER", "1")

from scripts.hard_eval_v2 import (  # noqa: E402  (after path setup)
    PHYSICS_CASES,
    LOGIC_CASES,
    grade_physics,
    grade_logic,
)
from app.router import predict_with_metadata  # noqa: E402
from app.schemas import QARequest  # noqa: E402


def run_one(case: dict, *, planner_on: bool) -> dict:
    """Run a single case in-process, return graded outcome."""
    if planner_on:
        os.environ["URA_USE_METHOD_PLANNER"] = "1"
    else:
        os.environ.pop("URA_USE_METHOD_PLANNER", None)
    payload = {
        "question": case["question"],
        "task": "physics" if case["id"].startswith("phys") else "logic",
    }
    if "premises" in case:
        payload["premises"] = case["premises"]
    req = QARequest(allow_llm_fallback=True, **payload)
    started = time.perf_counter()
    error = None
    answer = ""
    meta: dict = {}
    try:
        response, meta = predict_with_metadata(req)
        answer = (response.answer or "").strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_s = time.perf_counter() - started

    is_phys = case["id"].startswith("phys")
    if error:
        passed, note = False, f"runtime_error: {error[:80]}"
    elif is_phys:
        passed, note = grade_physics(case["expected"], answer)
    else:
        passed, note = grade_logic(case["expected"], answer)
    return {
        "id": case["id"],
        "category": "physics" if is_phys else "logic",
        "expected": case["expected"],
        "answer": answer,
        "passed": bool(passed),
        "note": note,
        "latency_s": float(latency_s),
        "selected_method": meta.get("selected_method_id") if meta else None,
        "solver_used": meta.get("solver_used") if meta else None,
        "methods_tried": meta.get("planner_methods_tried") if meta else None,
        "discovery_attempted": bool(meta.get("discovery_attempted")) if meta else False,
        "error": error,
    }


def main() -> int:
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "planner_vs_legacy_cases.jsonl"
    out_summary = out_dir / "planner_vs_legacy_summary.md"

    all_cases = PHYSICS_CASES + LOGIC_CASES
    print(f"Running {len(all_cases)} hard cases through both pipelines IN-PROCESS")
    print(f"  LLM: {os.environ.get('URA_LLM_BASE_URL', '(default)')}")
    print(f"  Model: {os.environ.get('URA_LLM_MODEL', '(default)')}")
    print()

    pairs: list[tuple[dict, dict]] = []
    legacy_pass = 0
    planner_pass = 0
    physics_total = sum(1 for c in all_cases if c["id"].startswith("phys"))
    logic_total = len(all_cases) - physics_total
    physics_legacy_pass = 0
    physics_planner_pass = 0
    logic_legacy_pass = 0
    logic_planner_pass = 0

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(all_cases, 1):
            print(f"[{i:02d}/{len(all_cases)}] {case['id'][:35]:35s} ", end="", flush=True)
            legacy = run_one(case, planner_on=False)
            planner = run_one(case, planner_on=True)
            pairs.append((legacy, planner))
            legacy_pass += int(legacy["passed"])
            planner_pass += int(planner["passed"])
            if legacy["category"] == "physics":
                physics_legacy_pass += int(legacy["passed"])
                physics_planner_pass += int(planner["passed"])
            else:
                logic_legacy_pass += int(legacy["passed"])
                logic_planner_pass += int(planner["passed"])
            mark_l = "✓" if legacy["passed"] else "✗"
            mark_p = "✓" if planner["passed"] else "✗"
            print(
                f"legacy={mark_l} ({legacy['latency_s']:.1f}s)  "
                f"planner={mark_p} ({planner['latency_s']:.1f}s)  "
                f"method={planner.get('selected_method') or '-'}"
            )
            fh.write(
                json.dumps({"legacy": legacy, "planner": planner}, ensure_ascii=False)
                + "\n"
            )

    # Build markdown summary.
    rows = [
        "| metric | legacy | planner | delta |",
        "|---|---|---|---|",
        f"| total pass | {legacy_pass}/{len(all_cases)} "
        f"| {planner_pass}/{len(all_cases)} | "
        f"{planner_pass - legacy_pass:+d} |",
        f"| physics pass | {physics_legacy_pass}/{physics_total} "
        f"| {physics_planner_pass}/{physics_total} | "
        f"{physics_planner_pass - physics_legacy_pass:+d} |",
        f"| logic pass | {logic_legacy_pass}/{logic_total} "
        f"| {logic_planner_pass}/{logic_total} | "
        f"{logic_planner_pass - logic_legacy_pass:+d} |",
    ]

    flips: list[str] = []
    for legacy, planner in pairs:
        if legacy["passed"] != planner["passed"]:
            direction = "LEGACY_ONLY" if legacy["passed"] else "PLANNER_ONLY"
            flips.append(
                f"- [{direction}] **{legacy['id']}** — "
                f"expected `{legacy['expected'][:60]}`; "
                f"legacy `{legacy['answer'][:40]}` / planner `{planner['answer'][:40]}` "
                f"(planner method: `{planner.get('selected_method') or '-'}`)"
            )

    body = [
        "# Planner vs Legacy — Hard Eval v2",
        "",
        f"Date/time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"LLM: `{os.environ.get('URA_LLM_BASE_URL', '(default)')}`",
        f"Model: `{os.environ.get('URA_LLM_MODEL', '(default)')}`",
        "",
        "## Aggregate",
        "",
        *rows,
        "",
        "## Flips (cases that changed pass/fail)",
        "",
        *(flips if flips else ["_None — every case had identical pass/fail across pipelines._"]),
    ]
    out_summary.write_text("\n".join(body), encoding="utf-8")
    print()
    print(f"Wrote {out_summary} and {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
