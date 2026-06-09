"""Deep test of the method-centric planner on the hard_eval_50 dataset.

Goals:
  1. Measure pass-rate planner-on (single config, no legacy A/B).
  2. Per-method usage breakdown — which Methods actually fire?
  3. Level-6 evidence — count discovery events, list discovered methods,
     verify they get reused on later questions.
  4. Per-case latency + abstain reason for failures.

Output:
  reports/deep_test_planner_summary.md
  reports/deep_test_planner_cases.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Active config: planner ON, web search ON, qualitative parser ON,
# persistence ON for the duration of this run.
os.environ["URA_USE_METHOD_PLANNER"] = "1"
os.environ.setdefault("URA_ENABLE_QUALITATIVE_PARSER", "1")
os.environ.setdefault("URA_ENABLE_WEB_METHOD_SEARCH", "1")
os.environ["URA_METHODS_PERSISTENCE"] = "1"

# Restore-friendly: snapshot the persistence file so we can reset at end.
PERSIST = ROOT / "models" / "methods.json"
BACKUP = ROOT / "models" / "methods.json.deeptest_backup"
if PERSIST.exists():
    BACKUP.write_text(PERSIST.read_text(encoding="utf-8"), encoding="utf-8")

# Fresh library so the run isn't biased by prior state.
from app.methods.library import reset_default_library, get_default_library
reset_default_library()
# Wipe persistence before run so discovery is empirical.
if PERSIST.exists():
    PERSIST.unlink()

from app.router import predict_with_metadata
from app.schemas import QARequest

# Grading helpers — re-use hard_eval_v2's graders.
from scripts.hard_eval_v2 import grade_physics, grade_logic


CASES_PATH = ROOT / "reports" / "hard_eval_50_cases.jsonl"
cases = [json.loads(l) for l in CASES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Loaded {len(cases)} cases from {CASES_PATH.name}")
print(f"LLM = {os.environ.get('URA_LLM_BASE_URL')}")
print()

# ---------------------------------------------------------------------------

per_case_log: list[dict] = []
method_counter: Counter[str] = Counter()
discovery_events: list[dict] = []
pass_count = 0
phys_total = sum(1 for c in cases if c.get("task") == "physics")
logic_total = len(cases) - phys_total
phys_pass = 0
logic_pass = 0
abstain_reasons: Counter[str] = Counter()

for i, case in enumerate(cases, 1):
    is_phys = case.get("task") == "physics"
    payload = {"question": case["question"], "task": case["task"]}
    if "premises" in case:
        payload["premises"] = case["premises"]
    req = QARequest(allow_llm_fallback=True, **payload)

    t0 = time.perf_counter()
    err = None
    answer = ""
    meta: dict = {}
    try:
        resp, meta = predict_with_metadata(req)
        answer = (resp.answer or "").strip()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - t0

    if err:
        passed, note = False, f"runtime_error: {err[:80]}"
    elif is_phys:
        passed, note = grade_physics(case["expected"], answer)
    else:
        passed, note = grade_logic(case["expected"], answer)

    if passed:
        pass_count += 1
        if is_phys: phys_pass += 1
        else: logic_pass += 1

    selected_method = meta.get("selected_method_id") if meta else None
    if selected_method:
        method_counter[selected_method] += 1
    elif meta and meta.get("solver_used") == "method_planner":
        method_counter["(none decisive)"] += 1
    elif meta and meta.get("legacy_fallback_invoked"):
        method_counter["(legacy_fallback)"] += 1

    if meta and meta.get("discovery_attempted"):
        discovery_events.append({
            "case_id": case["id"],
            "outcome": meta.get("discovery_outcome"),
            "method_id": selected_method,
        })

    if not passed and meta:
        outcome = meta.get("planner_outcome") or {}
        abstain_reasons[outcome.get("abstain_reason") or "no_reason"] += 1

    mark = "✓" if passed else "✗"
    print(
        f"[{i:02d}/{len(cases)}] {case['id'][:32]:32s} {mark} "
        f"({latency:.1f}s) method={selected_method or '-'} ans={answer[:30]!r}"
    )

    per_case_log.append({
        "case_id": case["id"],
        "task": case["task"],
        "expected": case.get("expected"),
        "answer": answer,
        "passed": bool(passed),
        "note": note,
        "latency_s": float(latency),
        "selected_method": selected_method,
        "methods_tried": meta.get("planner_methods_tried") if meta else None,
        "discovery_attempted": bool(meta.get("discovery_attempted")) if meta else False,
        "discovery_outcome": meta.get("discovery_outcome") if meta else None,
        "error": err,
    })

# ---------------------------------------------------------------------------
# Persist final library state and gather discovery summary.
# ---------------------------------------------------------------------------

library = get_default_library()
library.persist()
final_methods = [m for m in library.all() if m.source.value.startswith("discovered")]
discovered_persisted = []
if PERSIST.exists():
    payload = json.loads(PERSIST.read_text(encoding="utf-8"))
    discovered_persisted = [
        e for e in payload.get("entries", [])
        if e.get("source", "").startswith("discovered")
    ]

# ---------------------------------------------------------------------------

out_dir = ROOT / "reports"
out_jsonl = out_dir / "deep_test_planner_cases.jsonl"
with out_jsonl.open("w", encoding="utf-8") as fh:
    for entry in per_case_log:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

method_rows = "\n".join(
    f"| `{m}` | {n} |" for m, n in method_counter.most_common()
)
discovery_rows = "\n".join(
    f"- `{d['case_id']}` → {d['method_id'] or '(failed)'}  ({d['outcome']})"
    for d in discovery_events
)
persisted_rows = "\n".join(
    f"- `{e['method_id']}` (signature `{e['signature']}`, source={e['source']})"
    for e in discovered_persisted
)

summary = [
    "# Deep Test — MethodPlanner over hard_eval_50",
    "",
    f"Date/time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"LLM: `{os.environ.get('URA_LLM_BASE_URL', '(default)')}`",
    f"Model: `{os.environ.get('URA_LLM_MODEL', '(default)')}`",
    "",
    "## Aggregate",
    "",
    f"- Total: **{pass_count}/{len(cases)}** ({100*pass_count/max(1,len(cases)):.1f}%)",
    f"- Physics: **{phys_pass}/{phys_total}**",
    f"- Logic: **{logic_pass}/{logic_total}**",
    "",
    "## Method usage",
    "",
    "| method_id | times selected |",
    "|---|---|",
    method_rows or "| _(none)_ | 0 |",
    "",
    "## Level-6 discovery events during this run",
    "",
    f"Total discovery attempts: **{len(discovery_events)}**",
    "",
    discovery_rows or "_(none — no question triggered discovery)_",
    "",
    "## Methods persisted at end of run",
    "",
    f"Total persisted (`models/methods.json`): **{len(discovered_persisted)}**",
    "",
    persisted_rows or "_(none)_",
    "",
    "## Top abstain reasons among failures",
    "",
    *(f"- `{r}` × {n}" for r, n in abstain_reasons.most_common()),
    "",
    "---",
    "",
    f"Per-case detail: `reports/deep_test_planner_cases.jsonl`",
]

(out_dir / "deep_test_planner_summary.md").write_text("\n".join(summary), encoding="utf-8")
print()
print(f"Wrote {out_dir / 'deep_test_planner_summary.md'}")

# ---------------------------------------------------------------------------
# Cleanup teardown — restore the original persistence file.
# ---------------------------------------------------------------------------

if BACKUP.exists():
    PERSIST.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
    BACKUP.unlink()
    print("Restored pre-test models/methods.json from backup.")
else:
    # Nothing to restore — the deep-test discoveries remain in models/methods.json
    # because the user explicitly wants Level-6 reuse across runs. Keep it.
    print("Kept the deep-test discoveries in models/methods.json (no backup to restore).")
