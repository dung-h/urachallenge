"""Test Level 6 discovery on questions that should NOT be solvable by
existing methods — physics topics outside the equation_graph + adapter
coverage. Level 6's purpose is exactly THIS: when nothing matches,
search the web, extract the formula, register, apply.

We craft questions on:
  * Specific gravity / buoyancy beyond the FluidsAdapter
  * Doppler shift (the eval grader-failed case)
  * de Broglie wavelength (quantum)
  * Adiabatic process (thermodynamics)
  * Black-body radiation (Stefan-Boltzmann)
  * Specific heat capacity transfer

For each: run through the planner, observe whether discovery fires,
whether the answer is correct, and whether the method gets registered
and reused on a structurally similar second question.
"""

from __future__ import annotations

import json
import os
import sys
import time
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

# Force planner mode + web search + qualitative parser.
os.environ["URA_USE_METHOD_PLANNER"] = "1"
os.environ.setdefault("URA_ENABLE_QUALITATIVE_PARSER", "1")
os.environ.setdefault("URA_ENABLE_WEB_METHOD_SEARCH", "1")
os.environ["URA_METHODS_PERSISTENCE"] = "1"

# Snapshot persistence for restore.
PERSIST = ROOT / "models" / "methods.json"
BACKUP = ROOT / "models" / "methods.json.l6test_backup"
if PERSIST.exists():
    BACKUP.write_text(PERSIST.read_text(encoding="utf-8"), encoding="utf-8")
    PERSIST.unlink()  # start fresh

from app.methods.library import reset_default_library, get_default_library
reset_default_library()

from app.router import predict_with_metadata
from app.schemas import QARequest


CASES = [
    # (id, question, expected_or_None, second_similar_question)
    (
        "doppler_observer_approaching",
        "A police siren emits sound at 500 Hz. The siren approaches a stationary observer at 30 m/s. The speed of sound is 340 m/s. What frequency does the observer hear?",
        "548 Hz",
        "An ambulance horn emits 600 Hz and approaches a stationary listener at 25 m/s. Speed of sound is 340 m/s. What frequency does the listener hear?",
    ),
    (
        "de_broglie_wavelength",
        "An electron has a momentum of 1.0e-24 kg·m/s. What is its de Broglie wavelength? Use h = 6.626e-34 J·s.",
        "6.6e-10 m",  # ~6.626e-10 m
        "A neutron has a momentum of 5.0e-25 kg·m/s. Compute its de Broglie wavelength (h = 6.626e-34 J·s).",
    ),
    (
        "stefan_boltzmann_power",
        "A black body has a surface area of 2.0 m^2 and a temperature of 500 K. Calculate the total radiated power. Use sigma = 5.67e-8 W/(m^2 K^4).",
        "7088 W",  # 2 * 5.67e-8 * 500^4 ≈ 7087.5 W
        "A black body of surface area 0.5 m^2 is at 1000 K. Compute its radiated power (sigma = 5.67e-8 W/(m^2 K^4)).",
    ),
    (
        "specific_heat_water",
        "How much heat is required to raise 2.0 kg of water from 20 C to 80 C? Specific heat of water is 4186 J/(kg·K).",
        "502320 J",  # 2 * 4186 * 60 = 502320 J
        "Compute the energy needed to heat 1.5 kg of aluminum from 25 C to 100 C if its specific heat is 897 J/(kg·K).",
    ),
    (
        "centripetal_acceleration",
        "A car travels around a circular track of radius 50 m at a constant speed of 20 m/s. What is its centripetal acceleration?",
        "8 m/s^2",  # v^2/r = 400/50 = 8
        "A satellite orbits a planet at speed 7000 m/s in a circular orbit of radius 1.0e7 m. Find its centripetal acceleration.",
    ),
]


print(f"LLM = {os.environ.get('URA_LLM_BASE_URL')}")
print(f"Library starts with {len(get_default_library().all())} methods\n")

log = []

for case_id, q1, expected, q2 in CASES:
    print(f"\n=== {case_id} ===")
    print(f"Q1: {q1}")
    print(f"Expected (~): {expected}")

    # First call — possibly triggers discovery.
    req1 = QARequest(question=q1, task="physics", allow_llm_fallback=True)
    t0 = time.perf_counter()
    err1 = None
    try:
        resp1, meta1 = predict_with_metadata(req1)
        ans1 = (resp1.answer or "").strip()
    except Exception as e:
        ans1 = ""
        err1 = f"{type(e).__name__}: {e}"
        meta1 = {}
    dt1 = time.perf_counter() - t0
    method1 = meta1.get("selected_method_id") or "(legacy_fallback)"
    discovered = bool(meta1.get("discovery_attempted"))
    print(f"  Q1 answer: {ans1!r} (method={method1}, discovery={discovered}, {dt1:.1f}s)")
    if err1:
        print(f"  ERROR: {err1}")

    # Second call — same family, should reuse a discovered method without
    # another search.
    print(f"\nQ2 (similar): {q2}")
    req2 = QARequest(question=q2, task="physics", allow_llm_fallback=True)
    t0 = time.perf_counter()
    err2 = None
    try:
        resp2, meta2 = predict_with_metadata(req2)
        ans2 = (resp2.answer or "").strip()
    except Exception as e:
        ans2 = ""
        err2 = f"{type(e).__name__}: {e}"
        meta2 = {}
    dt2 = time.perf_counter() - t0
    method2 = meta2.get("selected_method_id") or "(legacy_fallback)"
    print(f"  Q2 answer: {ans2!r} (method={method2}, {dt2:.1f}s)")

    log.append({
        "case_id": case_id,
        "expected_q1": expected,
        "q1_answer": ans1, "q1_method": method1, "q1_discovery": discovered, "q1_latency": dt1, "q1_error": err1,
        "q2_answer": ans2, "q2_method": method2, "q2_latency": dt2, "q2_error": err2,
    })

# Library state at end.
library = get_default_library()
discovered_methods = [m for m in library.all() if m.source.value.startswith("discovered")]

print(f"\n\nLibrary at end: {len(library.all())} methods, {len(discovered_methods)} discovered")
for m in discovered_methods:
    print(f"  - {m.method_id} ({m.source.value})")

# Persist and read back.
library.persist()
persisted_count = 0
if PERSIST.exists():
    payload = json.loads(PERSIST.read_text(encoding="utf-8"))
    persisted_count = len(payload.get("entries", []))
print(f"\nPersisted to {PERSIST}: {persisted_count} entries")

# Write report.
report = ROOT / "reports" / "level6_deep_test.md"
lines = [
    "# Level-6 Deep Test — Uncovered Physics Topics",
    "",
    f"Date/time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"LLM: `{os.environ.get('URA_LLM_BASE_URL')}`",
    "",
    "## Per-case",
    "",
    "| case | Q1 method | discovery? | Q1 ans | Q2 method | Q2 latency |",
    "|---|---|---|---|---|---|",
]
for e in log:
    lines.append(
        f"| `{e['case_id']}` | `{e['q1_method']}` | {'YES' if e['q1_discovery'] else 'no'} "
        f"| `{e['q1_answer']}` | `{e['q2_method']}` | {e['q2_latency']:.1f}s |"
    )
lines.extend([
    "",
    f"## Discovered methods at end of run: **{len(discovered_methods)}**",
    "",
    *(f"- `{m.method_id}`" for m in discovered_methods),
    "",
    f"Persisted entries in `models/methods.json`: **{persisted_count}**",
])
report.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {report}")

# Restore.
if BACKUP.exists():
    PERSIST.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
    BACKUP.unlink()
    print("Restored pre-test models/methods.json.")
else:
    if PERSIST.exists():
        PERSIST.unlink()
        print("Removed test-only models/methods.json.")
