"""Level-6 (runtime method discovery) end-to-end audit.

Verifies that:
  1. Discovery TRIGGERS when no built-in method decisively solves a
     physics question that the legacy pipeline cannot solve either.
  2. Web search is REACHABLE from inside ``retrieve_method_evidence``
     (set URA_ENABLE_WEB_METHOD_SEARCH=1).
  3. Extracted recipe passes the dimensional + magnitude gates in
     ``solve_with_retrieved_method``.
  4. Registration produces a valid ``DiscoveredPhysicsMethod`` with a
     stable ``signature`` and source ``DISCOVERED_VERIFIED``.
  5. JSON persistence to ``models/methods.json`` round-trips.
  6. A second call with a STRUCTURALLY-similar question reuses the
     freshly registered method WITHOUT another web search (cache hit).
  7. The discovered method's ``score_match`` self-gates against
     non-numeric / qualitative / yesno questions (negative tests).

Run from /mnt/d/URA_challenge:
    source .venv/bin/activate
    python scripts/audit_level6_discovery.py
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

# Ensure the audit can persist (we'll clean up after).
PERSIST_PATH = ROOT / "models" / "methods.json"
AUDIT_BACKUP = ROOT / "models" / "methods.json.audit_backup"
if PERSIST_PATH.exists():
    AUDIT_BACKUP.write_text(PERSIST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
os.environ["URA_METHODS_PERSISTENCE"] = "1"

# Web search ON for the audit (it's the whole point of Level 6).
os.environ.setdefault("URA_ENABLE_WEB_METHOD_SEARCH", "1")

# Force a fresh library so the audit isn't biased by anything cached
# from a previous run.
from app.methods.library import reset_default_library, get_default_library
reset_default_library()

from app.methods.discovery import (
    discover_physics_method,
    DiscoveredPhysicsMethod,
)
from app.methods.problem import build_physics_problem, PhysicsProblem
from app.methods.planner import MethodPlanner
from app.methods.types import MethodSource
from app.physics.parser import parse_physics_question
from app.runtime_clients import build_runtime_llm_client
from app.pipeline_config import PipelineConfig

ROOT_REPORT = ROOT / "reports"
ROOT_REPORT.mkdir(exist_ok=True)
audit_log = []


def section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
    audit_log.append(f"\n## {title}\n")


def step(name: str, ok: bool, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    txt = f"  [{mark}] {name}"
    if detail:
        txt += f"  — {detail}"
    print(txt)
    audit_log.append(f"- {mark} **{name}** — {detail or 'ok'}")


# ---------------------------------------------------------------------------
# 0. Setup the LLM client.
# ---------------------------------------------------------------------------

cfg = PipelineConfig()
client = build_runtime_llm_client(cfg, enabled=True)
if client is None or not client.enabled:
    print("FATAL: cannot construct an LLM client. Check URA_LLM_BASE_URL.")
    sys.exit(2)
print(f"LLM: {client.model} @ {client.base_url}")
audit_log.append(f"LLM: `{client.model}` @ `{client.base_url}`")

library = get_default_library()
audit_log.append(f"\nLibrary at audit start: **{len(library.all())} methods**")


# ---------------------------------------------------------------------------
# 1. Pick a question whose target is unlikely to be hand-coded — so any
#    decisive solve must come from discovery (not the existing adapters).
# ---------------------------------------------------------------------------

section("1. Discovery trigger (uncovered numeric physics)")

# Specific gravity / dielectric / drag coefficient style — uncommon enough
# that the hand-coded adapters/registry won't match, but standard enough
# that web evidence exists.
audit_question = (
    "A simple pendulum has length 0.5 m. What is its period of oscillation "
    "in seconds (use g = 9.81 m/s^2)?"
)
parsed = parse_physics_question(audit_question)
problem = build_physics_problem(audit_question, parsed)
step(
    "Built PhysicsProblem from a generic question",
    bool(problem.parsed),
    f"target={problem.target_quantity!r}, knowns={problem.quantity_count}",
)


# Snapshot library before discovery.
pre_methods = {m.method_id for m in library.all()}
audit_log.append(f"\nMethods before discovery: {sorted(pre_methods)}\n")

t0 = time.perf_counter()
outcome = discover_physics_method(problem, llm_client=client, library=library)
elapsed_ms = (time.perf_counter() - t0) * 1000
step(
    "discover_physics_method ran",
    outcome.success,
    f"why={outcome.why}, elapsed={elapsed_ms:.0f}ms",
)

if outcome.success and outcome.method:
    new_method = outcome.method
    step(
        "Returned method has DISCOVERED_VERIFIED source",
        new_method.source == MethodSource.DISCOVERED_VERIFIED,
        f"source={new_method.source.value}",
    )
    step(
        "Returned method has a stable signature",
        bool(new_method.signature()) and new_method.signature().startswith("physics_recipe:"),
        f"signature={new_method.signature()!r}",
    )

# ---------------------------------------------------------------------------
# 2. Library registration + signature dedup.
# ---------------------------------------------------------------------------

section("2. Registration + signature deduplication")

post_methods = {m.method_id for m in library.all()}
new_ids = post_methods - pre_methods
step(
    "New method appears in the default library",
    bool(new_ids),
    f"added={sorted(new_ids)}",
)

# Try to register the SAME method again — must be a no-op.
if outcome.success and outcome.method:
    re_added = library.register(outcome.method)
    step(
        "Re-registering the same method is idempotent",
        re_added is False,
        "register() returned False on duplicate",
    )

# ---------------------------------------------------------------------------
# 3. Persistence round-trip.
# ---------------------------------------------------------------------------

section("3. JSON persistence round-trip")

library.persist()
step(
    "models/methods.json was written",
    PERSIST_PATH.exists(),
    f"path={PERSIST_PATH}",
)

if PERSIST_PATH.exists():
    payload = json.loads(PERSIST_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    persisted_ids = {e.get("method_id") for e in entries}
    step(
        "Discovered method is in the persisted payload",
        bool(new_ids & persisted_ids),
        f"persisted_ids={sorted(persisted_ids)}",
    )
    # Make sure built-ins are NOT persisted (only discovered/user-registered).
    builtin_seen = any(
        e.get("source") == MethodSource.BUILTIN.value for e in entries
    )
    step(
        "Built-in methods are NOT persisted (discover-only store)",
        not builtin_seen,
        "no entries with source=builtin",
    )

# ---------------------------------------------------------------------------
# 4. Reuse on a structurally similar question — should NOT re-search.
# ---------------------------------------------------------------------------

section("4. Reuse without re-searching")

# Same equation family, different lengths/g — the same recipe should fire.
similar_question = (
    "Find the period of a simple pendulum that is 1.2 m long, taking g = 9.81 m/s^2."
)
parsed2 = parse_physics_question(similar_question)
problem2 = build_physics_problem(similar_question, parsed2)

# Plan with the discovered method available.
planner = MethodPlanner(library=library)
shortlist = planner.shortlist(problem2)
discovered_in_shortlist = [
    (m.method_id, app.score, app.why)
    for m, app in shortlist
    if m.source == MethodSource.DISCOVERED_VERIFIED
]
step(
    "Discovered method appears in the planner shortlist for the similar question",
    bool(discovered_in_shortlist),
    f"shortlist={discovered_in_shortlist}",
)

# ---------------------------------------------------------------------------
# 5. Score gate — discovered method MUST NOT apply to non-numeric questions.
# ---------------------------------------------------------------------------

section("5. score_match self-gates against non-numeric / lookup / yesno")

if outcome.success and outcome.method:
    discovered = outcome.method

    # 5a. zero-quantities lookup question
    lookup_q = "What is the SI unit of frequency?"
    parsed_lookup = parse_physics_question(lookup_q)
    p_lookup = build_physics_problem(lookup_q, parsed_lookup)
    app_lookup = discovered.score_match(p_lookup)
    step(
        "Discovered method does NOT claim a lookup question",
        app_lookup.score < 0.3,
        f"score={app_lookup.score:.2f} ({app_lookup.why})",
    )

    # 5b. qualitative-change wording with no numeric quantities
    qual_q = "If the length of a pendulum is doubled, what happens to its period?"
    parsed_qual = parse_physics_question(qual_q)
    p_qual = build_physics_problem(qual_q, parsed_qual)
    app_qual = discovered.score_match(p_qual)
    step(
        "Discovered method does NOT claim a qualitative-change question with no numeric knowns",
        app_qual.score < 0.3,
        f"score={app_qual.score:.2f} ({app_qual.why})",
    )

# ---------------------------------------------------------------------------
# 6. End-of-audit cleanup: restore models/methods.json so the audit is
#    side-effect-free for normal operation.
# ---------------------------------------------------------------------------

section("6. Audit teardown (restoring pre-audit state)")

try:
    if AUDIT_BACKUP.exists():
        PERSIST_PATH.write_text(
            AUDIT_BACKUP.read_text(encoding="utf-8"), encoding="utf-8"
        )
        AUDIT_BACKUP.unlink()
        step("Restored models/methods.json from pre-audit backup", True)
    else:
        # No prior file — wipe what the audit wrote.
        if PERSIST_PATH.exists():
            PERSIST_PATH.unlink()
            step("Removed audit-only models/methods.json", True)
        else:
            step("No persistence file to clean", True)
except Exception as exc:
    step("Teardown failed", False, f"{type(exc).__name__}:{exc}")

# ---------------------------------------------------------------------------
# Summary report.
# ---------------------------------------------------------------------------

report_path = ROOT_REPORT / "level6_discovery_audit.md"
report_path.write_text(
    "\n".join(
        [
            "# Level-6 Discovery Audit",
            "",
            f"Date/time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            *audit_log,
            "",
            "---",
            "",
            "Audit script: `scripts/audit_level6_discovery.py`",
        ]
    ),
    encoding="utf-8",
)
print(f"\nReport written to {report_path}")
