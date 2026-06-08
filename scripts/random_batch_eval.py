#!/usr/bin/env python3
"""Random-batch generalization probe.

A fresh batch of questions NOT present in hard_eval_v2, used to find new
root-cause failures (per the agent workflow: run random questions -> trace
root cause -> fix). Diverse across physics domains and logic structures so we
surface generalization gaps rather than re-testing tuned cases.

Usage:
  source .venv/bin/activate && python scripts/random_batch_eval.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT_S = 120.0

# Each item: id, question, optional premises, expected (substring/qualitative), tags
PHYSICS = [
    # Ohm / power variants with different phrasing
    {"id": "rp_phys_01", "question": "A 9 V battery drives 0.3 A through a resistor. What is the resistor's value?", "expected": "30", "tags": ["ohm", "resistance"]},
    {"id": "rp_phys_02", "question": "A heater draws 4 A at 230 V. How much power does it consume?", "expected": "920", "tags": ["power"]},
    # series / parallel resistor numeric
    {"id": "rp_phys_03", "question": "Three resistors of 10 ohm, 20 ohm and 30 ohm are connected in series. What is the total resistance?", "expected": "60", "tags": ["series"]},
    {"id": "rp_phys_04", "question": "Two 100 ohm resistors are connected in parallel. What is the equivalent resistance?", "expected": "50", "tags": ["parallel"]},
    # capacitor energy / charge
    {"id": "rp_phys_05", "question": "A 10 microfarad capacitor is charged to 12 V. How much energy is stored?", "expected": "7.2e-4", "tags": ["capacitor", "energy"]},
    {"id": "rp_phys_06", "question": "A capacitor stores 6 mC of charge at 3 V. What is its capacitance?", "expected": "2e-3", "tags": ["capacitor"]},
    # kinematics / mechanics new phrasings
    {"id": "rp_phys_07", "question": "A car accelerates from rest at 3 m/s^2 for 5 s. What is its final speed?", "expected": "15 m/s", "tags": ["kinematics"]},
    {"id": "rp_phys_08", "question": "A 1500 kg car moving at 20 m/s brakes to a stop. How much kinetic energy is dissipated?", "expected": "300000", "tags": ["kinetic_energy"]},
    {"id": "rp_phys_09", "question": "A 5 kg object is lifted 3 m vertically. How much work is done against gravity? Take g = 9.8 m/s^2.", "expected": "147", "tags": ["work"]},
    # optics / waves new
    {"id": "rp_phys_10", "question": "Light goes from water (n=1.33) into air (n=1) hitting the surface at 20 degrees. What is the refraction angle?", "expected": "27", "tags": ["snell"]},
    {"id": "rp_phys_11", "question": "A sound wave travels at 340 m/s with a frequency of 170 Hz. What is its wavelength?", "expected": "2 m", "tags": ["wave"]},
    # coulomb / field
    {"id": "rp_phys_12", "question": "Two point charges of 2 microcoulomb each are 10 cm apart. What is the electrostatic force between them?", "expected": "3.6", "tags": ["coulomb"]},
    # qualitative variants
    {"id": "rp_phys_13", "question": "If the current through a resistor is doubled while resistance stays the same, how does the power dissipated change?", "expected": "4 times", "tags": ["qualitative", "power"]},
    {"id": "rp_phys_14", "question": "If the capacitance of a capacitor is doubled at constant voltage, what happens to the stored charge?", "expected": "doubled", "tags": ["qualitative", "charge"]},
    {"id": "rp_phys_15", "question": "If the radius is tripled, how does the electric field of a point charge change?", "expected": "1/9", "tags": ["qualitative", "field"]},
    # conceptual / unit lookups
    {"id": "rp_phys_16", "question": "What is the SI unit of electric charge?", "expected": "Coulomb", "tags": ["unit"]},
    {"id": "rp_phys_17", "question": "What is the SI unit of power?", "expected": "Watt", "tags": ["unit"]},
    # thermal / fluids
    {"id": "rp_phys_18", "question": "How much heat is required to raise 0.5 kg of water by 10 degrees C? Specific heat is 4186 J/kg/K.", "expected": "20930", "tags": ["thermal"]},
    {"id": "rp_phys_19", "question": "An object of density 800 kg/m^3 floats in oil of density 900 kg/m^3. What fraction is submerged?", "expected": "0.889", "tags": ["buoyancy"]},
    # momentum
    {"id": "rp_phys_20", "question": "A 4 kg cart at 3 m/s collides and sticks to a 2 kg cart at rest. What is their common velocity?", "expected": "2 m/s", "tags": ["momentum"]},
]

LOGIC = [
    {"id": "rp_log_01", "question": "Is Tom a reptile?",
     "premises": ["P1: All snakes are reptiles.", "P2: Tom is a snake."], "expected": "yes", "tags": ["syllogism"]},
    {"id": "rp_log_02", "question": "Is Mia a bird?",
     "premises": ["P1: No mammals are birds.", "P2: Mia is a mammal."], "expected": "no", "tags": ["universal_negative"]},
    {"id": "rp_log_03", "question": "Is Leo hungry?",
     "premises": ["P1: If Leo did not eat, then Leo is hungry.", "P2: Leo did not eat."], "expected": "yes", "tags": ["modus_ponens", "negation"]},
    {"id": "rp_log_04", "question": "Will the alarm sound?",
     "premises": ["P1: The alarm sounds unless the code is entered.", "P2: The code was not entered."], "expected": "yes", "tags": ["unless"]},
    {"id": "rp_log_05", "question": "Is the package delivered?",
     "premises": ["P1: The package is delivered only if someone signs.", "P2: Nobody signed."], "expected": "no", "tags": ["only_if"]},
    {"id": "rp_log_06", "question": "Who is the oldest?",
     "premises": ["P1: Anna is older than Ben.", "P2: Ben is older than Cara.", "P3: Dan is older than Anna."], "expected": "Dan", "tags": ["ranking"]},
    {"id": "rp_log_07", "question": "Is X heavier than Z?",
     "premises": ["P1: X is heavier than Y.", "P2: Y is heavier than Z."], "expected": "yes", "tags": ["transitive"]},
    {"id": "rp_log_08", "question": "Is P equal to R?",
     "premises": ["P1: P equals Q.", "P2: Q equals R."], "expected": "yes", "tags": ["equality"]},
    {"id": "rp_log_09", "question": "Is Sam eligible for the loan?",
     "premises": ["P1: To be eligible for the loan, one must have a job or a guarantor.", "P2: Sam has a steady job."], "expected": "yes", "tags": ["disjunction"]},
    {"id": "rp_log_10", "question": "Did event A happen before event D?",
     "premises": ["P1: A happened before B.", "P2: B happened before C.", "P3: C happened before D."], "expected": "yes", "tags": ["temporal"]},
    {"id": "rp_log_11", "question": "Is Kim a doctor?",
     "premises": ["P1: Kim is neither a doctor nor a nurse."], "expected": "no", "tags": ["neither_nor"]},
    {"id": "rp_log_12", "question": "Is the room dark?",
     "premises": ["P1: Some rooms are dark.", "P2: The hall is a room."], "expected": "unknown", "tags": ["quantifier", "some"]},
    {"id": "rp_log_13", "question": "Is the bridge safe?",
     "premises": ["P1: The bridge is safe.", "P2: The bridge is not safe."], "expected": "contradiction", "tags": ["contradiction"]},
    {"id": "rp_log_14", "question": "Will Maria graduate?",
     "premises": ["P1: If Maria passes all exams, she graduates.", "P2: Maria passed all exams."], "expected": "yes", "tags": ["modus_ponens"]},
    {"id": "rp_log_15", "question": "Is the statement enough to know if Joe drives?",
     "premises": ["P1: Joe drives only if he has a license.", "P2: Joe has a license."], "expected": "unknown", "tags": ["only_if", "affirming_consequent"]},
]


def _grade_physics(expected: str, actual: str) -> bool:
    if not actual or actual.lower() == "unknown":
        return False
    exp = expected.lower().replace(" ", "")
    act = actual.lower().replace(" ", "")
    # qualitative
    qual = {"4times": ["4times", "quadrupl"], "doubled": ["doubl", "2times", "twice"],
            "1/9": ["1/9", "0.111", "ninth"], "tripled": ["tripl", "3times"]}
    for key, variants in qual.items():
        if key.replace(" ", "") in exp:
            return any(v in act for v in variants)
    # word units
    for w in ["coulomb", "watt", "weber", "tesla", "joule", "ohm", "henry"]:
        if w in exp:
            return w in act
    # numeric tolerance
    import re
    # Normalize common SI prefixes so "720 μJ" matches "7.2e-4" (J) and
    # "2 mF" matches "2e-3" (F): scale the actual numeric by its prefix.
    prefix_scale = {
        "μ": 1e-6, "u": 1e-6, "micro": 1e-6,
        "m": 1e-3, "milli": 1e-3,
        "n": 1e-9, "nano": 1e-9,
        "k": 1e3, "kilo": 1e3,
        "p": 1e-12, "pico": 1e-12,
    }
    m_e = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", exp)
    m_a = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", act)
    if m_e and m_a:
        try:
            ev = float(m_e.group())
            av = float(m_a.group())
            # Detect an SI prefix ONLY when it is followed by a unit letter
            # (e.g. "mF", "μJ", "kΩ"). A lone "m"/"n" is the unit metre/newton,
            # NOT a milli/nano prefix, so it must not be scaled.
            tail = act[m_a.end():].lstrip()
            base_units = {"f", "j", "c", "w", "v", "a", "s", "g", "l", "ω"}
            for pfx, scale in prefix_scale.items():
                if tail.startswith(pfx) and len(tail) > len(pfx):
                    nxt = tail[len(pfx):len(pfx) + 1]
                    if nxt in base_units:
                        av *= scale
                        break
            if ev == 0:
                return abs(av) < 1e-6
            return abs(ev - av) / abs(ev) < 0.05
        except ValueError:
            return False
    return exp in act


def _grade_logic(expected: str, actual: str) -> bool:
    e = (expected or "").strip().lower()
    a = (actual or "").strip().lower()
    if not a:
        return False
    yes = {"yes", "true"}
    no = {"no", "false"}
    unk = {"unknown", "cannot determine", "insufficient", "not enough"}
    if e in yes:
        return a in yes
    if e in no:
        return a in no
    if e in unk:
        return any(u in a for u in unk)
    if e == "contradiction":
        return "contradict" in a or a in unk
    return e in a


def main() -> None:
    results = []
    p_pass = l_pass = 0
    all_items = [("physics", x) for x in PHYSICS] + [("logic", x) for x in LOGIC]
    print(f"Running {len(all_items)} random-batch questions against {API_URL}")
    for i, (kind, item) in enumerate(all_items, 1):
        payload = {"question": item["question"]}
        if "premises" in item:
            payload["premises"] = item["premises"]
        start = time.time()
        actual = ""
        err = None
        try:
            with httpx.Client(timeout=TIMEOUT_S) as c:
                r = c.post(API_URL, json=payload)
                actual = (r.json() or {}).get("answer", "")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - start
        if kind == "physics":
            ok = _grade_physics(item["expected"], actual)
            p_pass += int(ok)
        else:
            ok = _grade_logic(item["expected"], actual)
            l_pass += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{i:02d}/{len(all_items)}] {item['id']:14s} {status} exp={item['expected'][:14]!r} got={str(actual)[:22]!r}")
        results.append({**item, "kind": kind, "actual": actual, "passed": ok, "latency_s": round(elapsed, 1), "error": err})

    out = Path("reports/random_batch_results.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Physics: {p_pass}/{len(PHYSICS)}")
    print(f"Logic:   {l_pass}/{len(LOGIC)}")
    print(f"Total:   {p_pass + l_pass}/{len(all_items)}")
    print(f"Wrote reports/random_batch_results.jsonl")


if __name__ == "__main__":
    main()
