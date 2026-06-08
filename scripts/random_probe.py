#!/usr/bin/env python3
"""Random-question probe for the continuous improvement workflow.

Pattern (per user instruction): each turn run a fresh batch of NOVEL questions
(never the seen hard_eval set), capture answers + routing + solver provenance,
so the agent can analyze root causes and fix deeply. Questions are paraphrased /
restructured to test GENERALIZATION, not memorization (AGENTS.md §20).

Usage:
    python scripts/random_probe.py            # run default batch
    python scripts/random_probe.py --seed 7   # different sample
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT_S = 150.0

# NOVEL probe questions — paraphrased / new numbers / new domains, deliberately
# different from hard_eval_v2 so passing requires real generalization. Each has
# an expected answer (approx) and a tag for grouping root-cause analysis.
PROBES = [
    # --- Physics: paraphrased domains we just added (generalization check) ---
    {"id": "p_opt_refraction_2", "task": "auto",
     "q": "A ray of light enters water (n=1.33) from air (n=1.00) with an angle of incidence of 45 degrees. Find the angle of refraction.",
     "expected": "32.1", "tags": ["optics", "snell"]},
    {"id": "p_opt_lens_2", "task": "auto",
     "q": "An object sits 15 cm in front of a converging lens whose focal length is 10 cm. How far from the lens is the image?",
     "expected": "30 cm", "tags": ["optics", "lens"]},
    {"id": "p_fluid_2", "task": "auto",
     "q": "An ice cube of density 920 kg/m^3 floats in seawater of density 1025 kg/m^3. What fraction is below the surface?",
     "expected": "0.898", "tags": ["fluids", "buoyancy"]},
    {"id": "p_thermal_2", "task": "auto",
     "q": "Find the heat required to warm 0.5 kg of aluminum from 25 to 100 degrees, given specific heat 900 J/(kg·K).",
     "expected": "33750 J", "tags": ["thermal", "calorimetry"]},
    {"id": "p_mech_collision_2", "task": "auto",
     "q": "A 1 kg cart at 6 m/s strikes a stationary 2 kg cart and they couple together. What is the speed afterward?",
     "expected": "2 m/s", "tags": ["mechanics", "collision"]},
    {"id": "p_mech_electron_2", "task": "auto",
     "q": "A proton (mass 1.67e-27 kg, charge 1.6e-19 C) is accelerated through 200 V. What is its final speed?",
     "expected": "1.96e5 m/s", "tags": ["mechanics", "charged_particle"]},
    # --- Physics: electrical (regression guard) ---
    {"id": "p_ohm_2", "task": "auto",
     "q": "A resistor of 8 ohms carries a current of 1.5 A. What is the voltage across it?",
     "expected": "12 V", "tags": ["circuit", "ohm"]},
    {"id": "p_power_2", "task": "auto",
     "q": "A 220 V supply drives a 1100 W heater. What current does it draw?",
     "expected": "5 A", "tags": ["circuit", "power"]},
    {"id": "p_cap_energy_2", "task": "auto",
     "q": "A 10 μF capacitor is charged to 12 V. How much energy is stored?",
     "expected": "7.2e-4 J", "tags": ["capacitor", "energy"]},
    # --- Physics: qualitative (generalization of the factor logic) ---
    {"id": "p_qual_2", "task": "auto",
     "q": "If the current through a resistor is quadrupled, how does the power dissipated change?",
     "expected": "16 times", "tags": ["qualitative", "power"]},
    {"id": "p_qual_3", "task": "auto",
     "q": "When the radius of a circular orbit triples, what happens to the gravitational force? (inverse square)",
     "expected": "1/9", "tags": ["qualitative", "inverse_square"]},
    # --- Logic: comparison generalization ---
    {"id": "l_cmp_2", "task": "auto",
     "q": "Tom is faster than Jerry. Jerry is faster than Spike. Is Tom faster than Spike?",
     "expected": "yes", "tags": ["comparison", "transitive"]},
    {"id": "l_rank_2", "task": "auto",
     "q": "Mount A is higher than Mount B. Mount C is higher than Mount A. Mount B is higher than Mount D. Which is highest?",
     "expected": "C", "tags": ["comparison", "ranking"]},
    {"id": "l_eq_2", "task": "auto",
     "q": "X equals Y. Y equals Z. Z equals W. Is X equal to W?",
     "expected": "yes", "tags": ["comparison", "equality"]},
    # --- Logic: basic entailment (regression guard) ---
    {"id": "l_mp_2", "task": "auto",
     "q": "If a number is divisible by 4 then it is even. 12 is divisible by 4. Is 12 even?",
     "expected": "yes", "tags": ["modus_ponens"]},
    {"id": "l_disj_2", "task": "auto",
     "q": "To pass, a candidate must score above 50 or have a special exemption. Maria scored 72. Does Maria pass?",
     "expected": "yes", "tags": ["disjunction"]},
    # --- Logic: negation / quantifier ---
    {"id": "l_neg_2", "task": "auto",
     "q": "No reptiles are warm-blooded. A snake is a reptile. Is a snake warm-blooded?",
     "expected": "no", "tags": ["quantifier", "universal_negative"]},
    {"id": "l_nei_2", "task": "auto",
     "q": "Pat is neither a teacher nor a nurse. Is Pat a teacher?",
     "expected": "no", "tags": ["negation", "neither_nor"]},
]


def call(case: dict) -> dict:
    payload = {"question": case["q"], "task_type": case.get("task", "auto")}
    if "premises" in case:
        payload["premises"] = case["premises"]
    start = time.time()
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            resp = client.post(API_URL, json=payload)
            body = resp.json()
            err = None
    except Exception as exc:
        body, err = {}, f"{type(exc).__name__}: {exc}"
    return {
        "id": case["id"], "tags": case.get("tags", []),
        "question": case["q"], "expected": case["expected"],
        "answer": body.get("answer"), "task_type": body.get("task_type"),
        "fol": body.get("fol"), "confidence": body.get("confidence"),
        "explanation": (body.get("explanation") or "")[:160],
        "latency_s": round(time.time() - start, 2), "error": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path("reports/random_probe_results.jsonl")
    results = []
    print(f"Running {len(PROBES)} novel probe questions against {API_URL}")
    with out.open("w", encoding="utf-8") as f:
        for i, case in enumerate(PROBES, 1):
            r = call(case)
            flag = "" if r["answer"] else " [NO-ANSWER]"
            print(f"[{i:02d}/{len(PROBES)}] {case['id']:<22} task={r['task_type']} "
                  f"ans={str(r['answer'])[:22]!r} exp~={case['expected']!r}{flag}", flush=True)
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
