"""Hard evaluation v2: 60 hand-curated difficult cases (30 physics + 30 logic).

Extended from v1 to include:
- Qualitative physics questions (increase/decrease/unchanged)
- Physics with NO direct formula in registry (requires reasoning or search)
- Physics with complex multi-step reasoning
- Logic with nested quantifiers, temporal reasoning

Usage:
    wsl bash -c "cd /mnt/d/URA_challenge && python scripts/hard_eval_v2.py"
"""
from __future__ import annotations

import json
import time
import re
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT_S = 120.0

# ---------------------------------------------------------------------------
# 30 Physics hard cases
# ---------------------------------------------------------------------------
PHYSICS_CASES = [
    # ===== SECTION A: Qualitative questions (increase/decrease/unchanged) =====
    {
        "id": "phys_01_qual_R_increase_I",
        "question": "In a circuit with constant voltage, if the resistance is doubled, what happens to the current?",
        "expected": "halved",
        "tags": ["qualitative", "ohm_law", "inverse"],
    },
    {
        "id": "phys_02_qual_V_increase_P",
        "question": "If the voltage across a resistor is tripled while resistance stays constant, how does the power change?",
        "expected": "9 times",
        "tags": ["qualitative", "power", "squared"],
    },
    {
        "id": "phys_03_qual_cap_disconnect",
        "question": "A capacitor is charged to voltage V and then disconnected from the battery. If a dielectric with κ=2 is inserted, what happens to the voltage?",
        "expected": "halved",
        "tags": ["qualitative", "capacitor", "dielectric", "disconnected"],
    },
    {
        "id": "phys_04_qual_cap_connected",
        "question": "A capacitor remains connected to a voltage source. If a dielectric with κ=3 is inserted, what happens to the charge stored?",
        "expected": "tripled",
        "tags": ["qualitative", "capacitor", "dielectric", "connected"],
    },
    {
        "id": "phys_05_qual_E_field_distance",
        "question": "If the distance from a point charge is doubled, how does the electric field magnitude change?",
        "expected": "1/4",
        "tags": ["qualitative", "electric_field", "inverse_square"],
    },
    {
        "id": "phys_06_qual_coulomb_distance",
        "question": "Two charges are separated by distance r. If the distance is halved, what happens to the force between them?",
        "expected": "4 times",
        "tags": ["qualitative", "coulomb", "inverse_square"],
    },
    {
        "id": "phys_07_qual_LC_energy",
        "question": "In an LC circuit when the current is zero, where is all the energy stored?",
        "expected": "capacitor",
        "tags": ["qualitative", "LC", "energy_storage"],
    },
    {
        "id": "phys_08_qual_solenoid_turns",
        "question": "If the number of turns in a solenoid is doubled while keeping length and current constant, what happens to the magnetic field?",
        "expected": "doubled",
        "tags": ["qualitative", "solenoid", "magnetic_field"],
    },
    # ===== SECTION B: No direct formula - requires reasoning/search =====
    {
        "id": "phys_09_no_formula_doppler",
        "question": "A train horn emits sound at 500 Hz. The train moves toward a stationary observer at 30 m/s. Speed of sound is 340 m/s. What frequency does the observer hear?",
        "expected": "555 Hz",
        "tags": ["no_formula", "doppler", "waves"],
    },
    {
        "id": "phys_10_no_formula_refraction",
        "question": "Light travels from air (n=1) into glass (n=1.5) at an incident angle of 30°. What is the refraction angle?",
        "expected": "19.5°",
        "tags": ["no_formula", "snell_law", "optics"],
    },
    {
        "id": "phys_11_no_formula_buoyancy",
        "question": "A wooden block of density 600 kg/m³ floats in water (density 1000 kg/m³). What fraction of the block is submerged?",
        "expected": "0.6",
        "tags": ["no_formula", "buoyancy", "fluids"],
    },
    {
        "id": "phys_12_no_formula_thermal",
        "question": "How much heat is needed to raise the temperature of 2 kg of water from 20°C to 80°C? (Specific heat of water = 4186 J/kg·K)",
        "expected": "502320 J",
        "tags": ["no_formula", "thermal", "heat"],
    },
    {
        "id": "phys_13_no_formula_lens",
        "question": "An object is placed 30 cm from a convex lens of focal length 20 cm. Where is the image formed?",
        "expected": "60 cm",
        "tags": ["no_formula", "lens", "optics"],
    },
    {
        "id": "phys_14_no_formula_mirror",
        "question": "A concave mirror has a radius of curvature of 40 cm. An object is at 30 cm from the mirror. Find the image distance.",
        "expected": "60 cm",
        "tags": ["no_formula", "mirror", "optics"],
    },
    # ===== SECTION C: Complex multi-step or tricky numeric =====
    {
        "id": "phys_15_multistep_wheatstone",
        "question": "In a Wheatstone bridge, R1=100Ω, R2=200Ω, R3=150Ω. What value of R4 balances the bridge?",
        "expected": "300 Ω",
        "tags": ["multistep", "wheatstone", "circuit"],
    },
    {
        "id": "phys_16_multistep_two_capacitors",
        "question": "A 4μF capacitor charged to 100V is connected to an uncharged 6μF capacitor. What is the final voltage across both?",
        "expected": "40 V",
        "tags": ["multistep", "capacitor", "charge_conservation"],
    },
    {
        "id": "phys_17_fraction_input",
        "question": "A capacitor of capacitance 1/4 F is charged to 8 V. What is the stored energy?",
        "expected": "8 J",
        "tags": ["fraction_input", "capacitor", "energy"],
    },
    {
        "id": "phys_18_scientific_notation",
        "question": "An electron (mass 9.11e-31 kg) is accelerated through 100 V. What is its final speed? (e = 1.6e-19 C)",
        "expected": "5.93e6 m/s",
        "tags": ["scientific_notation", "electron", "energy"],
    },
    {
        "id": "phys_19_unit_trap",
        "question": "A 2200 μF capacitor is charged to 50 mV. What charge is stored?",
        "expected": "0.11 μC",
        "tags": ["unit_trap", "capacitor", "micro_milli"],
    },
    {
        "id": "phys_20_negative_charge",
        "question": "Two charges q1 = +5 μC and q2 = -3 μC are 20 cm apart. Is the force attractive or repulsive, and what is its magnitude?",
        "expected": "attractive, 3.37 N",
        "tags": ["sign", "coulomb", "direction"],
    },
    # ===== SECTION D: Mechanics (mostly missing from registry) =====
    {
        "id": "phys_21_projectile_time",
        "question": "A ball is thrown vertically upward with initial velocity 20 m/s. How long until it returns to the starting point? (g = 10 m/s²)",
        "expected": "4 s",
        "tags": ["mechanics", "projectile", "time"],
    },
    {
        "id": "phys_22_work_energy",
        "question": "A 10 kg box is pushed 5 m up a frictionless 30° incline. What work is done against gravity? (g = 10 m/s²)",
        "expected": "250 J",
        "tags": ["mechanics", "work", "incline"],
    },
    {
        "id": "phys_23_momentum",
        "question": "A 2 kg ball moving at 5 m/s collides head-on with a 3 kg ball moving at 2 m/s in opposite direction. They stick together. What is their final velocity?",
        "expected": "0.8 m/s",
        "tags": ["mechanics", "momentum", "collision"],
    },
    {
        "id": "phys_24_circular_motion",
        "question": "A 0.5 kg ball on a 1 m string moves in a horizontal circle at 4 m/s. What is the centripetal force?",
        "expected": "8 N",
        "tags": ["mechanics", "circular", "force"],
    },
    {
        "id": "phys_25_torque",
        "question": "A 2 m beam balances on a fulcrum at its center. A 30 N weight is placed 0.5 m from one end. What weight must be placed at the other end to balance?",
        "expected": "30 N",
        "tags": ["mechanics", "torque", "balance"],
    },
    # ===== SECTION E: Yes/No and conceptual =====
    {
        "id": "phys_26_yesno_resonance",
        "question": "An RLC circuit has L = 0.1 H and C = 100 μF. Is 50 Hz the resonance frequency?",
        "expected": "yes",
        "tags": ["yesno", "resonance", "verification"],
    },
    {
        "id": "phys_27_conceptual_field_inside",
        "question": "What is the electric field inside a conducting sphere in electrostatic equilibrium?",
        "expected": "0",
        "tags": ["conceptual", "conductor", "field"],
    },
    {
        "id": "phys_28_conceptual_parallel_plate",
        "question": "In a parallel plate capacitor, if the plate area is doubled and separation is halved, by what factor does capacitance change?",
        "expected": "4",
        "tags": ["conceptual", "capacitor", "factor"],
    },
    {
        "id": "phys_29_comparison",
        "question": "Two identical bulbs are connected: first in series, then in parallel to the same battery. In which case is total power consumption greater?",
        "expected": "parallel",
        "tags": ["comparison", "circuit", "power"],
    },
    {
        "id": "phys_30_si_unit",
        "question": "What is the SI unit of magnetic flux?",
        "expected": "Weber",
        "tags": ["conceptual", "unit", "magnetic"],
    },
]


# ---------------------------------------------------------------------------
# 30 Logic hard cases
# ---------------------------------------------------------------------------
LOGIC_CASES = [
    # ===== SECTION A: Multi-hop and chained reasoning =====
    {
        "id": "logic_01_four_hop",
        "question": "Is Eve qualified?",
        "premises": [
            "P1: If someone completes training, they are certified.",
            "P2: If someone is certified, they can apply for the job.",
            "P3: If someone applies for the job and passes interview, they are hired.",
            "P4: If someone is hired, they are qualified.",
            "P5: Eve completed training.",
            "P6: Eve passed the interview.",
        ],
        "expected": "yes",
        "tags": ["multi_hop", "four_hop"],
    },
    {
        "id": "logic_02_broken_chain",
        "question": "Is Frank qualified?",
        "premises": [
            "P1: If someone completes training, they are certified.",
            "P2: If someone is certified, they can apply for the job.",
            "P3: If someone applies for the job and passes interview, they are hired.",
            "P4: If someone is hired, they are qualified.",
            "P5: Frank completed training.",
        ],
        "expected": "unknown",
        "tags": ["multi_hop", "broken_chain", "missing_condition"],
    },
    {
        "id": "logic_03_transitive_5",
        "question": "Is object X larger than object A?",
        "premises": [
            "P1: A is larger than B.",
            "P2: B is larger than C.",
            "P3: C is larger than D.",
            "P4: D is larger than E.",
            "P5: E is larger than X.",
        ],
        "expected": "no",
        "tags": ["transitivity", "comparison", "deep"],
    },
    # ===== SECTION B: Quantifier traps =====
    {
        "id": "logic_04_all_some_trap",
        "question": "Is every student in the club?",
        "premises": [
            "P1: Some students are in the club.",
            "P2: All club members are students.",
        ],
        "expected": "unknown",
        "tags": ["quantifier", "some_all_trap"],
    },
    {
        "id": "logic_05_no_implies_not",
        "question": "Is Charlie a singer?",
        "premises": [
            "P1: No doctors are singers.",
            "P2: Charlie is a doctor.",
        ],
        "expected": "no",
        "tags": ["quantifier", "universal_negative"],
    },
    {
        "id": "logic_06_existential_scope",
        "question": "Does every engineer have a degree?",
        "premises": [
            "P1: Some engineers have a degree.",
            "P2: Bob is an engineer with a degree.",
        ],
        "expected": "unknown",
        "tags": ["quantifier", "existential_scope"],
    },
    {
        "id": "logic_07_nested_quantifier",
        "question": "Is there a student who passed every exam?",
        "premises": [
            "P1: Every exam was passed by some student.",
            "P2: There are 3 exams.",
        ],
        "expected": "unknown",
        "tags": ["quantifier", "nested", "scope"],
    },
    # ===== SECTION C: Negation and contradiction =====
    {
        "id": "logic_08_triple_negation",
        "question": "Is it not the case that John is not unhappy?",
        "premises": [
            "P1: John is happy.",
        ],
        "expected": "no",
        "tags": ["negation", "triple", "confusing"],
    },
    {
        "id": "logic_09_unless",
        "question": "Will the game be cancelled?",
        "premises": [
            "P1: The game will be cancelled unless it stops raining.",
            "P2: It continues to rain.",
        ],
        "expected": "yes",
        "tags": ["negation", "unless", "conditional"],
    },
    {
        "id": "logic_10_only_if",
        "question": "Did the alarm trigger?",
        "premises": [
            "P1: The alarm triggers only if the door is opened.",
            "P2: The door was not opened.",
        ],
        "expected": "no",
        "tags": ["conditional", "only_if"],
    },
    {
        "id": "logic_11_neither_nor",
        "question": "Is Alex a doctor?",
        "premises": [
            "P1: Alex is neither a doctor nor a lawyer.",
        ],
        "expected": "no",
        "tags": ["negation", "neither_nor"],
    },
    {
        "id": "logic_12_contradiction_explicit",
        "question": "What can we conclude about Sam's location?",
        "premises": [
            "P1: Sam is in Paris.",
            "P2: Sam is in Tokyo.",
            "P3: A person cannot be in two places at once.",
        ],
        "expected": "contradiction",
        "tags": ["contradiction", "explicit"],
    },
    # ===== SECTION D: Policy and compound conditions =====
    {
        "id": "logic_13_policy_3_of_4",
        "question": "Is Dana admitted?",
        "premises": [
            "P1: To be admitted, a student needs at least 3 of: GPA>3.0, recommendation letter, entrance exam pass, interview pass.",
            "P2: Dana has GPA 3.5.",
            "P3: Dana has a recommendation letter.",
            "P4: Dana passed the entrance exam.",
            "P5: Dana failed the interview.",
        ],
        "expected": "yes",
        "tags": ["policy", "threshold", "3_of_4"],
    },
    {
        "id": "logic_14_policy_exception",
        "question": "Does Mark get a discount?",
        "premises": [
            "P1: Members get a 10% discount.",
            "P2: However, the discount does not apply to sale items.",
            "P3: Mark is a member.",
            "P4: Mark is buying a sale item.",
        ],
        "expected": "no",
        "tags": ["policy", "exception", "override"],
    },
    {
        "id": "logic_15_or_both",
        "question": "Is Grace eligible?",
        "premises": [
            "P1: To be eligible, one must have a degree or 5 years experience.",
            "P2: Grace has both a degree and 7 years experience.",
        ],
        "expected": "yes",
        "tags": ["disjunction", "both_satisfied"],
    },
    {
        "id": "logic_16_xor",
        "question": "Can Henry enter?",
        "premises": [
            "P1: To enter, one must have either a ticket or be on the guest list, but not both.",
            "P2: Henry has a ticket.",
            "P3: Henry is on the guest list.",
        ],
        "expected": "no",
        "tags": ["disjunction", "xor", "exclusive"],
    },
    # ===== SECTION E: Temporal and causal reasoning =====
    {
        "id": "logic_17_temporal_before",
        "question": "Did event A happen before event C?",
        "premises": [
            "P1: Event A happened before event B.",
            "P2: Event B happened before event C.",
        ],
        "expected": "yes",
        "tags": ["temporal", "transitivity", "before"],
    },
    {
        "id": "logic_18_temporal_ambiguous",
        "question": "Did X happen before Z?",
        "premises": [
            "P1: X happened before Y.",
            "P2: Z happened after W.",
            "P3: Y and W happened at the same time.",
        ],
        "expected": "unknown",
        "tags": ["temporal", "ambiguous"],
    },
    {
        "id": "logic_19_causal_chain",
        "question": "Did the power outage cause the data loss?",
        "premises": [
            "P1: The storm caused the power outage.",
            "P2: The power outage caused the server to shut down.",
            "P3: The server shutting down caused the data loss.",
        ],
        "expected": "yes",
        "tags": ["causal", "chain"],
    },
    {
        "id": "logic_20_necessary_not_sufficient",
        "question": "Will the plant grow?",
        "premises": [
            "P1: Water is necessary for the plant to grow.",
            "P2: The plant has water.",
        ],
        "expected": "unknown",
        "tags": ["necessary", "not_sufficient"],
    },
    # ===== SECTION F: MCQ and comparison =====
    {
        "id": "logic_21_mcq_elimination",
        "question": "Which animal does Sarah own? A) Cat B) Dog C) Fish D) Bird",
        "premises": [
            "P1: Sarah owns a pet that can fly.",
            "P2: Sarah's pet is not a mammal.",
        ],
        "expected": "D",
        "tags": ["mcq", "elimination"],
    },
    {
        "id": "logic_22_mcq_all_wrong",
        "question": "What color is Tom's car? A) Red B) Blue C) Green",
        "premises": [
            "P1: Tom's car is yellow.",
        ],
        "expected": "none",
        "tags": ["mcq", "none_correct"],
    },
    {
        "id": "logic_23_comparison_ranking",
        "question": "Who is tallest?",
        "premises": [
            "P1: Alice is taller than Bob.",
            "P2: Carol is shorter than Bob.",
            "P3: Diana is taller than Alice.",
        ],
        "expected": "Diana",
        "tags": ["comparison", "ranking", "superlative"],
    },
    {
        "id": "logic_24_comparison_equal",
        "question": "Is A equal to C?",
        "premises": [
            "P1: A equals B.",
            "P2: B equals C.",
        ],
        "expected": "yes",
        "tags": ["comparison", "equality", "transitive"],
    },
    # ===== SECTION G: Fallacy traps and edge cases =====
    {
        "id": "logic_25_fallacy_composition",
        "question": "Is the machine heavy?",
        "premises": [
            "P1: The machine is made of light parts.",
            "P2: Each part weighs less than 1 kg.",
            "P3: The machine has 1000 parts.",
        ],
        "expected": "unknown",
        "tags": ["fallacy", "composition"],
    },
    {
        "id": "logic_26_fallacy_division",
        "question": "Is each player skilled?",
        "premises": [
            "P1: The team is highly skilled.",
            "P2: John is a player on the team.",
        ],
        "expected": "unknown",
        "tags": ["fallacy", "division"],
    },
    {
        "id": "logic_27_red_herring",
        "question": "Should we fund the project?",
        "premises": [
            "P1: The project will cost 1 million dollars.",
            "P2: Dr. Smith has won many awards.",
            "P3: The weather is nice today.",
            "P4: Projects costing over 500K need board approval.",
        ],
        "expected": "unknown",
        "tags": ["distractor", "red_herring", "irrelevant"],
    },
    {
        "id": "logic_28_circular",
        "question": "Is the law just?",
        "premises": [
            "P1: The law is just because it is legal.",
            "P2: It is legal because it follows the law.",
        ],
        "expected": "unknown",
        "tags": ["fallacy", "circular"],
    },
    {
        "id": "logic_29_empty_premise",
        "question": "Is the sky blue?",
        "premises": [],
        "expected": "unknown",
        "tags": ["edge", "no_premises"],
    },
    {
        "id": "logic_30_self_reference",
        "question": "Is statement P1 true?",
        "premises": [
            "P1: This statement is false.",
        ],
        "expected": "unknown",
        "tags": ["paradox", "self_reference"],
    },
]


# ---------------------------------------------------------------------------
# Grading functions
# ---------------------------------------------------------------------------
def normalize_answer(s: str | None) -> str:
    if not s:
        return ""
    return str(s).strip().lower().replace(",", "").replace("_", " ")


def grade_physics(expected: str, actual: str) -> tuple[bool, str]:
    """Lenient physics grading for numeric, qualitative, and conceptual answers."""
    if not actual or actual.lower() == "unknown":
        return False, "actual_unknown"
    
    exp_low = expected.lower()
    act_low = actual.lower()
    
    # Qualitative answers: check for keyword match
    qualitative_keywords = {
        "halved": ["halved", "half", "1/2", "0.5 times", "divided by 2"],
        "doubled": ["doubled", "double", "2 times", "twice"],
        "tripled": ["tripled", "triple", "3 times", "three times"],
        "4 times": ["4 times", "four times", "quadrupled", "4x"],
        "9 times": ["9 times", "nine times", "9x"],
        "1/4": ["1/4", "quarter", "one fourth", "0.25"],
        "capacitor": ["capacitor", "electric field", "electric energy"],
        "parallel": ["parallel"],
        "yes": ["yes", "correct", "true"],
        "no": ["no", "incorrect", "false"],
        "0": ["zero", "0", "none"],
        "attractive": ["attractive", "attract"],
        "weber": ["weber", "wb"],
    }
    
    for key, variants in qualitative_keywords.items():
        if key in exp_low:
            if any(v in act_low for v in variants):
                return True, "qualitative_match"
    
    # Numeric grading with tolerance
    num_pattern = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    e_match = re.search(num_pattern, expected.replace(",", ""))
    a_match = re.search(num_pattern, actual.replace(",", ""))
    
    if e_match and a_match:
        try:
            e_val = float(e_match.group(1))
            a_val = float(a_match.group(1))
            # Try to extract the unit and convert BOTH sides to SI base before
            # comparing. "9 mA" and "0.009 A" must compare equal.
            try:
                from app.physics.unit_converter import convert_value, normalize_unit
                e_unit_match = re.search(num_pattern + r"\s*([a-zA-ZμΩ°/^*·]+)", expected)
                a_unit_match = re.search(num_pattern + r"\s*([a-zA-ZμΩ°/^*·]+)", actual)
                e_unit = normalize_unit(e_unit_match.group(2)) if e_unit_match else ""
                a_unit = normalize_unit(a_unit_match.group(2)) if a_unit_match else ""
                if e_unit and a_unit:
                    e_si_val, e_si_unit = convert_value(e_val, e_unit)
                    a_si_val, a_si_unit = convert_value(a_val, a_unit)
                    # Compare in SI ONLY when the SI base units match — that
                    # confirms the answer is the same physical quantity.
                    # If SI bases differ ("kJ" vs "kW"), don't silently
                    # equate them; fall through to the raw-number check
                    # which will mark them as differing.
                    if e_si_unit == a_si_unit:
                        e_val, a_val = e_si_val, a_si_val
            except Exception:
                pass  # Best-effort SI normalization; fall back to raw nums.
            if e_val == 0:
                return abs(a_val) < 1e-6, "zero_compare"
            rel_err = abs(e_val - a_val) / abs(e_val)
            return rel_err < 0.05, f"rel_err={rel_err:.3f}"
        except ValueError:
            return False, "parse_error"
    
    # Exact string match fallback
    if exp_low == act_low:
        return True, "exact"
    
    return False, f"no_match: exp={exp_low}, act={act_low}"


def grade_logic(expected: str, actual: str) -> tuple[bool, str]:
    """Logic grading with flexible matching."""
    e = normalize_answer(expected)
    a = normalize_answer(actual)
    
    if e == a:
        return True, "exact"
    
    # Yes/No variants
    yes_variants = {"yes", "true", "correct", "affirmative"}
    no_variants = {"no", "false", "incorrect", "negative"}
    unknown_variants = {"unknown", "cannot determine", "insufficient information", "not enough information", "indeterminate"}
    
    if e in yes_variants and a in yes_variants:
        return True, "yes_match"
    if e in no_variants and a in no_variants:
        return True, "no_match"
    if e in unknown_variants and a in unknown_variants:
        return True, "unknown_match"
    
    # MCQ letter
    if len(e) == 1 and e.isalpha():
        if a.startswith(e) or a.endswith(f"({e})") or a.endswith(f"{e})"):
            return True, "mcq_match"
        if e.upper() == a.upper():
            return True, "letter_match"
    
    # Name extraction for ranking questions
    if e in a:
        return True, "contains"
    
    return False, f"e={e} != a={a}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def call_predict(case: dict) -> dict:
    """Send one case to the live API."""
    payload = {
        "question": case["question"],
        "task": "physics" if case["id"].startswith("phys") else "logic",
    }
    if "premises" in case:
        payload["premises"] = case["premises"]

    start = time.time()
    error = None
    body = None
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            resp = client.post(API_URL, json=payload)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start
    return {"latency_s": elapsed, "error": error, "body": body or {}}


def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "hard_eval_v2_cases.jsonl"
    out_summary = out_dir / "hard_eval_v2_summary.md"

    results = []
    physics_pass = 0
    logic_pass = 0
    physics_total = len(PHYSICS_CASES)
    logic_total = len(LOGIC_CASES)

    all_cases = PHYSICS_CASES + LOGIC_CASES
    print(f"Running {len(all_cases)} hard cases (v2) against {API_URL}...")
    print(f"  - {physics_total} physics (incl. qualitative, no-formula, mechanics)")
    print(f"  - {logic_total} logic (incl. temporal, quantifier traps, MCQ)")
    print()

    for i, case in enumerate(all_cases, 1):
        is_physics = case["id"].startswith("phys")
        print(f"[{i:02d}/{len(all_cases)}] {case['id'][:35]:35s} ... ", end="", flush=True)
        
        rsp = call_predict(case)
        body = rsp["body"]
        actual = body.get("answer", "")
        
        if rsp["error"]:
            passed = False
            grade_note = f"http_error: {rsp['error'][:50]}"
        elif is_physics:
            passed, grade_note = grade_physics(case["expected"], actual)
        else:
            passed, grade_note = grade_logic(case["expected"], actual)

        if passed:
            if is_physics:
                physics_pass += 1
            else:
                logic_pass += 1

        status = "PASS" if passed else "FAIL"
        print(f"{status} (exp={case['expected'][:15]}, got={str(actual)[:20]})")

        result = {
            "id": case["id"],
            "task": "physics" if is_physics else "logic",
            "tags": case.get("tags", []),
            "question": case["question"][:200],
            "expected": case["expected"],
            "actual": actual,
            "passed": passed,
            "grade_note": grade_note,
            "latency_s": round(rsp["latency_s"], 2),
            "confidence": body.get("confidence"),
            "explanation": (body.get("explanation") or "")[:150],
            "error": rsp["error"],
        }
        if "premises" in case:
            result["premises"] = case["premises"]
        results.append(result)

    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Failure analysis
    failures = [r for r in results if not r["passed"]]
    failure_by_tag: dict[str, list[str]] = {}
    for r in failures:
        for tag in r.get("tags", []):
            failure_by_tag.setdefault(tag, []).append(r["id"])

    # Summary
    summary_lines = [
        "# Hard Evaluation v2 — Summary",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**API**: {API_URL}",
        f"**Total cases**: {len(all_cases)}",
        "",
        "## Results",
        "",
        "| Task | Pass | Total | Accuracy |",
        "|------|------|-------|----------|",
        f"| Physics | {physics_pass} | {physics_total} | {physics_pass/physics_total*100:.1f}% |",
        f"| Logic | {logic_pass} | {logic_total} | {logic_pass/logic_total*100:.1f}% |",
        f"| **Total** | **{physics_pass+logic_pass}** | **{len(all_cases)}** | **{(physics_pass+logic_pass)/len(all_cases)*100:.1f}%** |",
        "",
        "## Failure Distribution by Tag",
        "",
    ]
    for tag, ids in sorted(failure_by_tag.items(), key=lambda x: -len(x[1])):
        summary_lines.append(f"- **{tag}** ({len(ids)}): {', '.join(ids[:5])}{'...' if len(ids)>5 else ''}")
    
    summary_lines.extend(["", "## Failed Cases Detail", ""])
    for r in failures[:20]:  # Limit to 20
        summary_lines.append(f"### {r['id']}")
        summary_lines.append(f"- **Q**: {r['question'][:100]}")
        summary_lines.append(f"- **Expected**: `{r['expected']}`")
        summary_lines.append(f"- **Actual**: `{r['actual']}`")
        summary_lines.append(f"- **Tags**: {r.get('tags', [])}")
        summary_lines.append("")

    out_summary.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Physics: {physics_pass}/{physics_total} ({physics_pass/physics_total*100:.1f}%)")
    print(f"Logic:   {logic_pass}/{logic_total} ({logic_pass/logic_total*100:.1f}%)")
    print(f"Total:   {physics_pass+logic_pass}/{len(all_cases)} ({(physics_pass+logic_pass)/len(all_cases)*100:.1f}%)")
    print("=" * 60)
    print(f"\nWrote: {out_jsonl}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
