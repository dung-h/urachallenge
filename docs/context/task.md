# Task: Parallel-Block-in-Series Resistor Network Solver

**Date:** 2026-06-05  
**Status:** ✅ Complete  
**Last Verified:** 2026-06-05 19:50 UTC

## Objective

Build and verify a deterministic circuit solver for parallel-block-in-series resistor networks in the URA EXACT Challenge physics pipeline.

## Target Question Pattern

```
Two resistors R1 = 2 kΩ and R2 = 3 kΩ are in parallel, and that block is in series with R3 = 500 Ω.
Source V = 12 V. Find the voltage drop across R3 and the current through branch R1.
```

## Expected Results

| Quantity | Formula | Value |
|----------|---------|-------|
| R_parallel | 1/(1/R1 + 1/R2) | 1200 Ω |
| R_total | R_parallel + R3 | 1700 Ω |
| I_total | V / R_total | 7.059 mA |
| V_R3 | I_total × R3 | **3.529 V** |
| V_parallel | V - V_R3 | 8.471 V |
| I_R1 | V_parallel / R1 | **4.235 mA** |

## Implementation

### Files Modified

1. **`app/physics/parser.py`** (lines 1713-1719)
   - Prevents incorrect selection of `direct_voltage_source` formula when the question asks for voltage drop or branch current in a circuit network containing multiple resistors.

2. **`app/physics/adapters/circuit.py`**
   - Added `_solve_series_parallel_resistors()` method implementing deterministic math solver
   - Robust regex-based variable extraction with SI unit normalization
   - Topology detection for parallel pair (supports R1∥R2, R2∥R3, or R1∥R3 configurations)
   - Multi-target output (voltage drop AND branch current in single response)

### Key Design Decisions

- **Deterministic over LLM**: All arithmetic is Python-computed, not LLM-generated
- **Unit normalization**: kΩ → Ω, mA → A conversions happen during parsing
- **Flexible topology**: Detects which resistor pair is parallel from question text
- **Multi-answer support**: Returns both requested quantities with proper formatting

## Verification

### API Endpoint Test

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question": "Two resistors R1 = 2 kΩ and R2 = 3 kΩ are in parallel, and that block is in series with R3 = 500 Ω. Source V = 12 V. Find the voltage drop across R3 and the current through branch R1.", "task": "physics"}'
```

**Result:** ✅ Correct
```json
{
  "answer": "3.529412 V; 0.004235 A",
  "explanation": "Solved series-parallel resistor network: voltage drop across R3 = 3.529412 V, current through R1 = 0.004235 A.",
  "fol": "series_parallel_resistor_network",
  "confidence": 0.8
}
```

### Router Test Suite

```
pytest tests/test_router.py -q
```

**Result:** 54 passed, 4 failed (pre-existing issues unrelated to circuit changes)

- ✅ `test_predict_physics_validated_json` — PASSED
- ✅ `test_predict_physics_validated_json_with_noise` — PASSED
- ❌ `test_predict_logic_validated_json` — Pre-existing schema issue (`proof_steps` serialization)
- ❌ `test_predict_logic_validated_json_with_noise` — Same pre-existing schema issue
- ❌ `test_router_keeps_llm_client_alive_through_heuristic_fallback` — Pre-existing planner_source mismatch
- ❌ `test_predict_physics_llm_orchestrator_rescues_unknown_capacitor_energy` — Pre-existing planner_source mismatch

## Next Steps

1. Root-cause the `proof_steps` serialization issue in `app/schemas.py:114`
2. Resolve planner_source metadata discrepancy (`llm_orchestrator` vs `deterministic_router`)
3. Expand circuit adapter to handle more complex topologies (nested parallel, ladder networks)
