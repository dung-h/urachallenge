# Deep Test — MethodPlanner over hard_eval_50

Date/time: 2026-06-07 17:42:20
LLM: `https://star-dsc-letter-reference.trycloudflare.com/v1`
Model: `qwen2.5:7b-instruct`

## Aggregate

- Total: **49/50** (98.0%)
- Physics: **24/25**
- Logic: **25/25**

## Method usage

| method_id | times selected |
|---|---|
| `physics.equation_graph` | 16 |
| `logic.legacy_pipeline` | 16 |
| `(legacy_fallback)` | 9 |
| `physics.legacy_pipeline` | 8 |
| `physics.retrieval_grounded` | 1 |

## Level-6 discovery events during this run

Total discovery attempts: **9**

- `logic_05_compound_missing` → (failed)  (discovery_response_unparseable)
- `logic_07_existential_no_match` → (failed)  (discovery_response_unparseable)
- `logic_08_some_to_some` → (failed)  (discovery_response_unparseable)
- `logic_11_contradiction` → (failed)  (validation_failed:regex_has_no_named_groups)
- `logic_16_irrelevant_only` → (failed)  (discovery_response_unparseable)
- `logic_18_policy_one_missing` → (failed)  (discovery_response_unparseable)
- `logic_19_policy_violated` → (failed)  (discovery_response_unparseable)
- `logic_22_affirming_consequent` → (failed)  (discovery_response_unparseable)
- `logic_23_denying_antecedent` → (failed)  (discovery_response_unparseable)

## Methods persisted at end of run

Total persisted (`models/methods.json`): **0**

_(none)_

## Top abstain reasons among failures

- `no_reason` × 1

---

Per-case detail: `reports/deep_test_planner_cases.jsonl`