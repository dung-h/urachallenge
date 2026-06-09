# Quality Eval Summary

Date/time: 2026-05-24T08:28:53.647431+00:00

## Aggregate

| Group | Rows | Answer Acc | Explain Consistency | Hallucinated Premise | Avg ms | Premise P | Premise R | Formula Acc | Numeric Acc | Unit Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 30 | 0.500 | 1.000 | 0.000 | 12.391 | - | - | - | - | - |
| logic | 562 | 0.975 | 1.000 | 0.000 | 12.409 | 0.954 | 0.913 | - | - | - |
| physics | 344 | 0.985 | 0.991 | 0.000 | 11.571 | - | - | 1.000 | 0.997 | 0.997 |
| structured_json | 5 | 0.000 | 1.000 | 0.000 | 11.724 | - | - | - | - | - |

## By Dataset

| Dataset | Rows | Answer Acc | Explain Consistency | Avg ms |
| --- | ---: | ---: | ---: | ---: |
| adversarial_logic.jsonl:logic | 80 | 0.988 | 1.000 | 12.505 |
| adversarial_physics.jsonl:physics | 87 | 1.000 | 1.000 | 10.919 |
| exact_style_academic_policy_mock.jsonl:logic | 120 | 1.000 | 1.000 | 14.208 |
| exact_style_physics_mock.jsonl:physics | 120 | 1.000 | 1.000 | 12.015 |
| hardcase_academic_policy_qualitative.jsonl:logic | 60 | 1.000 | 1.000 | 11.945 |
| hardcase_physics_qualitative.jsonl:physics | 61 | 1.000 | 1.000 | 11.227 |
| hardcase_unknown_refusal.jsonl:logic | 23 | 0.957 | 1.000 | 11.013 |
| hardcase_unknown_refusal.jsonl:physics | 15 | 0.933 | 0.933 | 11.142 |
| mini_benchmark.jsonl:logic | 10 | 0.300 | 1.000 | 11.108 |
| mini_benchmark.jsonl:physics | 10 | 0.900 | 1.000 | 11.582 |
| mini_benchmark.jsonl:structured_json | 5 | 0.000 | 1.000 | 11.724 |
| phase_12_academic_policy_failures_regression.jsonl:logic | 84 | 1.000 | 1.000 | 12.263 |
| phase_16_safety_failures_regression.jsonl:logic | 9 | 1.000 | 1.000 | 11.469 |
| phase_16_safety_failures_regression.jsonl:physics | 2 | 0.500 | 0.500 | 10.131 |
| phase_24_alt_approach_eval.jsonl:logic | 70 | 0.971 | 1.000 | 11.953 |
| phase_24_alt_approach_eval.jsonl:physics | 10 | 0.800 | 0.900 | 15.496 |
| phase_27_symbolic_model_search.jsonl:logic | 80 | 0.975 | 1.000 | 11.719 |
| phase_9_failures_regression.jsonl:auto | 1 | 0.000 | 1.000 | 12.623 |
| phase_9_failures_regression.jsonl:logic | 21 | 0.952 | 1.000 | 10.937 |
| phase_9_failures_regression.jsonl:physics | 33 | 1.000 | 1.000 | 11.533 |
| regression_from_errors.jsonl:logic | 5 | 1.000 | 1.000 | 10.017 |
| regression_from_errors.jsonl:physics | 6 | 1.000 | 1.000 | 10.859 |
| schema_robustness.jsonl:auto | 29 | 0.517 | 1.000 | 12.383 |

## Top Failure Categories

| Category | Count |
| --- | ---: |
| auto:schema_valid_physics_prompt | 15 |
| logic:uncat | 7 |
| structured_json:uncat | 5 |
| logic:failed sufficient condition | 4 |
| logic:conditional_modus_ponens | 2 |
| logic:very_hard | 1 |
| physics:very_hard | 1 |
| physics:should_have_returned_unknown | 1 |
| physics:open switch unsupported | 1 |
| physics:nested topology unsupported | 1 |
| physics:uncat | 1 |

Error cases: `reports/quality_eval_error_cases.jsonl`
