# Quality Eval Summary

Date/time: 2026-05-22T05:19:10.164722+00:00

## Aggregate

| Group | Rows | Answer Acc | Explain Consistency | Hallucinated Premise | Avg ms | Premise P | Premise R | Formula Acc | Numeric Acc | Unit Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 30 | 0.500 | 1.000 | 0.000 | 8.956 | - | - | - | - | - |
| logic | 562 | 0.920 | 1.000 | 0.000 | 8.779 | 0.954 | 0.913 | - | - | - |
| physics | 344 | 0.965 | 1.000 | 0.000 | 8.938 | - | - | 1.000 | 0.960 | 0.967 |
| structured_json | 5 | 0.000 | 1.000 | 0.000 | 8.457 | - | - | - | - | - |

## By Dataset

| Dataset | Rows | Answer Acc | Explain Consistency | Avg ms |
| --- | ---: | ---: | ---: | ---: |
| adversarial_logic.jsonl:logic | 80 | 0.988 | 1.000 | 7.765 |
| adversarial_physics.jsonl:physics | 87 | 0.989 | 1.000 | 8.605 |
| exact_style_academic_policy_mock.jsonl:logic | 120 | 1.000 | 1.000 | 9.975 |
| exact_style_physics_mock.jsonl:physics | 120 | 1.000 | 1.000 | 9.048 |
| hardcase_academic_policy_qualitative.jsonl:logic | 60 | 0.933 | 1.000 | 9.225 |
| hardcase_physics_qualitative.jsonl:physics | 61 | 0.852 | 1.000 | 9.841 |
| hardcase_unknown_refusal.jsonl:logic | 23 | 0.957 | 1.000 | 9.310 |
| hardcase_unknown_refusal.jsonl:physics | 15 | 1.000 | 1.000 | 8.105 |
| mini_benchmark.jsonl:logic | 10 | 0.300 | 1.000 | 8.671 |
| mini_benchmark.jsonl:physics | 10 | 0.900 | 1.000 | 8.938 |
| mini_benchmark.jsonl:structured_json | 5 | 0.000 | 1.000 | 8.457 |
| phase_12_academic_policy_failures_regression.jsonl:logic | 84 | 1.000 | 1.000 | 8.852 |
| phase_16_safety_failures_regression.jsonl:logic | 9 | 1.000 | 1.000 | 8.323 |
| phase_16_safety_failures_regression.jsonl:physics | 2 | 1.000 | 1.000 | 6.577 |
| phase_24_alt_approach_eval.jsonl:logic | 70 | 0.786 | 1.000 | 7.887 |
| phase_24_alt_approach_eval.jsonl:physics | 10 | 1.000 | 1.000 | 8.400 |
| phase_27_symbolic_model_search.jsonl:logic | 80 | 0.800 | 1.000 | 8.076 |
| phase_9_failures_regression.jsonl:auto | 1 | 0.000 | 1.000 | 10.018 |
| phase_9_failures_regression.jsonl:logic | 21 | 0.952 | 1.000 | 9.345 |
| phase_9_failures_regression.jsonl:physics | 33 | 0.970 | 1.000 | 8.432 |
| regression_from_errors.jsonl:logic | 5 | 1.000 | 1.000 | 9.643 |
| regression_from_errors.jsonl:physics | 6 | 1.000 | 1.000 | 8.909 |
| schema_robustness.jsonl:auto | 29 | 0.517 | 1.000 | 8.919 |

## Top Failure Categories

| Category | Count |
| --- | ---: |
| auto:schema_valid_physics_prompt | 15 |
| physics:hard | 9 |
| logic:failed sufficient condition | 8 |
| logic:uncat | 7 |
| structured_json:uncat | 5 |
| logic:very_hard | 4 |
| logic:conditional_modus_ponens | 2 |
| physics:unsupported_composite_circuit | 2 |
| logic:threshold boundary insufficient | 2 |
| logic:approved absence exception | 2 |
| logic:contradictory fee facts | 2 |
| logic:four-hop taxonomy | 2 |
| logic:three-hop universal | 2 |
| logic:negative taxonomy | 2 |
| logic:negative universal | 2 |
| logic:direct fact | 2 |
| logic:mcq option mapping | 2 |
| logic:contradictory facts | 2 |
| logic:retake unsupported | 2 |
| logic:hard | 1 |
| logic:negative universal entailment | 1 |
| physics:uncat | 1 |

Error cases: `reports/quality_eval_error_cases.jsonl`
