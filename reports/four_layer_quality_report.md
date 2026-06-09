# Four-Layer Quality Test Report

Date/time: 2026-05-22T07:08:27.068100+00:00

| Layer | Pass | Summary |
| --- | --- | --- |
| layer_1_smoke | yes | pytest passed |
| layer_2_quality_gates | yes | logic_answer_accuracy=ok, physics_answer_accuracy=ok, explanation_consistency=ok, hallucinated_premise_rate=ok, logic_latency_ms=ok, physics_latency_ms=ok |
| layer_3_failure_drilldown | yes | physics:hard=9, logic:failed sufficient condition=8, logic:very_hard=3, logic:threshold boundary insufficient=2, logic:approved absence exception=2 |
| layer_4_trace_audit | yes | trace audit passed |

## Details

### layer_1_smoke

- Command: `/mnt/d/URA_challenge/.venv/bin/python -m pytest -s -q`
- Pass: yes
- Summary: pytest passed
- Details:
```json
{
  "returncode": 0,
  "stdout": "..................................................................................................\n98 passed in 6.01s\n",
  "stderr": ""
}
```

### layer_2_quality_gates

- Command: `/mnt/d/URA_challenge/.venv/bin/python /mnt/d/URA_challenge/scripts/run_quality_eval.py --no-write`
- Pass: yes
- Summary: logic_answer_accuracy=ok, physics_answer_accuracy=ok, explanation_consistency=ok, hallucinated_premise_rate=ok, logic_latency_ms=ok, physics_latency_ms=ok
- Details:
```json
{
  "returncode": 0,
  "summary": {
    "aggregate": {
      "auto": {
        "answer_accuracy": 0.5,
        "avg_latency_ms": 8.846068466785558,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 30,
        "unit_accuracy": null
      },
      "logic": {
        "answer_accuracy": 0.9199288256227758,
        "avg_latency_ms": 8.690347957376762,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 0.954337899543379,
        "premise_recall": 0.91324200913242,
        "rows": 562,
        "unit_accuracy": null
      },
      "physics": {
        "answer_accuracy": 0.9651162790697675,
        "avg_latency_ms": 8.421982712280982,
        "explanation_consistency": 1.0,
        "formula_accuracy": 1.0,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.9603960396039604,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 344,
        "unit_accuracy": 0.966996699669967
      },
      "structured_json": {
        "answer_accuracy": 0.0,
        "avg_latency_ms": 7.954398399306228,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 5,
        "unit_accuracy": null
      }
    },
    "by_dataset": {
      "adversarial_logic.jsonl:logic": {
        "answer_accuracy": 0.9875,
        "avg_latency_ms": 8.342914912645938,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 80,
        "unit_accuracy": null
      },
      "adversarial_physics.jsonl:physics": {
        "answer_accuracy": 0.9885057471264368,
        "avg_latency_ms": 8.277533402523941,
        "explanation_consistency": 1.0,
        "formula_accuracy": 1.0,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.9885057471264368,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 87,
        "unit_accuracy": 0.9885057471264368
      },
      "exact_style_academic_policy_mock.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.5403284166811,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 1.0,
        "premise_recall": 0.95,
        "rows": 120,
        "unit_accuracy": null
      },
      "exact_style_physics_mock.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.85135530018791,
        "explanation_consistency": 1.0,
        "formula_accuracy": 1.0,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 1.0,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 120,
        "unit_accuracy": 1.0
      },
      "hardcase_academic_policy_qualitative.jsonl:logic": {
        "answer_accuracy": 0.9333333333333333,
        "avg_latency_ms": 8.628703133217641,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 60,
        "unit_accuracy": null
      },
      "hardcase_physics_qualitative.jsonl:physics": {
        "answer_accuracy": 0.8524590163934426,
        "avg_latency_ms": 8.165188114512826,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.8163265306122449,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 61,
        "unit_accuracy": 0.8571428571428571
      },
      "hardcase_unknown_refusal.jsonl:logic": {
        "answer_accuracy": 0.9565217391304348,
        "avg_latency_ms": 8.274953826596333,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 23,
        "unit_accuracy": null
      },
      "hardcase_unknown_refusal.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 7.585409133025678,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 15,
        "unit_accuracy": null
      },
      "mini_benchmark.jsonl:logic": {
        "answer_accuracy": 0.3,
        "avg_latency_ms": 7.824279199485318,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 0.0,
        "premise_recall": 0.0,
        "rows": 10,
        "unit_accuracy": null
      },
      "mini_benchmark.jsonl:physics": {
        "answer_accuracy": 0.9,
        "avg_latency_ms": 7.381528300174978,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.9,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 10,
        "unit_accuracy": 0.9
      },
      "mini_benchmark.jsonl:structured_json": {
        "answer_accuracy": 0.0,
        "avg_latency_ms": 7.954398399306228,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 5,
        "unit_accuracy": null
      },
      "phase_12_academic_policy_failures_regression.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.427550309620745,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 1.0,
        "premise_recall": 0.9642857142857143,
        "rows": 84,
        "unit_accuracy": null
      },
      "phase_16_safety_failures_regression.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.184727777309794,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 9,
        "unit_accuracy": null
      },
      "phase_16_safety_failures_regression.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.03412450113683,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 2,
        "unit_accuracy": null
      },
      "phase_24_alt_approach_eval.jsonl:logic": {
        "answer_accuracy": 0.7857142857142857,
        "avg_latency_ms": 8.454098842828119,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 70,
        "unit_accuracy": null
      },
      "phase_24_alt_approach_eval.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.003484400251182,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 1.0,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 10,
        "unit_accuracy": 1.0
      },
      "phase_27_symbolic_model_search.jsonl:logic": {
        "answer_accuracy": 0.8,
        "avg_latency_ms": 10.124460600400198,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 1.0,
        "premise_recall": 1.0,
        "rows": 80,
        "unit_accuracy": null
      },
      "phase_9_failures_regression.jsonl:auto": {
        "answer_accuracy": 0.0,
        "avg_latency_ms": 7.617726001626579,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 1,
        "unit_accuracy": null
      },
      "phase_9_failures_regression.jsonl:logic": {
        "answer_accuracy": 0.9523809523809523,
        "avg_latency_ms": 8.250388904603565,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 21,
        "unit_accuracy": null
      },
      "phase_9_failures_regression.jsonl:physics": {
        "answer_accuracy": 0.9696969696969697,
        "avg_latency_ms": 8.599764181615848,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.9696969696969697,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 33,
        "unit_accuracy": 0.9696969696969697
      },
      "regression_from_errors.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 9.767064200423192,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 5,
        "unit_accuracy": null
      },
      "regression_from_errors.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.214300833666735,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 1.0,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 6,
        "unit_accuracy": 1.0
      },
      "schema_robustness.jsonl:auto": {
        "answer_accuracy": 0.5172413793103449,
        "avg_latency_ms": 8.888425103515178,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 29,
        "unit_accuracy": null
      }
    },
    "failure_categories": [
      [
        "auto:schema_valid_physics_prompt",
        15
      ],
      [
        "physics:hard",
        9
      ],
      [
        "logic:failed sufficient condition",
        8
      ],
      [
        "logic:uncat",
        7
      ],
      [
        "structured_json:uncat",
        5
      ],
      [
        "logic:very_hard",
        4
      ],
      [
        "logic:conditional_modus_ponens",
        2
      ],
      [
        "physics:unsupported_composite_circuit",
        2
      ],
      [
        "logic:threshold boundary insufficient",
        2
      ],
      [
        "logic:approved absence exception",
        2
      ],
      [
        "logic:contradictory fee facts",
        2
      ],
      [
        "logic:four-hop taxonomy",
        2
      ],
      [
        "logic:three-hop universal",
        2
      ],
      [
        "logic:negative taxonomy",
        2
      ],
      [
        "logic:negative universal",
        2
      ],
      [
        "logic:direct fact",
        2
      ],
      [
        "logic:mcq option mapping",
        2
      ],
      [
        "logic:contradictory facts",
        2
      ],
      [
        "logic:retake unsupported",
        2
      ],
      [
        "logic:hard",
        1
      ],
      [
        "logic:negative universal entailment",
        1
      ],
      [
        "physics:uncat",
        1
      ]
    ]
  },
  "checks": [
    [
      "logic_answer_accuracy",
      true
    ],
    [
      "physics_answer_accuracy",
      true
    ],
    [
      "explanation_consistency",
      true
    ],
    [
      "hallucinated_premise_rate",
      true
    ],
    [
      "logic_latency_ms",
      true
    ],
    [
      "physics_latency_ms",
      true
    ]
  ]
}
```

### layer_3_failure_drilldown

- Command: `/mnt/d/URA_challenge/.venv/bin/python /mnt/d/URA_challenge/scripts/run_quality_eval.py --no-write --dataset datasets/eval/hardcase_academic_policy_qualitative.jsonl --dataset datasets/eval/hardcase_physics_qualitative.jsonl --dataset datasets/eval/phase_12_academic_policy_failures_regression.jsonl --dataset datasets/eval/phase_16_safety_failures_regression.jsonl --dataset datasets/eval/phase_24_alt_approach_eval.jsonl --dataset datasets/eval/phase_27_symbolic_model_search.jsonl`
- Pass: yes
- Summary: physics:hard=9, logic:failed sufficient condition=8, logic:very_hard=3, logic:threshold boundary insufficient=2, logic:approved absence exception=2
- Details:
```json
{
  "top_failures": [
    [
      "physics:hard",
      9
    ],
    [
      "logic:failed sufficient condition",
      8
    ],
    [
      "logic:very_hard",
      3
    ],
    [
      "logic:threshold boundary insufficient",
      2
    ],
    [
      "logic:approved absence exception",
      2
    ],
    [
      "logic:contradictory fee facts",
      2
    ],
    [
      "logic:four-hop taxonomy",
      2
    ],
    [
      "logic:three-hop universal",
      2
    ],
    [
      "logic:negative taxonomy",
      2
    ],
    [
      "logic:negative universal",
      2
    ]
  ],
  "focus_summary": {
    "datasets/eval/hardcase_academic_policy_qualitative.jsonl": {
      "hardcase_academic_policy_qualitative.jsonl:logic": {
        "answer_accuracy": 0.9333333333333333,
        "avg_latency_ms": 9.12872046665143,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 60,
        "unit_accuracy": null
      }
    },
    "datasets/eval/hardcase_physics_qualitative.jsonl": {
      "hardcase_physics_qualitative.jsonl:physics": {
        "answer_accuracy": 0.8524590163934426,
        "avg_latency_ms": 8.584390721211332,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 0.8163265306122449,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 61,
        "unit_accuracy": 0.8571428571428571
      }
    },
    "datasets/eval/phase_12_academic_policy_failures_regression.jsonl": {
      "phase_12_academic_policy_failures_regression.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 9.418597559358334,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 1.0,
        "premise_recall": 0.9642857142857143,
        "rows": 84,
        "unit_accuracy": null
      }
    },
    "datasets/eval/phase_16_safety_failures_regression.jsonl": {
      "phase_16_safety_failures_regression.jsonl:logic": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 9.28666833311177,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 9,
        "unit_accuracy": null
      },
      "phase_16_safety_failures_regression.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 7.623368999702507,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 2,
        "unit_accuracy": null
      }
    },
    "datasets/eval/phase_24_alt_approach_eval.jsonl": {
      "phase_24_alt_approach_eval.jsonl:logic": {
        "answer_accuracy": 0.7857142857142857,
        "avg_latency_ms": 8.585922156999004,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 70,
        "unit_accuracy": null
      },
      "phase_24_alt_approach_eval.jsonl:physics": {
        "answer_accuracy": 1.0,
        "avg_latency_ms": 8.262192500842502,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": 1.0,
        "premise_precision": null,
        "premise_recall": null,
        "rows": 10,
        "unit_accuracy": 1.0
      }
    },
    "datasets/eval/phase_27_symbolic_model_search.jsonl": {
      "phase_27_symbolic_model_search.jsonl:logic": {
        "answer_accuracy": 0.8,
        "avg_latency_ms": 8.713820299954023,
        "explanation_consistency": 1.0,
        "formula_accuracy": null,
        "hallucinated_premise_rate": 0.0,
        "numeric_accuracy": null,
        "premise_precision": 1.0,
        "premise_recall": 1.0,
        "rows": 80,
        "unit_accuracy": null
      }
    }
  }
}
```

### layer_4_trace_audit

- Command: `predict_response audit`
- Pass: yes
- Summary: trace audit passed
- Details:
```json
{
  "failures": [],
  "cases": 4
}
```
