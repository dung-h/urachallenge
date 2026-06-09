# Dataset Correction Report — Logic_Based_Educational_Queries

Date: 2026-06-07 (Session 11k). Manual per-case adjudication of all 411
items (808 questions) by 12 parallel subagents, each reasoning strictly
from premises + cross-checking the item's own `explanation`.

## Severity

**425 / 808 questions (53%) had gold labels contradicting both the logical
entailment from premises AND the item's own explanation.** 412 high+medium
confidence corrections were applied to produce a cleaned dataset.

## Files produced (original NEVER modified)

- `data/Logic_Based_Educational_Queries.json` — ORIGINAL, untouched.
- `data/Logic_Based_Educational_Queries.corrected.json` — cleaned copy.
  Each item keeps `answers_original` and an updated `answers`; premises and
  questions are byte-identical to the original (verified).
- `reports/dataset_corrections_audit.json` — per-change audit
  (pos, q_index, old, new, confidence, reason citing premise numbers).
- Per-chunk raw adjudications: `_tmp_corrections/chunk_*.corr.json`
  (kept until merge reviewed; can be regenerated).

## Correction statistics

- Items with ≥1 corrected answer: **281 / 411 (68%)**.
- Corrections applied: **412** (high=354, medium=58; low=13 NOT applied —
  conservative).
- By question type: MCQ (q0) **187**, binary (q1) **225**.

### Dominant error transitions (old → corrected)

| transition | count | meaning |
|---|---|---|
| No → Yes | 229 | binary statement is a verbatim premise / direct entailment, mislabeled "No" |
| Unknown → A/B/C/D | 142 | MCQ gold "Unknown" — not even a valid option; a letter is provably correct |
| No → Unknown | 13 | over-claimed "no" where statement is genuinely undetermined |
| Yes → No | 3 | rare reverse |
| letter → letter | ~18 | wrong MCQ letter (e.g. picked tautology/negation distractor) |

## Root-cause patterns (consistent across all 12 subagents)

1. **Binary "No" on a true statement.** The statement in
   "is the following statement true?" is frequently a VERBATIM premise or a
   one-step modus-ponens / contrapositive consequence, yet labeled "No".
   The item's `explanation` explicitly says "so the statement is true".
   Example (pos 87): statement "If someone understands the lecture, then
   they pass the exam" = premise 1 verbatim; gold "No"; explanation "So the
   statement is true." → corrected to "Yes".

2. **MCQ labeled "Unknown".** "Unknown" is not a valid MCQ option, yet ~142
   MCQ items used it. In each the explanation names a concrete letter that
   is entailed (usually the option restating a premise or the end of a
   universal/contrapositive chain). → corrected to that letter.

## Confidence & conservatism

- Only high+medium confidence mismatches were applied. Low-confidence
  (13) were left as the original gold to avoid over-correction.
- A handful of subagent calls noted genuinely-correct golds (e.g. "No"
  where the target predicate never appears as a rule consequent, or "No"
  for necessary-vs-sufficient under closed-world); those were NOT changed.
- Multi-step subagent reasoning + explanation cross-check means each
  applied change has a cited premise-based justification in the audit file.

## Implications

1. **Evaluate the solver against the CORRECTED labels.** The raw-label
   accuracy (~22-40%) was largely an artifact of bad labels; true
   capability is materially higher (see eval delta in session report).
2. **Never tune the solver to the ORIGINAL labels** — that fits noise.
3. If this is the official challenge dataset, the label inversion
   (especially the "is the statement true?" → "No" pattern and the MCQ
   "Unknown") should be reported to organizers.

## Verification

- `data/...corrected.json` integrity checked: 411 items, premises and
  questions byte-identical to original, 281 items with ≥1 answer changed,
  pos-87 spot check shows ['Unknown','No'] → ['A','Yes'].

Adjudication was performed by reasoning agents over the provided dataset;
no external data sources.
