# Candidate Benchmark Report — TEST-FIXTURE / NORMALIZATION-DATASET

Dataset type: TEST-FIXTURE / NORMALIZATION-DATASET
This benchmark is NOT representative of full training-data scale.

- Generated (UTC): 2026-09-26T20:45:03.961833+00:00
- Data source: {"mode": "TEST-FIXTURE / NORMALIZATION-DATASET", "note": "Small benchmark-only fixture derived from the existing normalization/blocking test expectations. This benchmark is NOT representative of full training-data scale. Do not claim training recall."}
- Dataset: S1=8, S2=7, S3=3, S2+S3=10, total positives=8, S1 with positives=6
- Exhaustive pairs S1x(S2+S3): 80

## Benchmark methodology

- Real `generate_candidates()` called per configuration; ground truth used only afterward for recall.
- Records normalized once (0.0118s); blocker timing excludes renormalization.
- Recall is measured over the COMPLETE S1 population (unmatched S1s included in cardinality, excluded from recall numerator/denominator).
- No global K exists in blocking.py (pass-specific caps only); the K sweep is a benchmark-only post-union truncation.

## A

- Flags: `{"pass_b_enabled": false, "pass_c_enabled": false, "pass_d_enabled": false, "pass_e_enabled": false}`
- Recall: 0.5000 (retained 4/8)
- S1 with positives: 6; completely missed: 4; zero-candidate S1: 6
- Cardinality: mean 0.500, median 0.0, P95 2, max 2, total 4
- Reduction ratio: 0.950000; factor: 20.00
- Blocker runtime: 0.0002s

## A+B

- Flags: `{"pass_b_enabled": true, "pass_c_enabled": false, "pass_d_enabled": false, "pass_e_enabled": false}`
- Recall: 0.8750 (retained 7/8)
- S1 with positives: 6; completely missed: 1; zero-candidate S1: 2
- Cardinality: mean 1.125, median 1.0, P95 2, max 2, total 9
- Reduction ratio: 0.887500; factor: 8.89
- Blocker runtime: 0.0009s

## A+B+C

- Flags: `{"pass_b_enabled": true, "pass_c_enabled": true, "pass_d_enabled": false, "pass_e_enabled": false}`
- Recall: 1.0000 (retained 8/8)
- S1 with positives: 6; completely missed: 0; zero-candidate S1: 2
- Cardinality: mean 1.375, median 1.5, P95 3, max 3, total 11
- Reduction ratio: 0.862500; factor: 7.27
- Blocker runtime: 0.0025s

## A+B+C+D

- Flags: `{"pass_b_enabled": true, "pass_c_enabled": true, "pass_d_enabled": true, "pass_e_enabled": false}`
- Recall: 1.0000 (retained 8/8)
- S1 with positives: 6; completely missed: 0; zero-candidate S1: 2
- Cardinality: mean 1.375, median 1.5, P95 3, max 3, total 11
- Reduction ratio: 0.862500; factor: 7.27
- Blocker runtime: 0.0051s

## A+B+C+D+E

- Flags: `{"pass_b_enabled": true, "pass_c_enabled": true, "pass_d_enabled": true, "pass_e_enabled": true}`
- Recall: 1.0000 (retained 8/8)
- S1 with positives: 6; completely missed: 0; zero-candidate S1: 2
- Cardinality: mean 1.375, median 1.5, P95 3, max 3, total 11
- Reduction ratio: 0.862500; factor: 7.27
- Blocker runtime: 0.0058s

## Incremental pass contribution

| stage | new pairs | newly recovered TP |
|---|---|---|
| A | 4 | 4 |
| A+B | 5 | 3 |
| A+B+C | 2 | 1 |
| A+B+C+D | 0 | 0 |
| A+B+C+D+E | 0 | 0 |

## K sweep — BENCHMARK-ONLY K SWEEP (cap on final map)

| K | recall | retained | mean | median | P95 | max | total |
|---|---|---|---|---|---|---|---|
| 5 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 10 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 15 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 20 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 30 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 50 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |
| 100 | 1.0000 | 8 | 1.375 | 1.5 | 3 | 3 | 11 |

## Zero-retention analysis (final A+B+C+D+E)

- S1 with positives but zero retained: 0

## Candidate cardinality

See per-configuration cardinality above; totals are bounded by pass-specific DF suppression plus caps (no global K in blocker).

## Runtime/memory

- Normalization: 0.0118s; total benchmark: 0.0388s
- Memory method: tracemalloc.get_traced_memory (Python allocations only; not process RSS peak). Portable but approximate.
- tracemalloc current peak: 88758 bytes
- Determinism double-run identical: True

## Observations (measured only)

- Fixture mode: numbers below describe the TEST-FIXTURE / NORMALIZATION-DATASET only and must not be quoted as training recall.
- Recall by configuration: A=0.5000, A+B=0.8750, A+B+C=1.0000, A+B+C+D=1.0000, A+B+C+D+E=1.0000
- Total pairs by configuration: A=4, A+B=9, A+B+C=11, A+B+C+D=11, A+B+C+D+E=11

## Frozen recommendation data (for team review)

No best-pass claim is made here. The JSON artifact carries the full numbers; the team decides the freeze from measured recall/cardinality/cost trade-offs.

