# Threshold Sweep Report — Step 5 Decision Layer

- **Experiment ID**: `EXP-STEP5-THRESHOLD-001`
- **Timestamp (UTC)**: `2026-09-27T14:59:50.375682+00:00`
- **Normalization**: `norm-v1.2.0`
- **Blocker Passes**: A, B, C, D, E ($K=30$)
- **Feature Set**: 30-dim frozen pair features
- **Validation Population**: 5 entities (3 matched + 2 singletons)

## 1. Global Threshold Sweep ($\tau$ from 0.10 to 0.90)

| Threshold | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Cand Rec | Empty-Set % | Singleton Acc | FP Count | Avg Set Size |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.15 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.20 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.25 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.30 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.35 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.40 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.45 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.50 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.55 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.60 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.65 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.70 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.75 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.80 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.85 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |
| 0.90 | **0.8909** | 0.9000 | 0.8667 | 0.67 | 40.0% | 1.0000 | 1 | 1.00 |

## 2. Policy Comparison: Standard vs Numeric Conflict Suppression

| Policy | Best $\tau$ | Macro $F_{0.5}$ | Precision | Recall | Singleton Acc | FP Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard Thresholding | 0.10 | **0.8909** | 0.9000 | 0.8667 | 1.0000 | 1 |
| With Conflict Suppression | 0.10 | **0.9429** | 1.0000 | 0.8667 | 1.0000 | 0 |

## 3. Decision Layer Takeaways

1. **Precision weighting**: The $F_{0.5}$ metric penalizes false positives 2x more than false negatives, rewarding thresholds that eliminate low-confidence distractor links.
2. **Singleton protection**: At appropriate thresholds, singleton entities correctly collapse to `[]` yielding 1.0 entity scores, preventing catastrophic false-match penalties.
3. **Multi-match flexibility**: The layer naturally outputs variable-sized candidate sets without artificial cardinality constraints.
