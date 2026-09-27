# Model Experiment Report — Step 4 Fast Tree Baseline

- **Experiment ID**: `EXP-STEP4-001`
- **Timestamp (UTC)**: `2026-09-27T14:50:40.697679+00:00`
- **Normalization Version**: `norm-v1.2.0`
- **Blocker Passes**: Pass A (Exact Core), Pass B (Rare Token), Pass C (Address Token), Pass D (Typo), Pass E (Devanagari)
- **Candidate Cap ($K$)**: `30`
- **Feature Dimension**: `30-dim (30 features: lexical, address, numeric, missingness, country, script)`
- **Model**: `Fast Tree Baseline (HistGradientBoostingClassifier, random_state=42)`
- **Candidate Recall**: `100.0%`

## Threshold Sweep ($\tau$) vs Macro $F_{0.5}$

| Threshold ($\tau$) | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc |
|---|---|---|---|---|
| 0.10 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.20 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.30 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.40 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.50 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.60 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.70 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.80 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.90 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

### Best Operating Point: $\tau = 0.10$
- **Macro $F_0.5$**: `1.0000`
- **Macro Precision**: `1.0000`
- **Macro Recall**: `1.0000`
- **Singleton Accuracy**: `1.0000`

## Runtime & Resource Diagnostics

- **Total Runtime**: `1.6255s`
  - Normalization: `0.0023s`
  - Blocker: `0.002s`
  - Model Train: `1.6148s`
  - Model Predict: `0.0058s`
- **Peak Memory**: `0.74 MB`
