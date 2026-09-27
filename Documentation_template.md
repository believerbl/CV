# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CV  
**Team Members:** Parimarjan Shukla, Nidhi Singh  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary
We present an end-to-end, high-precision entity resolution system designed for the Amazon ML Challenge 2026. The solution integrates a 5-pass bounded blocking pipeline, a 30-dimensional engineered pair feature representation capturing lexical, structural, phonetic, and numeric agreement, a gradient-boosted tree matcher, and a precision-weighted decision policy. Operating under strict physical memory constraints (< 1.5 GB), our streaming pipeline successfully resolved all 1,732,544 test $S_1$ reference entities against ~9.97 million candidate records, completely passing all official submission validator checks with zero candidate-subset violations and zero nonexistent IDs.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory analysis of the multi-source dataset revealed substantial noise and heterogeneity across sources:
- **Lexical and Structural Noise in Names:** Heavy usage of abbreviations (`Pvt`, `Ltd`, `Corp`, `Co`), punctuation variants (`&` vs `and`), and legal entity suffixes.
- **Address Heterogeneity:** Landmark-based addresses without formal street numbers in India, missing postal codes, and municipal numbering differences.
- **Cross-Source Asymmetry:** Source 1 is deduplicated reference data, while Sources 2 and 3 contain partial fragments, spelling typos, and variable formatting.
- **Strict Invariants:** Ground truth analysis confirmed a 100% same-country matching invariant (0% cross-country true matches).
- **Metric Asymmetry:** The target metric is macro-averaged $F_{0.5}$, which weights precision twice as heavily as recall ($1.25 \times P \times R / (0.25 P + R)$). A false merge (wrong business match) severely damages the score compared to an omitted link, making precision gating paramount.

### 2.2 Solution Strategy
Our architecture follows a modular, reproducible 6-stage pipeline:
1. **Normalization (`norm-v1.2.0`):** Deterministic unicode normalization, lowercase conversion, legal suffix extraction, phone/PIN extraction, and city/country harmonization.
2. **Blocking (Passes A–E):** Multi-pass inverted index generation combining exact core name, normalized name, token set, and phonetics, capped at a maximum of $K=30$ candidates per $S_1$.
3. **Pair Feature Engineering:** 30 frozen, domain-tailored features spanning token Jaccard similarities, Levenshtein edit distances (with prefix and length cutoffs), token sort ratios, postal code matches, and numeric contradiction indicators.
4. **Matcher Model:** A fast `HistGradientBoostingClassifier` trained on candidate pairs with negative downsampling, generating well-calibrated match probabilities.
5. **Set Decision Layer:** Variable-cardinality decision rule with singleton gating ($\tau = 0.25$), empty-set preservation, and strict suppression of candidate pairs exhibiting numeric address/unit conflict.
6. **Streaming Inference Engine:** SQLite on-disk candidate caching and vectorized batch feature scoring to guarantee sub-1.5 GB RAM consumption across 1.73M entities.

**Approach Type:** Multi-Pass Blocking + 30-D Pair Feature GBDT Classifier + Metric-Optimized Set Decision Layer  
**Core Innovation:** Precision-first architecture pairing bounded multi-index blocking with explicit numeric-conflict suppression and singleton gating tailored to Macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To prune the $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ comparison space, we employ a 5-pass blocking scheme:
- **Pass A (Core & Norm Match):** Exact matching on normalized core business name and full normalized name.
- **Pass B (Relaxed Token Jaccard):** Significant token overlap on core business names.
- **Pass C (Phonetic / Double Metaphone):** Sound-alike name phonetic clustering.
- **Pass D (Address / Postal Pin Blocking):** High address token overlap within identical country and postal boundaries.
- **Pass E (Acronym & Prefix Anchor):** Anchor initials and character 3-gram prefix indices.

**Key Parameters & Properties:**
- **Country Filter:** Strict country equality filtering, enforcing the ground-truth invariant.
- **Candidate Budget:** Bounded to a maximum of $K = 30$ candidates per $S_1$ entity.
- **Candidate Pairs Generated:** 1,582,276 $S_1$ entities received candidate sets, while 150,268 singletons with zero index matches were cleanly routed to empty candidate sets.
- **Recall Preservation:** Validation benchmarks demonstrated 100% ground-truth candidate recall across all multi-pass configurations.

---

## 4. Matching Model

### Features Used (30 Dimensions)
- **Name Features (12):** Token Jaccard, token overlap coefficient, character 3-gram similarity, length-capped Levenshtein edit similarity, token sort ratio, token set ratio, exact match indicator, prefix similarity, core name match.
- **Address Features (10):** Address token Jaccard, address edit similarity, postal code exact match, city match, street number match, building/suite agreement.
- **Consistency & Contradiction Signals (8):** Same-country indicator, numeric contradiction indicator (differing street/unit numbers), token difference count, name length ratio, address length ratio, missing address penalty.

### Model Architecture & Training
- **Model Type:** `HistGradientBoostingClassifier` (`max_iter=100`, `learning_rate=0.08`, `min_samples_leaf=20`, `l2_regularization=1.0`).
- **Training Strategy:** Balanced candidate sampling from training ground truth (1:3 positive-to-negative ratio).
- **Threshold Selection:** Global threshold optimization sweep evaluating $\tau \in [0.10, 0.90]$ against entity-level Macro $F_{0.5}$. A decision threshold of $\tau = 0.25$ combined with numeric-conflict suppression delivered optimal performance, achieving a validation Macro $F_{0.5}$ of **0.9429** with zero false positive merges.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ (Validation):** **0.9429** (Precision: 1.0000, Recall: 0.8667, Singleton Accuracy: 1.0000).
- **Full Test Set Output:**
  - Total $S_1$ Entities: 1,732,544
  - Entities with Predicted Matches: 690,130
  - Singletons Predicted (Empty Match Set): 1,042,414
  - Total Predicted Matches: 2,090,100
- **Validation Audit:**
  - Official Submission Validator: **`PASS — no blocking issues found. Safe to submit.`**
  - ID Existence Verification (`--check-ids`): **PASS across all 9,969,589 candidate IDs with 0 missing IDs.**
  - Candidate-Subset Invariant ($M \subseteq C$): **PASS with 0 violations.**
- **Error Analysis:**
  - *False Positives:* Virtually eliminated by numeric-conflict suppression (e.g., rejecting "Store #101" vs "Store #102" despite identical name and street).
  - *False Negatives:* Primarily attributable to extreme address omission or heavy transliteration noise where neither phonetic nor token indices align.

---

## 6. Conclusion
The developed system demonstrates that an entity resolution architecture combining deterministic multi-pass blocking, interpretable pair-feature engineering, and a metric-aligned decision layer can achieve exceptional precision and scalability. The entire pipeline was executed on commercial hardware without external data lookups, strictly satisfying all Amazon ML Challenge rules and producing fully validated submission artifacts.

---

## Appendix

### A. Code Artefacts
The complete runnable source code is organized under `code/business_entity_resolution/`:
- `src/normalize.py`: Normalization contract (`norm-v1.2.0`).
- `src/blocking.py`: Multi-pass candidate blocker (Passes A–E, $K=30$).
- `src/features.py`: 30-dim pair feature extractor.
- `src/model.py`: GBDT candidate matcher.
- `src/decision.py`: Macro $F_{0.5}$ set decision logic.
- `src/pipeline.py`: Streaming end-to-end execution engine.
- `tests/`: 150 unit tests verifying all modules.

**Reproduction Command:**
```powershell
$env:PYTHONPATH="code\business_entity_resolution\src"
python code/business_entity_resolution/src/pipeline.py `
  --train-dir dataset/train `
  --test-dir dataset/test `
  --output-dir output
```

### B. Additional Results
| Experiment ID | Policy | $\tau$ | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `EXP-STEP5-BASELINE` | Standard Threshold | 0.25 | 0.8909 | 0.9000 | 0.8667 | 1.0000 |
| `EXP-STEP5-SUPPRESS` | With Numeric Conflict Suppression | 0.25 | **0.9429** | **1.0000** | 0.8667 | **1.0000** |
