"""
threshold_experiment.py
=======================
Systematic threshold sweep experiment for Amazon ML Challenge 2026 (Step 5).
Evaluates decision threshold tau across [0.10, 0.15, ..., 0.90] and numeric
conflict suppression policies against entity-level Macro F0.5.

Strict Isolation Protocol (per SRS):
  same validation S1
        ↓
  same normalized data (norm-v1.2.0)
        ↓
  same blocker (Passes A-E)
        ↓
  same K (max_candidates_per_s1=30)
        ↓
  same 30-dim feature vectors (features.py)
        ↓
  same trained model (HistGradientBoostingClassifier)
        ↓
  ONLY decision parameters change
        ↓
  official evaluate.py -> entity-level Macro F0.5

Records for each threshold:
  - Macro F0.5
  - Macro precision
  - Macro recall
  - Candidate recall
  - Empty-set rate
  - Singleton accuracy
  - False-positive count
  - Average predicted set size

Usage:
  python threshold_experiment.py [--out-dir reports]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "code", "business_entity_resolution", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from normalize import normalize_dataframe_rows, NORMALIZATION_VERSION
from blocking import generate_candidates
from features import FEATURE_NAMES, build_pair_features
from model import train_model, predict_scores, build_training_dataset
from decision import sweep_thresholds, make_decisions, DecisionConfig


def build_validation_fixture() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Set[str]]]:
    """Self-consistent multi-entity dataset with diverse match cardinalities (0, 1, 2, 3, 5+)."""
    s1 = [
        # Train entities
        {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield, 62704", "country": "India"},
        {"entity_id": "S1-002", "business_name": "Zxqwv Traders", "business_address": "2 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-003", "business_name": "Blue Ocean Traders", "business_address": "500 Market St, 62704", "country": "India"},
        {"entity_id": "S1-004", "business_name": "Continental Traders", "business_address": "3 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-005", "business_name": "शर्मा टेक्सटाइल्स", "business_address": "4 Main St, Delhi, 110001", "country": "India"},
        {"entity_id": "S1-006", "business_name": "Singleton Alpha Corp", "business_address": "100 Nowhere Blvd", "country": "India"},
        # Validation entities (Multi-match, single match, and singleton entities)
        {"entity_id": "S1-VAL-01", "business_name": "Retail Hub International", "business_address": "100 Market St Suite 5", "country": "US"},
        {"entity_id": "S1-VAL-02", "business_name": "Tech Innovation Labs", "business_address": "25 Science Park", "country": "US"},
        {"entity_id": "S1-VAL-03", "business_name": "Global Cargo & Logistics", "business_address": "88 Harbor Road", "country": "India"},
        {"entity_id": "S1-VAL-04", "business_name": "True Singleton Outpost", "business_address": "999 Remote Station", "country": "US"},
        {"entity_id": "S1-VAL-05", "business_name": "Desert Oasis Solitary", "business_address": "77 Empty Dunes", "country": "India"},
    ]

    s2 = [
        {"entity_id": "S2-001", "business_name": "Acme LLC", "business_address": "1 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-002", "business_name": "Zxqwv Foods", "business_address": "2 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-003", "business_name": "Blue Ocean Products", "business_address": "500 Market St, 62704", "country": "India"},
        {"entity_id": "S2-004", "business_name": "Continantal Traders", "business_address": "3 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-005", "business_name": "शर्मा ट्रेडर्स", "business_address": "4 Main St, Delhi, 110001", "country": "India"},
        {"entity_id": "S2-VAL-01", "business_name": "Retail Hub Intl LLC", "business_address": "100 Market St Unit 5", "country": "US"},
        {"entity_id": "S2-VAL-02", "business_name": "Retail Hub Express", "business_address": "100 Market St Suite 5", "country": "US"},
        {"entity_id": "S2-VAL-03", "business_name": "Tech Innovation Labs Inc", "business_address": "25 Science Park", "country": "US"},
        {"entity_id": "S2-VAL-04", "business_name": "Global Cargo Logistics Ltd", "business_address": "88 Harbor Road", "country": "India"},
        # Distractors & Numeric conflict cases
        {"entity_id": "S2-VAL-CONF", "business_name": "Retail Hub Intl", "business_address": "900 Market St", "country": "US"},
        {"entity_id": "S2-DISTRACTOR", "business_name": "Completely Unrelated Inc", "business_address": "123 Random Way", "country": "US"},
    ]

    s3 = [
        {"entity_id": "S3-001", "business_name": "Acme Incorporated", "business_address": "1 Main St, Springfield", "country": "India"},
        {"entity_id": "S3-VAL-01", "business_name": "Retail Hub World", "business_address": "100 Market St", "country": "US"},
        {"entity_id": "S3-VAL-02", "business_name": "Global Cargo Freight", "business_address": "88 Harbor Road", "country": "India"},
        {"entity_id": "S3-DISTRACTOR", "business_name": "Another Distractor LLC", "business_address": "456 Other Lane", "country": "India"},
    ]

    gt = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002"},
        "S1-003": {"S2-003"},
        "S1-004": {"S2-004"},
        "S1-005": {"S2-005"},
        "S1-006": set(),  # Train singleton
        "S1-VAL-01": {"S2-VAL-01", "S2-VAL-02", "S3-VAL-01"},  # 3 matches
        "S1-VAL-02": {"S2-VAL-03"},                            # 1 match
        "S1-VAL-03": {"S2-VAL-04", "S3-VAL-02"},               # 2 matches
        "S1-VAL-04": set(),                                    # True singleton
        "S1-VAL-05": set(),                                    # True singleton
    }

    return s1, s2, s3, gt


def run_threshold_experiment(out_dir: str = "reports") -> Dict[str, Any]:
    tracemalloc.start()
    t_start = time.perf_counter()

    s1_raw, s2_raw, s3_raw, gt = build_validation_fixture()

    train_s1_ids = {"S1-001", "S1-002", "S1-003", "S1-004", "S1-005", "S1-006"}
    val_s1_ids = {"S1-VAL-01", "S1-VAL-02", "S1-VAL-03", "S1-VAL-04", "S1-VAL-05"}

    # 1. Normalize
    s1_norm = normalize_dataframe_rows(s1_raw)
    s2_norm = normalize_dataframe_rows(s2_raw)
    s3_norm = normalize_dataframe_rows(s3_raw)

    s1_lookup = {r.entity_id: r for r in s1_norm}
    cand_lookup = {r.entity_id: r for r in s2_norm + s3_norm}

    # 2. Block with Passes A-E and K=30
    blocker_cfg = {
        "pass_b_enabled": True,
        "pass_c_enabled": True,
        "pass_d_enabled": True,
        "pass_e_enabled": True,
        "max_candidates_per_s1": 30,
    }
    candidate_map = generate_candidates(s1_norm, s2_norm, s3_norm, config=blocker_cfg)

    train_cands = {sid: cands for sid, cands in candidate_map.items() if sid in train_s1_ids}
    val_cands = {sid: cands for sid, cands in candidate_map.items() if sid in val_s1_ids}

    # 3. Build training features & fit model
    X_train, y_train, _ = build_training_dataset(
        candidate_map=train_cands,
        ground_truth=gt,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
        balance_ratio=3.0,
        random_state=42,
    )
    model = train_model(
        X=X_train,
        y=y_train,
        model_type="hist_gbdt",
        hyperparameters={"max_iter": 50, "min_samples_leaf": 2},
        random_state=42,
    )

    # 4. Predict validation scores
    val_scores = predict_scores(
        model=model,
        candidate_map=val_cands,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
    )

    val_gt = {sid: gt[sid] for sid in val_s1_ids}

    # Precompute numeric conflict map for validation pairs
    numeric_conflict_map: Dict[str, Set[str]] = {}
    for sid, cands in val_cands.items():
        s1_rec = s1_lookup.get(sid)
        conflicts = set()
        for cid in cands:
            cand_rec = cand_lookup.get(cid)
            if s1_rec and cand_rec:
                f_dict = build_pair_features(s1_rec, cand_rec)
                if f_dict.get("numeric_conflict", 0.0) == 1.0:
                    conflicts.add(cid)
        if conflicts:
            numeric_conflict_map[sid] = conflicts

    # 5. Threshold sweep (tau from 0.10 to 0.90 in increments of 0.05)
    thresholds = [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ]

    sweep_standard = sweep_thresholds(
        scores=val_scores,
        ground_truth=val_gt,
        thresholds=thresholds,
        suppress_numeric_conflict=False,
    )

    sweep_with_conflict_suppression = sweep_thresholds(
        scores=val_scores,
        ground_truth=val_gt,
        thresholds=thresholds,
        numeric_conflict_map=numeric_conflict_map,
        suppress_numeric_conflict=True,
    )

    # Find optimal operating point
    best_standard = max(sweep_standard, key=lambda r: r["macro_f05"])
    best_conflict_suppress = max(sweep_with_conflict_suppression, key=lambda r: r["macro_f05"])

    t_total = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "experiment_id": "EXP-STEP5-THRESHOLD-001",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "blocker_passes": ["Pass A", "Pass B", "Pass C", "Pass D", "Pass E"],
        "max_candidates_per_s1_K": 30,
        "feature_dim": len(FEATURE_NAMES),
        "validation_entities_count": len(val_s1_ids),
        "true_singleton_count": sum(1 for sid, m in val_gt.items() if len(m) == 0),
        "true_matched_entities": sum(1 for sid, m in val_gt.items() if len(m) > 0),
        "threshold_sweep_standard": sweep_standard,
        "threshold_sweep_conflict_suppression": sweep_with_conflict_suppression,
        "best_standard_policy": best_standard,
        "best_conflict_suppression_policy": best_conflict_suppress,
        "runtime_seconds": round(t_total, 4),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
    }

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "threshold_sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(out_dir, "threshold_sweep_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Threshold Sweep Report — Step 5 Decision Layer\n\n")
        f.write(f"- **Experiment ID**: `{results['experiment_id']}`\n")
        f.write(f"- **Timestamp (UTC)**: `{results['timestamp_utc']}`\n")
        f.write(f"- **Normalization**: `{results['normalization_version']}`\n")
        f.write(f"- **Blocker Passes**: A, B, C, D, E ($K=30$)\n")
        f.write(f"- **Feature Set**: {results['feature_dim']}-dim frozen pair features\n")
        f.write(f"- **Validation Population**: {results['validation_entities_count']} entities "
                f"({results['true_matched_entities']} matched + {results['true_singleton_count']} singletons)\n\n")

        f.write(r"## 1. Global Threshold Sweep ($\tau$ from 0.10 to 0.90)" + "\n\n")
        f.write("| Threshold | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Cand Rec | Empty-Set % | Singleton Acc | FP Count | Avg Set Size |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for row in sweep_standard:
            f.write(
                f"| {row['threshold']:.2f} | **{row['macro_f05']:.4f}** | {row['macro_precision']:.4f} | "
                f"{row['macro_recall']:.4f} | {row['candidate_recall']:.2f} | {row['empty_set_rate']*100:.1f}% | "
                f"{row['singleton_accuracy']:.4f} | {row['false_positive_count']} | {row['average_predicted_set_size']:.2f} |\n"
            )

        f.write("\n" + r"## 2. Policy Comparison: Standard vs Numeric Conflict Suppression" + "\n\n")
        f.write(r"| Policy | Best $\tau$ | Macro $F_{0.5}$ | Precision | Recall | Singleton Acc | FP Count |" + "\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(
            f"| Standard Thresholding | {best_standard['threshold']:.2f} | **{best_standard['macro_f05']:.4f}** | "
            f"{best_standard['macro_precision']:.4f} | {best_standard['macro_recall']:.4f} | "
            f"{best_standard['singleton_accuracy']:.4f} | {best_standard['false_positive_count']} |\n"
        )
        f.write(
            f"| With Conflict Suppression | {best_conflict_suppress['threshold']:.2f} | **{best_conflict_suppress['macro_f05']:.4f}** | "
            f"{best_conflict_suppress['macro_precision']:.4f} | {best_conflict_suppress['macro_recall']:.4f} | "
            f"{best_conflict_suppress['singleton_accuracy']:.4f} | {best_conflict_suppress['false_positive_count']} |\n"
        )

        f.write("\n## 3. Decision Layer Takeaways\n\n")
        f.write("1. **Precision weighting**: The $F_{0.5}$ metric penalizes false positives 2x more than false negatives, rewarding thresholds that eliminate low-confidence distractor links.\n")
        f.write("2. **Singleton protection**: At appropriate thresholds, singleton entities correctly collapse to `[]` yielding 1.0 entity scores, preventing catastrophic false-match penalties.\n")
        f.write("3. **Multi-match flexibility**: The layer naturally outputs variable-sized candidate sets without artificial cardinality constraints.\n")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    res = run_threshold_experiment(out_dir=args.out_dir)
    print(f"Threshold sweep complete. Results saved to {args.out_dir}/")
    print(f"Optimal threshold: tau={res['best_standard_policy']['threshold']:.2f} (Macro F0.5={res['best_standard_policy']['macro_f05']:.4f})")
