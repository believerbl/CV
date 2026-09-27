"""
model_experiment.py
===================
Controlled ML Matcher experiment for Amazon ML Challenge 2026 (Step 4).
Evaluates the Fast Tree Baseline (HistGradientBoostingClassifier) on engineered
30-dimensional pair features against entity-level Macro F0.5.

Experiment Pipeline (per SRS):
  validation S1s
        ↓
  Nidhi normalization (norm-v1.2.0)
        ↓
  Blocking A-E + bounded K (max_candidates_per_s1=30)
        ↓
  30-dim pair features (features.py)
        ↓
  Fast Tree Matcher (model.py: HistGradientBoostingClassifier)
        ↓
  Decision threshold sweep (tau in [0.1..0.9])
        ↓
  Official Entity-Level Macro F0.5 (evaluate.py)

Usage:
  python model_experiment.py [--out-dir reports]
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

# Ensure src modules are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "code", "business_entity_resolution", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from normalize import normalize_dataframe_rows, NORMALIZATION_VERSION
from blocking import generate_candidates
from features import FEATURE_NAMES, build_feature_vector
from model import train_model, predict_scores, build_training_dataset
from evaluate import evaluate_predictions


def build_experiment_fixture() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Set[str]]]:
    """Self-consistent benchmark fixture mirroring real train data noise patterns.

    Contains:
      - S1-001: Acme Corp -> S2-001 (Acme LLC, suffix diff) + S3-001 (Acme Inc)
      - S1-002: Zxqwv Traders -> S2-002 (Zxqwv Foods, rare token match)
      - S1-003: Blue Ocean Traders -> S2-003 (numeric/postal match)
      - S1-004: Continental Traders -> S2-004 (Continantal typo)
      - S1-005: Sharma Textiles (Devanagari script pair) -> S2-005
      - S1-006: Alpha Retail Store -> S2-007 + S3-003 (address match)
      - S1-007: Tech Innovators Inc -> S2-008 (tech innovators)
      - S1-008: Global Export Hub -> S2-009 (global export)
      - S1-009: Singleton Entity (No match in S2 or S3) -> empty true set
      - S1-010: Unmatched Entity 2 -> empty true set
    """
    s1 = [
        {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield, 62704", "country": "India"},
        {"entity_id": "S1-002", "business_name": "Zxqwv Traders", "business_address": "2 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-003", "business_name": "Blue Ocean Traders", "business_address": "500 Market St, 62704", "country": "India"},
        {"entity_id": "S1-004", "business_name": "Continental Traders", "business_address": "3 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-005", "business_name": "शर्मा टेक्सटाइल्स", "business_address": "4 Main St, Delhi, 110001", "country": "India"},
        {"entity_id": "S1-006", "business_name": "Alpha Retail Store", "business_address": "100 Market St Suite 5", "country": "US"},
        {"entity_id": "S1-007", "business_name": "Tech Innovators Inc", "business_address": "25 Innovation Way", "country": "US"},
        {"entity_id": "S1-008", "business_name": "Global Export Hub", "business_address": "88 Harbor Blvd", "country": "India"},
        {"entity_id": "S1-009", "business_name": "Completely Unique Singleton", "business_address": "99 Nowhere Lane", "country": "US"},
        {"entity_id": "S1-010", "business_name": "Lone Star Outpost", "business_address": "77 Desert Road", "country": "India"},
    ]

    s2 = [
        {"entity_id": "S2-001", "business_name": "Acme LLC", "business_address": "1 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-002", "business_name": "Zxqwv Foods", "business_address": "2 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-003", "business_name": "Blue Ocean Products", "business_address": "500 Market St, 62704", "country": "India"},
        {"entity_id": "S2-004", "business_name": "Continantal Traders", "business_address": "3 Main St, Springfield", "country": "India"},
        {"entity_id": "S2-005", "business_name": "शर्मा ट्रेडर्स", "business_address": "4 Main St, Delhi, 110001", "country": "India"},
        {"entity_id": "S2-006", "business_name": "Unrelated Business One", "business_address": "999 Far Away Road", "country": "India"},
        {"entity_id": "S2-007", "business_name": "Alpha Retail Store", "business_address": "100 Market St Unit 5", "country": "US"},
        {"entity_id": "S2-008", "business_name": "Tech Innovators Corp", "business_address": "25 Innovation Way", "country": "US"},
        {"entity_id": "S2-009", "business_name": "Global Export Hub LLC", "business_address": "88 Harbor Blvd", "country": "India"},
        {"entity_id": "S2-010", "business_name": "Random Distractor LLC", "business_address": "123 Random Way", "country": "US"},
    ]

    s3 = [
        {"entity_id": "S3-001", "business_name": "Acme Incorporated", "business_address": "1 Main St, Springfield", "country": "India"},
        {"entity_id": "S3-002", "business_name": "Unrelated Business Two", "business_address": "456 Other St", "country": "US"},
        {"entity_id": "S3-003", "business_name": "Alpha Retail LLC", "business_address": "100 Market St Suite 5", "country": "US"},
        {"entity_id": "S3-004", "business_name": "Global Export Partners", "business_address": "88 Harbor Blvd", "country": "India"},
    ]

    gt = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002"},
        "S1-003": {"S2-003"},
        "S1-004": {"S2-004"},
        "S1-005": {"S2-005"},
        "S1-006": {"S2-007", "S3-003"},
        "S1-007": {"S2-008"},
        "S1-008": {"S2-009", "S3-004"},
        "S1-009": set(),  # Singleton
        "S1-010": set(),  # Singleton
    }

    return s1, s2, s3, gt


def run_experiment(out_dir: str = "reports") -> Dict[str, Any]:
    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Load Data & Ground Truth
    s1_raw, s2_raw, s3_raw, gt = build_experiment_fixture()

    # Split S1 into Train and Validation entities (preserving complete S1 entities)
    train_s1_ids = {"S1-001", "S1-002", "S1-003", "S1-004", "S1-005", "S1-009"}
    val_s1_ids = {"S1-006", "S1-007", "S1-008", "S1-010"}

    # 2. Normalize records (timed)
    t_norm_start = time.perf_counter()
    s1_norm = normalize_dataframe_rows(s1_raw)
    s2_norm = normalize_dataframe_rows(s2_raw)
    s3_norm = normalize_dataframe_rows(s3_raw)
    t_norm = time.perf_counter() - t_norm_start

    s1_lookup = {r.entity_id: r for r in s1_norm}
    cand_lookup = {r.entity_id: r for r in s2_norm + s3_norm}

    # 3. Blocker Candidate Generation (Passes A-E, max_candidates_per_s1=30)
    t_block_start = time.perf_counter()
    blocking_config = {
        "pass_b_enabled": True,
        "pass_c_enabled": True,
        "pass_d_enabled": True,
        "pass_e_enabled": True,
        "max_candidates_per_s1": 30,
    }
    candidate_map = generate_candidates(
        s1_records=s1_norm,
        s2_records=s2_norm,
        s3_records=s3_norm,
        config=blocking_config,
    )
    t_block = time.perf_counter() - t_block_start

    # Partition candidate map into train and val
    train_candidate_map = {sid: cands for sid, cands in candidate_map.items() if sid in train_s1_ids}
    val_candidate_map = {sid: cands for sid, cands in candidate_map.items() if sid in val_s1_ids}

    # Blocker candidate recall check on val
    val_gt = {sid: gt[sid] for sid in val_s1_ids}
    val_pos_total = sum(len(matches) for matches in val_gt.values())
    val_pos_retained = 0
    for sid, true_matches in val_gt.items():
        cands_set = set(val_candidate_map.get(sid, []))
        val_pos_retained += len(true_matches & cands_set)
    cand_recall = val_pos_retained / val_pos_total if val_pos_total > 0 else 1.0

    # 4. Feature Extraction & Model Training on Train Split
    t_train_start = time.perf_counter()
    X_train, y_train, train_pair_ids = build_training_dataset(
        candidate_map=train_candidate_map,
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
    t_train = time.perf_counter() - t_train_start

    # 5. Predict Scores on Validation Candidates
    t_pred_start = time.perf_counter()
    val_scores = predict_scores(
        model=model,
        candidate_map=val_candidate_map,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
    )
    t_pred = time.perf_counter() - t_pred_start

    # 6. Threshold Sweep across tau in [0.1 .. 0.9]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = []
    best_tau = 0.5
    best_f05 = -1.0
    best_report = None

    for tau in thresholds:
        pred_matches: Dict[str, Set[str]] = {}
        for sid in val_s1_ids:
            scores_dict = val_scores.get(sid, {})
            # Candidates exceeding tau threshold
            selected = {cid for cid, score in scores_dict.items() if score >= tau}
            pred_matches[sid] = selected

        eval_report = evaluate_predictions(val_gt, pred_matches)
        sweep_results.append({
            "threshold": tau,
            "macro_f05": eval_report.macro_f05,
            "macro_precision": eval_report.macro_precision,
            "macro_recall": eval_report.macro_recall,
            "singleton_accuracy": eval_report.singleton_accuracy,
            "micro_f05": eval_report.micro_f05,
        })

        if eval_report.macro_f05 > best_f05:
            best_f05 = eval_report.macro_f05
            best_tau = tau
            best_report = eval_report

    t_total = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "experiment_id": "EXP-STEP4-001",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "blocker_passes": ["Pass A (Exact Core)", "Pass B (Rare Token)", "Pass C (Address Token)", "Pass D (Typo)", "Pass E (Devanagari)"],
        "max_candidates_per_s1_K": 30,
        "feature_set": f"30-dim ({len(FEATURE_NAMES)} features: lexical, address, numeric, missingness, country, script)",
        "model_architecture": "Fast Tree Baseline (HistGradientBoostingClassifier, random_state=42)",
        "training_pairs_count": len(X_train),
        "validation_entities_count": len(val_s1_ids),
        "candidate_pool_recall": round(cand_recall, 4),
        "threshold_sweep": sweep_results,
        "best_operating_point": {
            "threshold": best_tau,
            "macro_f05": best_report.macro_f05 if best_report else 0.0,
            "macro_precision": best_report.macro_precision if best_report else 0.0,
            "macro_recall": best_report.macro_recall if best_report else 0.0,
            "singleton_accuracy": best_report.singleton_accuracy if best_report else 0.0,
        },
        "performance": {
            "runtime_seconds_total": round(t_total, 4),
            "runtime_normalization": round(t_norm, 4),
            "runtime_blocking": round(t_block, 4),
            "runtime_train": round(t_train, 4),
            "runtime_predict": round(t_pred, 4),
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        },
    }

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "model_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(out_dir, "model_experiment_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Experiment Report — Step 4 Fast Tree Baseline\n\n")
        f.write(f"- **Experiment ID**: `{results['experiment_id']}`\n")
        f.write(f"- **Timestamp (UTC)**: `{results['timestamp_utc']}`\n")
        f.write(f"- **Normalization Version**: `{results['normalization_version']}`\n")
        f.write(f"- **Blocker Passes**: {', '.join(results['blocker_passes'])}\n")
        f.write(f"- **Candidate Cap ($K$)**: `{results['max_candidates_per_s1_K']}`\n")
        f.write(f"- **Feature Dimension**: `{results['feature_set']}`\n")
        f.write(f"- **Model**: `{results['model_architecture']}`\n")
        f.write(f"- **Candidate Recall**: `{results['candidate_pool_recall'] * 100:.1f}%`\n\n")

        f.write(r"## Threshold Sweep ($\tau$) vs Macro $F_{0.5}$" + "\n\n")
        f.write(r"| Threshold ($\tau$) | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc |" + "\n")
        f.write("|---|---|---|---|---|\n")
        for row in sweep_results:
            f.write(f"| {row['threshold']:.2f} | **{row['macro_f05']:.4f}** | {row['macro_precision']:.4f} | {row['macro_recall']:.4f} | {row['singleton_accuracy']:.4f} |\n")

        f.write(f"\n### Best Operating Point: $\\tau = {best_tau:.2f}$\n")
        f.write(f"- **Macro $F_{0.5}$**: `{results['best_operating_point']['macro_f05']:.4f}`\n")
        f.write(f"- **Macro Precision**: `{results['best_operating_point']['macro_precision']:.4f}`\n")
        f.write(f"- **Macro Recall**: `{results['best_operating_point']['macro_recall']:.4f}`\n")
        f.write(f"- **Singleton Accuracy**: `{results['best_operating_point']['singleton_accuracy']:.4f}`\n\n")

        f.write("## Runtime & Resource Diagnostics\n\n")
        f.write(f"- **Total Runtime**: `{results['performance']['runtime_seconds_total']}s`\n")
        f.write(f"  - Normalization: `{results['performance']['runtime_normalization']}s`\n")
        f.write(f"  - Blocker: `{results['performance']['runtime_blocking']}s`\n")
        f.write(f"  - Model Train: `{results['performance']['runtime_train']}s`\n")
        f.write(f"  - Model Predict: `{results['performance']['runtime_predict']}s`\n")
        f.write(f"- **Peak Memory**: `{results['performance']['peak_memory_mb']} MB`\n")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    res = run_experiment(out_dir=args.out_dir)
    print(f"Experiment finished successfully. Results saved to {args.out_dir}/")
    print(f"Best Macro F0.5: {res['best_operating_point']['macro_f05']:.4f} at tau={res['best_operating_point']['threshold']}")
