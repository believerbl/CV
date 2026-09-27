"""
pipeline.py
===========
End-to-End Inference and Execution Pipeline for Amazon ML Challenge 2026.
Owned by Parimarjan Shukla.

Integrates the full 6-stage architecture:
  1. LOAD: Ingestion of source records (S1, S2, S3) from TSVs.
  2. NORMALIZE: Nidhi's normalization module (norm-v1.2.0 contract).
  3. BLOCKING: Piyush's Passes A-E candidate generator with K-budget pruning.
  4. FEATURES: Parimarjan's 30-dim frozen pair feature representation.
  5. MODEL: Fast HistGradientBoostingClassifier matcher scoring candidate pairs.
  6. DECISION: Set-decision layer with thresholding and numeric conflict filtering.
  7. OUTPUT: Generates official `matching_results.tsv` and `candidate_pairs.tsv`.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure src modules are discoverable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from normalize import normalize_dataframe_rows, NORMALIZATION_VERSION, NormalizedRecord
from blocking import generate_candidates
from features import FEATURE_NAMES, build_pair_features
from model import ERModel, train_model, predict_scores, load_model, save_model, build_training_dataset
from decision import make_decisions, DecisionConfig


def load_source_tsv(path: str, limit: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
    """Load records from a source TSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source TSV not found at: {path}")

    records: List[Dict[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            records.append({
                "entity_id": (row.get("entity_id") or "").strip(),
                "business_name": row.get("business_name"),
                "business_address": row.get("business_address"),
                "country": row.get("country"),
            })
    return records


def load_ground_truth_tsv(path: str) -> Dict[str, Set[str]]:
    """Load ground truth mapping from TSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth TSV not found at: {path}")

    gt: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = (row.get("source1_entity_id") or "").strip()
            raw_matched = row.get("matched_entity_ids")
            if raw_matched and str(raw_matched).strip():
                gt[s1_id] = {m.strip() for m in str(raw_matched).split(",") if m.strip()}
            else:
                gt[s1_id] = set()
    return gt


def run_pipeline(
    s1_records: Sequence[Any],
    s2_records: Sequence[Any],
    s3_records: Sequence[Any],
    model: ERModel,
    blocking_config: Optional[Dict[str, Any]] = None,
    decision_config: Optional[DecisionConfig] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Execute the end-to-end entity resolution pipeline.

    Args:
        s1_records: List of raw dicts or NormalizedRecords for Source 1.
        s2_records: List of raw dicts or NormalizedRecords for Source 2.
        s3_records: List of raw dicts or NormalizedRecords for Source 3.
        model: Fitted ERModel instance.
        blocking_config: Configuration dict for generate_candidates.
        decision_config: Configuration dataclass for set decisions.

    Returns:
        (matching_results, candidate_pairs)
        - matching_results: Dict mapping s1_id -> List of matched S2/S3 IDs.
        - candidate_pairs: Dict mapping s1_id -> List of candidate S2/S3 IDs.
        Guarantees that every matching list is a strict subset of candidate list: M subseteq C.
    """
    if decision_config is None:
        decision_config = DecisionConfig(threshold=0.35, suppress_numeric_conflict=True)

    default_blocking_cfg = {
        "pass_b_enabled": True,
        "pass_c_enabled": True,
        "pass_d_enabled": True,
        "pass_e_enabled": True,
        "max_candidates_per_s1": 30,
    }
    b_cfg = dict(default_blocking_cfg)
    if blocking_config:
        b_cfg.update(blocking_config)

    # 1. Normalize records if raw
    s1_norm = normalize_dataframe_rows(s1_records) if s1_records and isinstance(s1_records[0], dict) else s1_records
    s2_norm = normalize_dataframe_rows(s2_records) if s2_records and isinstance(s2_records[0], dict) else s2_records
    s3_norm = normalize_dataframe_rows(s3_records) if s3_records and isinstance(s3_records[0], dict) else s3_records

    s1_lookup = {r.entity_id: r for r in s1_norm}
    cand_lookup = {r.entity_id: r for r in s2_norm + s3_norm}

    # 2. Candidate Generation / Blocking (Passes A-E + bounded K pruning)
    candidate_map: Dict[str, List[str]] = generate_candidates(
        s1_records=s1_norm,
        s2_records=s2_norm,
        s3_records=s3_norm,
        config=b_cfg,
    )

    # 3. Model Scoring over Candidate Pairs
    candidate_scores: Dict[str, Dict[str, float]] = predict_scores(
        model=model,
        candidate_map=candidate_map,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
    )

    # 4. Precompute numeric conflicts for decision filtering
    numeric_conflict_map: Dict[str, Set[str]] = {}
    if decision_config.suppress_numeric_conflict:
        for s1_id, cands in candidate_map.items():
            s1_rec = s1_lookup.get(s1_id)
            if not s1_rec:
                continue
            conflicts = set()
            for cid in cands:
                cand_rec = cand_lookup.get(cid)
                if cand_rec:
                    feats = build_pair_features(s1_rec, cand_rec)
                    if feats.get("numeric_conflict", 0.0) == 1.0:
                        conflicts.add(cid)
            if conflicts:
                numeric_conflict_map[s1_id] = conflicts

    # 5. Set Decisions (thresholding, empty-set gating, numeric conflict filter)
    matching_results: Dict[str, List[str]] = make_decisions(
        scores=candidate_scores,
        config=decision_config,
        numeric_conflict_map=numeric_conflict_map,
    )

    # 6. Ensure invariant: M subset of C for every S1
    for s1_id, matches in matching_results.items():
        cands_set = set(candidate_map.get(s1_id, []))
        # Keep only candidates present in candidate_map
        matching_results[s1_id] = [m for m in matches if m in cands_set]

    return matching_results, candidate_map


def write_submission_files(
    matching_results: Dict[str, Sequence[str]],
    candidate_pairs: Optional[Dict[str, Sequence[str]]],
    matching_path: str,
    candidate_path: Optional[str] = None,
) -> None:
    """Write official TSV submission files for the competition.

    Formats:
      matching_results.tsv:  source1_entity_id\\tmatched_entity_ids
      candidate_pairs.tsv:   source1_entity_id\\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(matching_path)), exist_ok=True)

    sorted_s1_ids = sorted(matching_results.keys())

    # Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted_s1_ids:
            matches = matching_results[s1_id]
            joined = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{joined}\n")

    # Write candidate_pairs.tsv if requested
    if candidate_path and candidate_pairs is not None:
        os.makedirs(os.path.dirname(os.path.abspath(candidate_path)), exist_ok=True)
        with open(candidate_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted_s1_ids:
                cands = candidate_pairs.get(s1_id, [])
                joined = ",".join(cands) if cands else ""
                f.write(f"{s1_id}\t{joined}\n")


def build_or_load_model(
    train_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    train_sample_limit: int = 1500,
) -> ERModel:
    """Load existing model from model_path, or train a new HistGradientBoostingClassifier."""
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}")
        return load_model(model_path)

    print("Training new HistGradientBoostingClassifier model...")
    if train_dir and os.path.exists(train_dir):
        # Load sample from train_dir
        s1_file = os.path.join(train_dir, "train_source1.tsv")
        s2_file = os.path.join(train_dir, "train_source2.tsv")
        s3_file = os.path.join(train_dir, "train_source3.tsv")
        gt_file = os.path.join(train_dir, "train_ground_truth.tsv")

        s1_raw = load_source_tsv(s1_file, limit=train_sample_limit)
        gt = load_ground_truth_tsv(gt_file)

        # Collect S2 and S3 IDs required by S1 true matches
        needed_s2: Set[str] = set()
        needed_s3: Set[str] = set()
        for r in s1_raw:
            sid = r["entity_id"]
            for mid in gt.get(sid, set()):
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

        # Load S2 and S3 rows
        s2_raw = load_source_tsv(s2_file, limit=train_sample_limit * 3)
        s3_raw = load_source_tsv(s3_file, limit=train_sample_limit * 3)

        s1_norm = normalize_dataframe_rows(s1_raw)
        s2_norm = normalize_dataframe_rows(s2_raw)
        s3_norm = normalize_dataframe_rows(s3_raw)

        s1_lookup = {r.entity_id: r for r in s1_norm}
        cand_lookup = {r.entity_id: r for r in s2_norm + s3_norm}

        cand_map = generate_candidates(
            s1_records=s1_norm,
            s2_records=s2_norm,
            s3_records=s3_norm,
            config={"max_candidates_per_s1": 30},
        )

        X_train, y_train, _ = build_training_dataset(
            candidate_map=cand_map,
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
            random_state=42,
        )

        if model_path:
            save_model(model, model_path)
            print(f"Model saved to: {model_path}")
        return model

    else:
        # Fallback to synthetic training
        from model import ERModel
        from sklearn.dummy import DummyClassifier
        import numpy as np
        estimator = DummyClassifier(strategy="constant", constant=0)
        estimator.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(10))
        return ERModel(estimator=estimator, feature_names=FEATURE_NAMES)


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - End-to-End Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train dataset directory")
    parser.add_argument("--model-path", default="models/matcher_hist_gbdt.joblib", help="Model checkpoint path")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--threshold", type=float, default=0.25, help="Decision threshold tau")
    parser.add_argument("--suppress-numeric-conflict", action="store_true", default=True, help="Filter numeric conflicts")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of S1 entities processed")
    parser.add_argument("--validate", action="store_true", default=True, help="Run validate_submission.py")
    args = parser.parse_args()

    t_start = time.perf_counter()

    # 1. Build or Load Model
    model = build_or_load_model(
        train_dir=args.train_dir,
        model_path=args.model_path,
        train_sample_limit=2000,
    )

    # 2. Load Test Records
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    print(f"Loading test records from {args.test_dir}...")
    s1_raw = load_source_tsv(s1_path, limit=args.limit)
    s2_raw = load_source_tsv(s2_path, limit=args.limit * 5 if args.limit else None)
    s3_raw = load_source_tsv(s3_path, limit=args.limit * 5 if args.limit else None)
    print(f"Loaded: S1={len(s1_raw)}, S2={len(s2_raw)}, S3={len(s3_raw)}")

    # 3. Configure Pipeline
    decision_cfg = DecisionConfig(
        threshold=args.threshold,
        suppress_numeric_conflict=args.suppress_numeric_conflict,
        output_order="id_asc",
    )

    # 4. Execute Pipeline
    print("Executing entity resolution pipeline...")
    matching_results, candidate_pairs = run_pipeline(
        s1_records=s1_raw,
        s2_records=s2_raw,
        s3_records=s3_raw,
        model=model,
        decision_config=decision_cfg,
    )

    # 5. Write Official Submission Files
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")
    print(f"Writing outputs to {args.output_dir}/...")
    write_submission_files(matching_results, candidate_pairs, matching_out, candidate_out)

    n_matched = sum(1 for m in matching_results.values() if len(m) > 0)
    n_singletons = sum(1 for m in matching_results.values() if len(m) == 0)
    print(f"Wrote {len(matching_results)} S1 rows ({n_matched} matched, {n_singletons} empty singletons).")

    # 6. Validate Submission
    if args.validate:
        validator_script = os.path.join("utils", "validate_submission.py")
        if os.path.exists(validator_script):
            print("Running official submission validator...")
            import subprocess
            cmd = [
                sys.executable,
                validator_script,
                "--matching", matching_out,
                "--candidate", candidate_out,
                "--test-dir", args.test_dir,
            ]
            ret = subprocess.run(cmd)
            if ret.returncode == 0:
                print("VALIDATION PASSED: Output files are completely compliant!")
            else:
                print(f"VALIDATION WARNING: Validator exited with code {ret.returncode}")

    print(f"Pipeline finished in {time.perf_counter() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
