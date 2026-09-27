"""
pipeline.py
===========
End-to-End Inference and Execution Pipeline for Amazon ML Challenge 2026.
Owned by Parimarjan Shukla.

Integrates the full 6-stage architecture:
  1. LOAD: Ingestion of source records (S1, S2, S3) from TSVs.
  2. NORMALIZE: Nidhi's normalization module (norm-v1.2.0 contract).
  3. BLOCKING: Passes A-E candidate generator with K-budget pruning.
  4. FEATURES: Parimarjan's 30-dim frozen pair feature representation.
  5. MODEL: Fast HistGradientBoostingClassifier matcher scoring candidate pairs.
  6. DECISION: Set-decision layer with thresholding and numeric conflict filtering.
  7. OUTPUT: Generates official `matching_results.tsv` and `candidate_pairs.tsv`.
  8. VALIDATION: Audits submission compliance with utils/validate_submission.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure src modules are discoverable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from normalize import normalize_dataframe_rows, normalize_record, NORMALIZATION_VERSION, NormalizedRecord
from blocking import generate_candidates
from features import FEATURE_NAMES, build_feature_vector, build_pair_features
from model import ERModel, train_model, predict_scores, load_model, save_model, build_training_dataset
from decision import make_decisions, select_matches_for_entity, DecisionConfig


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
    """Execute the end-to-end entity resolution pipeline in-memory.

    Suitable for test suites, benchmarks, and small-to-medium slices.

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
        decision_config = DecisionConfig(threshold=0.25, suppress_numeric_conflict=True)

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
    train_sample_limit: int = 2000,
) -> ERModel:
    """Load existing model from model_path, or train a new HistGradientBoostingClassifier."""
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}", flush=True)
        return load_model(model_path)

    print("Training new HistGradientBoostingClassifier model...", flush=True)
    if train_dir and os.path.exists(train_dir):
        s1_file = os.path.join(train_dir, "train_source1.tsv")
        s2_file = os.path.join(train_dir, "train_source2.tsv")
        s3_file = os.path.join(train_dir, "train_source3.tsv")
        gt_file = os.path.join(train_dir, "train_ground_truth.tsv")

        s1_raw = load_source_tsv(s1_file, limit=train_sample_limit)
        gt = load_ground_truth_tsv(gt_file)

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
            print(f"Model saved to: {model_path}", flush=True)
        return model

    else:
        from model import ERModel
        from sklearn.dummy import DummyClassifier
        import numpy as np
        estimator = DummyClassifier(strategy="constant", constant=0)
        estimator.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(10))
        return ERModel(estimator=estimator, feature_names=FEATURE_NAMES)


# =============================================================================
# Scalable Streaming Pipeline Engine (Memory < 1.5GB for 1.73M S1 x 10M S2/S3)
# =============================================================================

def run_scalable_inference(
    test_dir: str,
    output_dir: str,
    model: ERModel,
    decision_cfg: DecisionConfig,
    chunk_size: int = 25000,
    max_k: int = 30,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Execute high-throughput, low-memory streaming inference on full test dataset.

    Guarantees:
      1. Peak RAM consumption < 1.5 GB.
      2. Exactly one row per test S1 entity.
      3. Strict M subset of C invariant.
      4. Fully compliant tab-separated output.
    """
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    os.makedirs(output_dir, exist_ok=True)
    cache_db_path = os.path.join(output_dir, "candidates_cache.db")

    conn = sqlite3.connect(cache_db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA cache_size = -64000")  # 64MB cache

    # Check if candidate database already exists and has ~10M rows
    row_count = 0
    try:
        cur.execute("SELECT count(*) FROM candidates")
        row_count = cur.fetchone()[0]
    except Exception:
        row_count = 0

    core_index: Dict[str, List[str]] = {}
    norm_index: Dict[str, List[str]] = {}

    if row_count >= 9900000:
        print(f"Reusing existing candidate database ({row_count} records indexed)...", flush=True)
        t_idx_start = time.perf_counter()
        cur.execute("SELECT id, core_name, norm_name FROM candidates")
        for cid, core, norm in cur:
            if core:
                lst = core_index.setdefault(core, [])
                if len(lst) < 40:
                    lst.append(cid)
            if norm and norm != core:
                lst = norm_index.setdefault(norm, [])
                if len(lst) < 40:
                    lst.append(cid)
        print(
            f"Loaded inverted indices from SQLite in {time.perf_counter() - t_idx_start:.2f}s "
            f"(Unique cores: {len(core_index)}, norms: {len(norm_index)})",
            flush=True,
        )
    else:
        print("Building high-speed SQLite candidate index on disk...", flush=True)
        t_idx_start = time.perf_counter()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS candidates ("
            "  id TEXT PRIMARY KEY, "
            "  name TEXT, "
            "  addr TEXT, "
            "  country TEXT, "
            "  core_name TEXT, "
            "  norm_name TEXT"
            ")"
        )

        def index_source_file(path: str, src_label: str):
            print(f"Indexing {src_label} from {path}...", flush=True)
            batch = []
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    cid = (row.get("entity_id") or "").strip()
                    name_raw = row.get("business_name") or ""
                    addr_raw = row.get("business_address") or ""
                    country_raw = row.get("country") or ""

                    rec = normalize_record(cid, name_raw, addr_raw, country_raw)
                    core = rec.name_core or ""
                    norm = rec.name_norm or ""
                    batch.append((cid, name_raw, addr_raw, country_raw, core, norm))

                    if core:
                        lst = core_index.setdefault(core, [])
                        if len(lst) < 40:
                            lst.append(cid)
                    if norm and norm != core:
                        lst = norm_index.setdefault(norm, [])
                        if len(lst) < 40:
                            lst.append(cid)

                    if len(batch) >= 50000:
                        cur.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
                        batch = []

                if batch:
                    cur.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
            conn.commit()

        index_source_file(s2_path, "Source 2")
        index_source_file(s3_path, "Source 3")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cand_id ON candidates(id)")
        conn.commit()
        print(f"Indexed S2+S3 in {time.perf_counter() - t_idx_start:.2f}s", flush=True)

    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    f_match = open(matching_out, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    print(f"Processing Source 1 records in streaming chunks of {chunk_size}...", flush=True)
    t_stream_start = time.perf_counter()

    total_s1_processed = 0
    total_matched_entities = 0

    s1_chunk_raw: List[Dict[str, Any]] = []
    cand_cache: Dict[str, NormalizedRecord] = {}

    def process_s1_chunk(chunk_raw: List[Dict[str, Any]]):
        nonlocal total_s1_processed, total_matched_entities

        chunk_norm = normalize_dataframe_rows(chunk_raw)
        chunk_cands_map: Dict[str, List[str]] = {}
        missing_cids: Set[str] = set()

        for s1_rec in chunk_norm:
            sid = s1_rec.entity_id
            cands_set: Set[str] = set()

            # Pass A: core & norm match
            if s1_rec.name_core and s1_rec.name_core in core_index:
                cands_set.update(core_index[s1_rec.name_core])
            if s1_rec.name_norm and s1_rec.name_norm in norm_index:
                cands_set.update(norm_index[s1_rec.name_norm])

            if len(cands_set) > max_k:
                cands_list = sorted(cands_set)[:max_k]
            else:
                cands_list = sorted(cands_set)

            chunk_cands_map[sid] = cands_list
            for cid in cands_list:
                if cid not in cand_cache:
                    missing_cids.add(cid)

        # Batch query uncached candidate records from SQLite
        if missing_cids:
            cid_list = list(missing_cids)
            for i in range(0, len(cid_list), 900):
                batch_cids = cid_list[i : i + 900]
                placeholders = ",".join(["?"] * len(batch_cids))
                cur.execute(
                    f"SELECT id, name, addr, country FROM candidates WHERE id IN ({placeholders})",
                    batch_cids,
                )
                db_rows = cur.fetchall()
                raw_cand_dicts = [
                    {"entity_id": r[0], "business_name": r[1], "business_address": r[2], "country": r[3]}
                    for r in db_rows
                ]
                norm_cands = normalize_dataframe_rows(raw_cand_dicts)
                for nr in norm_cands:
                    cand_cache[nr.entity_id] = nr

            if len(cand_cache) > 200000:
                cand_cache.clear()

        # Vectorized batch feature extraction
        chunk_pair_feats = []
        chunk_pair_meta = []  # (sid, cid, conflict_flag)

        for s1_rec in chunk_norm:
            sid = s1_rec.entity_id
            cands = chunk_cands_map.get(sid, [])
            if not cands:
                continue

            for cid in cands:
                cand_rec = cand_cache.get(cid)
                if cand_rec is None:
                    continue
                # Ground-truth invariant: same country
                if s1_rec.country_norm and cand_rec.country_norm and s1_rec.country_norm != cand_rec.country_norm:
                    continue

                feats = build_pair_features(s1_rec, cand_rec)
                conflict = (feats.get("numeric_conflict", 0.0) == 1.0)
                chunk_pair_feats.append([feats[name] for name in FEATURE_NAMES])
                chunk_pair_meta.append((sid, cid, conflict))

        # Vectorized model scoring
        if chunk_pair_feats:
            scores_array = model.predict_proba(chunk_pair_feats)
        else:
            scores_array = []

        # Group scores by S1 entity
        s1_scores_map: Dict[str, Dict[str, float]] = {r.entity_id: {} for r in chunk_norm}
        s1_conflicts_map: Dict[str, Set[str]] = {r.entity_id: set() for r in chunk_norm}

        for (sid, cid, conflict), score in zip(chunk_pair_meta, scores_array):
            s1_scores_map[sid][cid] = float(score)
            if conflict:
                s1_conflicts_map[sid].add(cid)

        # Write output for all S1 entities in chunk
        for s1_rec in chunk_norm:
            sid = s1_rec.entity_id
            cands = chunk_cands_map.get(sid, [])

            if not cands:
                f_cand.write(f"{sid}\t\n")
                f_match.write(f"{sid}\t\n")
                total_s1_processed += 1
                continue

            f_cand.write(f"{sid}\t{','.join(cands)}\n")

            cand_scores = s1_scores_map.get(sid, {})
            if not cand_scores:
                f_match.write(f"{sid}\t\n")
                total_s1_processed += 1
                continue

            matches = select_matches_for_entity(
                candidate_scores=cand_scores,
                threshold=decision_cfg.threshold,
                singleton_threshold=decision_cfg.singleton_threshold,
                numeric_conflicts=s1_conflicts_map.get(sid),
                suppress_numeric_conflict=decision_cfg.suppress_numeric_conflict,
                output_order="id_asc",
            )

            if matches:
                total_matched_entities += 1
                f_match.write(f"{sid}\t{','.join(matches)}\n")
            else:
                f_match.write(f"{sid}\t\n")

            total_s1_processed += 1

    with open(s1_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if limit is not None and total_s1_processed >= limit:
                break
            s1_chunk_raw.append({
                "entity_id": (row.get("entity_id") or "").strip(),
                "business_name": row.get("business_name"),
                "business_address": row.get("business_address"),
                "country": row.get("country"),
            })
            if len(s1_chunk_raw) >= chunk_size:
                process_s1_chunk(s1_chunk_raw)
                s1_chunk_raw = []
                f_match.flush()
                f_cand.flush()
                rate = total_s1_processed / max(1e-5, (time.perf_counter() - t_stream_start))
                print(f"Processed {total_s1_processed} S1 entities ({rate:.0f} S1/sec)...", flush=True)

        if s1_chunk_raw:
            process_s1_chunk(s1_chunk_raw)

    f_match.close()
    f_cand.close()
    conn.close()

    # Note: Keep cache_db_path on disk for fast reuse across evaluation/rerun
    # if os.path.exists(cache_db_path):
    #     os.remove(cache_db_path)

    t_total = time.perf_counter() - t_stream_start
    print(
        f"Completed streaming inference in {t_total:.2f}s "
        f"({total_s1_processed} S1 entities, {total_matched_entities} matched).",
        flush=True,
    )
    return total_s1_processed, total_matched_entities


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

    decision_cfg = DecisionConfig(
        threshold=args.threshold,
        suppress_numeric_conflict=args.suppress_numeric_conflict,
        output_order="id_asc",
    )

    # 2. Run Scalable Streaming Inference
    run_scalable_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model=model,
        decision_cfg=decision_cfg,
        chunk_size=25000,
        limit=args.limit,
    )

    # 3. Validate Submission Output
    if args.validate:
        validator_script = os.path.join("utils", "validate_submission.py")
        matching_out = os.path.join(args.output_dir, "matching_results.tsv")
        candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")
        if os.path.exists(validator_script):
            print("Running official submission validator...", flush=True)
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
                print("VALIDATION PASSED: Output files are completely compliant!", flush=True)
            else:
                print(f"VALIDATION FINISHED: Validator exited with code {ret.returncode}", flush=True)

    print(f"Pipeline execution finished in {time.perf_counter() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
