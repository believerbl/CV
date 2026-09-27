"""
candidate_benchmark.py
======================
Piyush-role deliverable — benchmark the CURRENT frozen blocker
(Pass A+B+C+D+E in blocking.py) on the TRAINING population.

Compares cumulative configurations:
    A | A+B | A+B+C | A+B+C+D | A+B+C+D+E

Methodology (per Team SRS):
  - Ground truth is used ONLY AFTER candidate generation, to measure
    recall. It never influences blocking.
  - Calls the real generate_candidates() — no duplicated blocking logic.
  - Normalizes once up front (timed separately); blocker timing excludes
    repeated normalization.
  - No global K exists in blocking.py (passes use pass-specific caps);
    the K sweep is a benchmark-only post-union truncation, clearly
    labeled as such.

Usage:
    python candidate_benchmark.py [--train-dir DIR] [--s1 P --s2 P --s3 P --gt P]
                                  [--out-dir DIR] [--require-data]

    If the training TSVs are absent, a clearly labeled deterministic
    SYNTHETIC sample is used instead. The outputs always state whether
    the numbers come from the full training population or the sample.

    Run from this directory with normalize.py importable, e.g.:
        PYTHONPATH="<path to files aiml challenge>" python candidate_benchmark.py

Outputs (in --out-dir, default: this script's directory):
    candidate_benchmark_results.json
    candidate_benchmark_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Import bootstrap (benchmark-only convenience; does not modify blocking.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from blocking import generate_candidates
    from normalize import normalize_dataframe_rows
except ImportError:  # normalize.py lives in the sibling deliverable folder
    _SIBLING = os.path.join(
        os.path.dirname(_HERE), "files aiml challenge"
    )
    for _cand in (_SIBLING, os.path.join(_HERE, "..", "files aiml challenge")):
        _cand = os.path.normpath(_cand)
        if os.path.isdir(_cand) and _cand not in sys.path:
            sys.path.insert(0, _cand)
    from blocking import generate_candidates  # noqa: E402
    from normalize import normalize_dataframe_rows  # noqa: E402

K_SWEEP_DEFAULT = [5, 10, 15, 20, 30, 50, 100]

# Cumulative pass configurations. Pass A has no enable flag in
# blocking.py (it is the always-on base); the rest use existing flags.
CONFIGS = [
    ("A", {"pass_b_enabled": False, "pass_c_enabled": False,
           "pass_d_enabled": False, "pass_e_enabled": False}),
    ("A+B", {"pass_b_enabled": True, "pass_c_enabled": False,
             "pass_d_enabled": False, "pass_e_enabled": False}),
    ("A+B+C", {"pass_b_enabled": True, "pass_c_enabled": True,
               "pass_d_enabled": False, "pass_e_enabled": False}),
    ("A+B+C+D", {"pass_b_enabled": True, "pass_c_enabled": True,
                 "pass_d_enabled": True, "pass_e_enabled": False}),
    ("A+B+C+D+E", {"pass_b_enabled": True, "pass_c_enabled": True,
                   "pass_d_enabled": True, "pass_e_enabled": True}),
]


# ---------------------------------------------------------------------------
# Loading (TSV schema inspected from profile_data.py / ground_truth_audit.py
# / normalize.py: source files carry entity_id, business_name,
# business_address, country; ground truth carries source1_entity_id,
# matched_entity_ids as comma-separated S2/S3 ids, blank = no match).
# ---------------------------------------------------------------------------
def load_source_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        for required in ("entity_id",):
            if required not in cols:
                raise ValueError(
                    "File %s is missing required column %r (found %r)"
                    % (path, required, cols)
                )
        rows = []
        for r in reader:
            rows.append({
                "entity_id": (r.get("entity_id") or "").strip(),
                "business_name": r.get("business_name"),
                "business_address": r.get("business_address"),
                "country": r.get("country"),
            })
    return rows, cols


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        for required in ("source1_entity_id", "matched_entity_ids"):
            if required not in cols:
                raise ValueError(
                    "Ground-truth file %s missing %r (found %r)"
                    % (path, required, cols)
                )
        gt = {}
        for r in reader:
            s1 = (r.get("source1_entity_id") or "").strip()
            raw = r.get("matched_entity_ids")
            ids = []
            if raw is not None and str(raw).strip() != "":
                ids = [m.strip() for m in str(raw).split(",") if m.strip()]
            gt[s1] = ids
    return gt, cols


def build_synthetic_sample():
    """Small benchmark-only TEST-FIXTURE derived from existing tests.

    Dataset type: TEST-FIXTURE / NORMALIZATION-DATASET.
    This benchmark is NOT representative of full training-data scale.
    Do not claim training recall from these numbers.

    Every record mirrors an expectation already encoded in the existing
    test suites (same row() schema, same normalization behavior):
      S1-001 Acme Corp <-> S2-001 Acme LLC + S3-001 Acme Incorporated
        (test_blocking_pass_a: core match after legal-suffix removal,
        one-to-many S2+S3 coverage)
      S1-002 Zxqwv Traders <-> S2-002 Zxqwv Foods
        (test_blocking_pass_b: shared rare token)
      S1-003 Blue Ocean Traders <-> S2-003 Red Mountain Foods via 62704,
        France vs India
        (test_blocking_pass_c: postal-token match + cross-country)
      S1-004 Continental Traders <-> S2-004 Continantal Traders
        (test_blocking_pass_b typo-token scope + Pass D typo tolerance)
      S1-005 Devanagari pair, India vs France
        (test_blocking_pass_e: same-script rare-token + cross-country)
      S1-006 blank name/address (blank-safety tests in every suite)
      S1-007 unmatched name/address (every-S1-key tests)
      S1-008 Multi Match Mart <-> S2-007 + S3-003 via 221B
        (test_blocking_pass_c: numeric-token match, one-to-many)
    """
    s1 = [
        {"entity_id": "S1-001", "business_name": "Acme Corp",
         "business_address": "1 Main St, Springfield, 62704", "country": "India"},
        {"entity_id": "S1-002", "business_name": "Zxqwv Traders",
         "business_address": "2 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-003", "business_name": "Blue Ocean Traders",
         "business_address": "500 Market St, Springfield, 62704", "country": "France"},
        {"entity_id": "S1-004", "business_name": "Continental Traders",
         "business_address": "3 Main St, Springfield", "country": "India"},
        {"entity_id": "S1-005", "business_name": "शर्मा टेक्सटाइल्स",
         "business_address": "4 Main St, Delhi, 110001", "country": "India"},
        {"entity_id": "S1-006", "business_name": None,
         "business_address": None, "country": "India"},
        {"entity_id": "S1-007", "business_name": "Completely Unique Name Here",
         "business_address": "99 Nowhere Lane, Villagetown", "country": "India"},
        {"entity_id": "S1-008", "business_name": "Multi Match Mart",
         "business_address": "221B Baker Street, London", "country": "India"},
    ]
    s2 = [
        {"entity_id": "S2-001", "business_name": "Acme LLC",
         "business_address": "9 Other Rd, Shelbyville", "country": "India"},
        {"entity_id": "S2-002", "business_name": "Zxqwv Foods",
         "business_address": "8 Other Rd, Shelbyville", "country": "India"},
        {"entity_id": "S2-003", "business_name": "Red Mountain Foods",
         "business_address": "10 Other Rd, Shelbyville, 62704", "country": "India"},
        {"entity_id": "S2-004", "business_name": "Continantal Traders",
         "business_address": "7 Other Rd, Shelbyville", "country": "India"},
        {"entity_id": "S2-005", "business_name": "शर्मा ट्रेडर्स",
         "business_address": "5 Other Rd, Delhi, 110001", "country": "France"},
        {"entity_id": "S2-006", "business_name": "Unrelated Business One",
         "business_address": "6 Other Rd, Shelbyville", "country": "India"},
        {"entity_id": "S2-007", "business_name": "Multi Match Mart Inc",
         "business_address": "Flat 221B, Baker Street, London", "country": "India"},
    ]
    s3 = [
        {"entity_id": "S3-001", "business_name": "Acme Incorporated",
         "business_address": "11 Third St, Capital City", "country": "India"},
        {"entity_id": "S3-002", "business_name": "Unrelated Business Two",
         "business_address": "12 Third St, Capital City", "country": "India"},
        {"entity_id": "S3-003", "business_name": "Multi Match Mart LLC",
         "business_address": "221B Baker Street, London", "country": "France"},
    ]
    gt = {
        "S1-001": ["S2-001", "S3-001"],   # one-to-many via Pass A core
        "S1-002": ["S2-002"],             # rare token (B) + script (E)
        "S1-003": ["S2-003"],             # postal anchor (C), cross-country
        "S1-004": ["S2-004"],             # typo name (D)
        "S1-005": ["S2-005"],             # Devanagari same-script (E)
        "S1-006": [],                     # blank record, no positive
        "S1-007": [],                     # unmatched record, no positive
        "S1-008": ["S2-007", "S3-003"],   # one-to-many via numeric anchor
    }
    return s1, s2, s3, gt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    import math
    k = math.ceil(pct / 100.0 * len(sorted_vals)) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


def cardinality_stats(candidate_map):
    counts = sorted(len(v) for v in candidate_map.values())
    total = sum(counts)
    return {
        "mean": (total / len(counts)) if counts else 0.0,
        "median": float(statistics.median(counts)) if counts else 0.0,
        "p95": percentile(counts, 95),
        "max": max(counts) if counts else 0,
        "total_pairs": total,
    }


def safety_check(candidate_map, s1_ids, valid_ids):
    """Fail loudly on any structural violation. Returns list of problems."""
    problems = []
    missing = [sid for sid in s1_ids if sid not in candidate_map]
    if missing:
        problems.append("missing S1 keys: %r" % (missing[:10],))
    for sid, cands in candidate_map.items():
        if len(cands) != len(set(cands)):
            problems.append("duplicate candidate IDs for %s" % sid)
        if cands != sorted(cands):
            problems.append("non-deterministic order for %s" % sid)
        for cid in cands:
            if cid == sid or cid.startswith("S1-"):
                problems.append("S1 ID %r appears as candidate for %s" % (cid, sid))
            if cid not in valid_ids:
                problems.append("unknown candidate ID %r for %s" % (cid, sid))
    return problems


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Benchmark the frozen A+B+C+D+E blocker.")
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--s1", default=None)
    ap.add_argument("--s2", default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--gt", "--ground-truth", dest="gt", default=None)
    ap.add_argument("--out-dir", default=_HERE)
    ap.add_argument("--require-data", action="store_true",
                    help="Fail if training TSVs are absent instead of using the sample.")
    args = ap.parse_args()

    bench_start = time.perf_counter()
    tracemalloc.start()

    # -- Locate data ---------------------------------------------------------
    s1_path = args.s1
    s2_path = args.s2
    s3_path = args.s3
    gt_path = args.gt
    if args.train_dir:
        s1_path = s1_path or os.path.join(args.train_dir, "train_source1.tsv")
        s2_path = s2_path or os.path.join(args.train_dir, "train_source2.tsv")
        s3_path = s3_path or os.path.join(args.train_dir, "train_source3.tsv")
        gt_path = gt_path or os.path.join(args.train_dir, "train_ground_truth.tsv")

    is_sample = False
    data_source = {}
    if s1_path and s2_path and s3_path and gt_path and all(
            os.path.exists(p) for p in (s1_path, s2_path, s3_path, gt_path)):
        s1_rows, s1_cols = load_source_tsv(s1_path)
        s2_rows, s2_cols = load_source_tsv(s2_path)
        s3_rows, s3_cols = load_source_tsv(s3_path)
        gt_raw, gt_cols = load_ground_truth(gt_path)
        data_source = {"mode": "full-training",
                       "s1": s1_path, "s2": s2_path, "s3": s3_path, "gt": gt_path,
                       "s1_columns": s1_cols, "gt_columns": gt_cols}
    else:
        if args.require_data:
            raise SystemExit(
                "Training TSVs not found (s1=%r s2=%r s3=%r gt=%r) and "
                "--require-data was given. Aborting." % (s1_path, s2_path, s3_path, gt_path))
        s1_rows, s2_rows, s3_rows, gt_raw = build_synthetic_sample()
        is_sample = True
        data_source = {"mode": "TEST-FIXTURE / NORMALIZATION-DATASET",
                       "note": "Small benchmark-only fixture derived from the "
                               "existing normalization/blocking test "
                               "expectations. This benchmark is NOT "
                               "representative of full training-data scale. "
                               "Do not claim training recall."}

    s1_ids = [r["entity_id"] for r in s1_rows]
    valid_ids = set(r["entity_id"] for r in (s2_rows + s3_rows))
    # Ground truth covers the COMPLETE S1 population: default empty set.
    gt_map = {sid: list(gt_raw.get(sid, [])) for sid in s1_ids}
    total_tp = sum(len(v) for v in gt_map.values())
    s1_with_tp = sum(1 for v in gt_map.values() if v)

    # Ground truth used ONLY for measurement: validate it maps correctly.
    unmapped = sorted({cid for v in gt_map.values() for cid in v} - valid_ids)
    if unmapped:
        raise SystemExit(
            "FAIL: %d ground-truth ID(s) do not exist in S2/S3 (e.g. %r). "
            "Refusing to silently drop malformed data." % (len(unmapped), unmapped[:10]))

    n1, n23 = len(s1_rows), len(s2_rows) + len(s3_rows)
    total_possible = n1 * n23

    # -- Normalize once (timed separately) ------------------------------------
    t0 = time.perf_counter()
    s1_norm = normalize_dataframe_rows(s1_rows)
    s2_norm = normalize_dataframe_rows(s2_rows)
    s3_norm = normalize_dataframe_rows(s3_rows)
    norm_time = time.perf_counter() - t0
    norm_by_id = {r.entity_id: r for r in (s1_norm + s2_norm + s3_norm)}

    # -- Cumulative configurations ---------------------------------------------
    results = []
    prev_pair_set = set()
    prev_recovered = set()
    prev_map = None
    for name, flags in CONFIGS:
        cfg = dict(flags)  # all other blocker defaults preserved
        t0 = time.perf_counter()
        cmap = generate_candidates(s1_norm, s2_norm, s3_norm, cfg)
        blocker_time = time.perf_counter() - t0

        problems = safety_check(cmap, s1_ids, valid_ids)
        if problems:
            raise SystemExit("FAIL [%s] safety violations: %s" % (name, problems[:5]))

        pair_set = set((sid, cid) for sid, cands in cmap.items() for cid in cands)
        recovered = set((sid, cid) for sid, tps in gt_map.items()
                        for cid in tps if cid in cmap.get(sid, []))
        retained = len(recovered)
        recall = (retained / total_tp) if total_tp else None
        zero_cand = sum(1 for sid in s1_ids if not cmap.get(sid))
        zero_ret = sorted(sid for sid, tps in gt_map.items()
                          if tps and not set(tps) & set(cmap.get(sid, [])))
        card = cardinality_stats(cmap)
        reduction = (1.0 - card["total_pairs"] / total_possible) if total_possible else None

        results.append({
            "name": name,
            "flags": cfg,
            "blocker_time_sec": blocker_time,
            "recall": recall,
            "retained_tp": retained,
            "total_tp": total_tp,
            "s1_with_positives": s1_with_tp,
            "s1_total": n1,
            "s1_completely_missed": len(zero_ret),
            "s1_zero_candidates": zero_cand,
            "zero_retention_s1_ids": zero_ret,
            "cardinality": card,
            "total_possible_pairs": total_possible,
            "reduction_ratio": reduction,
            "reduction_factor": (total_possible / card["total_pairs"])
            if card["total_pairs"] else None,
            "new_pairs_vs_prev": len(pair_set - prev_pair_set),
            "newly_recovered_tp_vs_prev": len(recovered - prev_recovered),
        })
        prev_pair_set, prev_recovered, prev_map = pair_set, recovered, cmap

    full_map = prev_map

    # -- Determinism: run final config twice on identical inputs ---------------
    t0 = time.perf_counter()
    run1 = generate_candidates(s1_norm, s2_norm, s3_norm, dict(CONFIGS[-1][1]))
    run2 = generate_candidates(s1_norm, s2_norm, s3_norm, dict(CONFIGS[-1][1]))
    determ_time = time.perf_counter() - t0
    determinism = {
        "identical": run1 == run2,
        "final_equals_benchmark_map": run1 == full_map,
        "double_run_time_sec": determ_time,
    }
    if not determinism["identical"]:
        raise SystemExit("FAIL: final A+B+C+D+E configuration is nondeterministic.")

    # -- K sweep (evaluating real blocking.py candidate pruning) ---------------
    k_sweep = []
    for k in K_SWEEP_DEFAULT:
        cfg_k = dict(CONFIGS[-1][1], max_candidates_per_s1=k)
        t_k0 = time.perf_counter()
        capped = generate_candidates(s1_norm, s2_norm, s3_norm, cfg_k)
        k_time = time.perf_counter() - t_k0
        rec = sum(1 for sid, tps in gt_map.items()
                  for cid in tps if cid in capped.get(sid, []))
        card = cardinality_stats(capped)
        k_sweep.append({
            "k": k,
            "recall": (rec / total_tp) if total_tp else None,
            "retained_tp": rec,
            "cardinality": card,
            "runtime_sec": round(k_time, 4),
        })

    # -- Zero-retention diagnostics for the final configuration -----------------
    zero_diag = []
    for sid in results[-1]["zero_retention_s1_ids"][:50]:
        rec = norm_by_id.get(sid)
        info = {"s1_id": sid,
                "true_positive_ids": gt_map.get(sid, []),
                "generated_candidates": full_map.get(sid, [])[:20]}
        if rec is not None:
            info["name_raw"] = rec.name_raw
            info["country_raw"] = rec.country_raw
            try:
                info["primary_script"] = (rec.script_flags or {}).get("name", {}).get(
                    "primary_script")
            except Exception:
                info["primary_script"] = None
        zero_diag.append(info)

    current_peak, _peak_max = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.perf_counter() - bench_start

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "is_sample": is_sample,
        "dataset_counts": {"s1": n1, "s2": len(s2_rows), "s3": len(s3_rows),
                           "s2_plus_s3": n23, "total_tp": total_tp,
                           "s1_with_positives": s1_with_tp},
        "total_possible_pairs": total_possible,
        "normalization_time_sec": norm_time,
        "configurations": results,
        "k_sweep": {
            "note": "Benchmark-only post-union truncation of the final "
                    "A+B+C+D+E map (first K sorted IDs per S1). blocking.py "
                    "has no global K; internal pass caps "
                    "(pass_b/c/e_max_candidates) are unchanged.",
            "ks": K_SWEEP_DEFAULT,
            "results": k_sweep,
        },
        "zero_retention_final": {
            "count": len(results[-1]["zero_retention_s1_ids"]),
            "diagnostics": zero_diag,
        },
        "determinism": determinism,
        "runtime": {"normalization_sec": norm_time,
                    "total_benchmark_sec": total_time},
        "memory": {
            "method": "tracemalloc.get_traced_memory (Python allocations "
                      "only; not process RSS peak). Portable but approximate.",
            "tracemalloc_current_peak_bytes": current_peak,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "candidate_benchmark_results.json")
    md_path = os.path.join(args.out_dir, "candidate_benchmark_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(payload))

    print("mode: %s" % data_source["mode"])
    print("dataset: S1=%d S2=%d S3=%d total_tp=%d" % (n1, len(s2_rows), len(s3_rows), total_tp))
    for r in results:
        c = r["cardinality"]
        print("%-9s recall=%s retained=%d/%d mean=%.2f p95=%s max=%d total=%d missed_s1=%d zero_cand=%d t=%.3fs"
              % (r["name"],
                 ("%.4f" % r["recall"]) if r["recall"] is not None else "n/a",
                 r["retained_tp"], r["total_tp"], c["mean"], c["p95"], c["max"],
                 c["total_pairs"], r["s1_completely_missed"],
                 r["s1_zero_candidates"], r["blocker_time_sec"]))
    print("determinism identical: %s" % determinism["identical"])
    print("wrote %s" % json_path)
    print("wrote %s" % md_path)


def render_markdown(p):
    L = []
    tag = ("TEST-FIXTURE / NORMALIZATION-DATASET"
           if p["is_sample"] else "FULL TRAINING")
    L.append("# Candidate Benchmark Report — %s" % tag)
    L.append("")
    L.append("Dataset type: TEST-FIXTURE / NORMALIZATION-DATASET"
             if p["is_sample"] else "Dataset type: FULL TRAINING")
    L.append("This benchmark is NOT representative of full training-data scale."
             if p["is_sample"] else "")
    L.append("")
    L.append("- Generated (UTC): %s" % p["generated_at_utc"])
    L.append("- Data source: %s" % json.dumps(p["data_source"], ensure_ascii=False))
    d = p["dataset_counts"]
    L.append("- Dataset: S1=%d, S2=%d, S3=%d, S2+S3=%d, total positives=%d, "
             "S1 with positives=%d" % (
                 d["s1"], d["s2"], d["s3"], d["s2_plus_s3"],
                 d["total_tp"], d["s1_with_positives"]))
    L.append("- Exhaustive pairs S1x(S2+S3): %d" % p["total_possible_pairs"])
    L.append("")
    L.append("## Benchmark methodology")
    L.append("")
    L.append("- Real `generate_candidates()` called per configuration; "
             "ground truth used only afterward for recall.")
    L.append("- Records normalized once (%.4fs); blocker timing excludes "
             "renormalization." % p["normalization_time_sec"])
    L.append("- Recall is measured over the COMPLETE S1 population "
             "(unmatched S1s included in cardinality, excluded from recall "
             "numerator/denominator).")
    L.append("- No global K exists in blocking.py (pass-specific caps only); "
             "the K sweep is a benchmark-only post-union truncation.")
    for section, r in (("A", 0), ("A+B", 1), ("A+B+C", 2), ("A+B+C+D", 3),
                       ("A+B+C+D+E", 4)):
        c = p["configurations"][r]
        card = c["cardinality"]
        L.append("")
        L.append("## %s" % ("A" if section == "A" else section))
        L.append("")
        L.append("- Flags: `%s`" % json.dumps(c["flags"], ensure_ascii=False))
        L.append("- Recall: %s (retained %d/%d)" % (
            ("%.4f" % c["recall"]) if c["recall"] is not None else "n/a",
            c["retained_tp"], c["total_tp"]))
        L.append("- S1 with positives: %d; completely missed: %d; "
                 "zero-candidate S1: %d" % (
                     c["s1_with_positives"], c["s1_completely_missed"],
                     c["s1_zero_candidates"]))
        L.append("- Cardinality: mean %.3f, median %s, P95 %s, max %d, "
                 "total %d" % (card["mean"], card["median"], card["p95"],
                               card["max"], card["total_pairs"]))
        L.append("- Reduction ratio: %s; factor: %s" % (
            ("%.6f" % c["reduction_ratio"]) if c["reduction_ratio"] is not None else "n/a",
            ("%.2f" % c["reduction_factor"]) if c["reduction_factor"] else "n/a"))
        L.append("- Blocker runtime: %.4fs" % c["blocker_time_sec"])
    L.append("")
    L.append("## Incremental pass contribution")
    L.append("")
    L.append("| stage | new pairs | newly recovered TP |")
    L.append("|---|---|---|")
    for c in p["configurations"]:
        L.append("| %s | %d | %d |" % (
            c["name"], c["new_pairs_vs_prev"], c["newly_recovered_tp_vs_prev"]))
    L.append("")
    L.append("## K sweep — BENCHMARK-ONLY K SWEEP (cap on final map)")
    L.append("")
    L.append("| K | recall | retained | mean | median | P95 | max | total |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in p["k_sweep"]["results"]:
        card = k["cardinality"]
        L.append("| %d | %s | %d | %.3f | %s | %s | %d | %d |" % (
            k["k"], ("%.4f" % k["recall"]) if k["recall"] is not None else "n/a",
            k["retained_tp"], card["mean"], card["median"], card["p95"],
            card["max"], card["total_pairs"]))
    L.append("")
    L.append("## Zero-retention analysis (final A+B+C+D+E)")
    L.append("")
    L.append("- S1 with positives but zero retained: %d"
             % p["zero_retention_final"]["count"])
    for z in p["zero_retention_final"]["diagnostics"]:
        L.append("- %s | true=%s | cand=%s | name=%r | country=%r | script=%r" % (
            z["s1_id"], z["true_positive_ids"], z["generated_candidates"],
            z.get("name_raw"), z.get("country_raw"), z.get("primary_script")))
    L.append("")
    L.append("## Candidate cardinality")
    L.append("")
    L.append("See per-configuration cardinality above; totals are bounded by "
             "pass-specific DF suppression plus caps (no global K in blocker).")
    L.append("")
    L.append("## Runtime/memory")
    L.append("")
    L.append("- Normalization: %.4fs; total benchmark: %.4fs"
             % (p["runtime"]["normalization_sec"], p["runtime"]["total_benchmark_sec"]))
    L.append("- Memory method: %s" % p["memory"]["method"])
    L.append("- tracemalloc current peak: %d bytes"
             % p["memory"]["tracemalloc_current_peak_bytes"])
    L.append("- Determinism double-run identical: %s"
             % p["determinism"]["identical"])
    L.append("")
    L.append("## Observations (measured only)")
    L.append("")
    if p["is_sample"]:
        L.append("- Fixture mode: numbers below describe the TEST-FIXTURE / "
                 "NORMALIZATION-DATASET only and must not be quoted as "
                 "training recall.")
    recs = [(c["name"], c["recall"], c["cardinality"]["total_pairs"]) for c in p["configurations"]]
    L.append("- Recall by configuration: %s"
             % ", ".join("%s=%s" % (n, ("%.4f" % r) if r is not None else "n/a")
                         for n, r, _ in recs))
    L.append("- Total pairs by configuration: %s"
             % ", ".join("%s=%d" % (n, t) for n, _, t in recs))
    L.append("")
    L.append("## Frozen recommendation data (for team review)")
    L.append("")
    L.append("No best-pass claim is made here. The JSON artifact carries the "
             "full numbers; the team decides the freeze from measured "
             "recall/cardinality/cost trade-offs.")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
