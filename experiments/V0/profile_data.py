"""
profile_data.py
================
Step 1 of Nidhi's plan: understand the data contract before coding.

Usage:
    python3 profile_data.py --train-dir dataset/train --test-dir dataset/test \
        --out reports/data_profile.md

Loads S1/S2/S3 (+ ground truth if present) WITHOUT altering raw text, and
reports: row counts, columns, dtypes, blank/whitespace-only rates per
column, and country value frequencies (no filtering/hard-coding).

Requires pandas (pip install pandas).
"""

import argparse
import os
import sys


def _blank_rate(series):
    return float((series.isna() | (series.astype(str).str.strip() == "")).mean())


def profile_file(path, name, pd):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    lines = [f"### {name}  (`{path}`)", "", f"- rows: {len(df):,}", f"- columns: {list(df.columns)}", ""]
    for col in df.columns:
        blank = _blank_rate(df[col])
        lines.append(f"  - `{col}`: blank/whitespace rate = {blank:.4%}")
    if "country" in df.columns:
        lines.append("")
        lines.append("  Country value frequencies (no filtering, open-set):")
        vc = df["country"].value_counts(dropna=False)
        for val, cnt in vc.items():
            lines.append(f"    - {val!r}: {cnt:,} ({cnt/len(df):.2%})")
    lines.append("")
    return "\n".join(lines)


def profile_ground_truth(path, pd):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    lines = [f"### train_ground_truth.tsv (`{path}`)", "", f"- rows (S1 entities): {len(df):,}", ""]
    total_links = 0
    singleton_count = 0
    for _, row in df.iterrows():
        val = row.get("matched_entity_ids", "")
        if val is None or str(val).strip() == "":
            singleton_count += 1
            continue
        total_links += len([x for x in str(val).split(",") if x.strip()])
    lines.append(f"- total positive links: {total_links:,}")
    lines.append(f"- singleton S1 entities (zero matches): {singleton_count:,}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out", default="reports/data_profile.md")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("This script requires pandas: pip install pandas", file=sys.stderr)
        sys.exit(1)

    sections = ["# Data Profile Report", ""]

    train_files = {
        "train_source1": "train_source1.tsv",
        "train_source2": "train_source2.tsv",
        "train_source3": "train_source3.tsv",
    }
    for label, fname in train_files.items():
        p = os.path.join(args.train_dir, fname)
        if os.path.exists(p):
            sections.append(profile_file(p, label, pd))

    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        sections.append(profile_ground_truth(gt_path, pd))

    test_files = {
        "test_source1": "test_source1.tsv",
        "test_source2": "test_source2.tsv",
        "test_source3": "test_source3.tsv",
    }
    for label, fname in test_files.items():
        p = os.path.join(args.test_dir, fname)
        if os.path.exists(p):
            sections.append(profile_file(p, label, pd))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
