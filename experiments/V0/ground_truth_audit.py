"""
ground_truth_audit.py
======================
Fast single-pass ground-truth audit.
Implements Audits A-D from the execution guide:
  A. Cross-country matches (separating missing/unknown country)
  B. Numeric-overlap categories
  C. Blank-address behavior
  D. Match-count distribution

Usage:
    python3 ground_truth_audit.py --train-dir dataset/train \
        --out reports/full_ground_truth_audit.md
"""

import argparse
import os
import re
import sys
from collections import Counter

NUMERIC_RE = re.compile(r"\d+[\w\-/]*\d*|\d+")


def is_blank_val(val):
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan" or s.lower() == "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--out", default="reports/full_ground_truth_audit.md")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("This script requires pandas: pip install pandas", file=sys.stderr)
        sys.exit(1)

    print("Loading datasets...", flush=True)
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    print("Building dictionaries...", flush=True)
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    s2_country = dict(zip(s2["entity_id"], s2["country"]))
    s3_country = dict(zip(s3["entity_id"], s3["country"]))

    s1_addr = dict(zip(s1["entity_id"], s1["business_address"]))
    s2_addr = dict(zip(s2["entity_id"], s2["business_address"]))
    s3_addr = dict(zip(s3["entity_id"], s3["business_address"]))

    s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
    s2_name = dict(zip(s2["entity_id"], s2["business_name"]))
    s3_name = dict(zip(s3["entity_id"], s3["business_name"]))

    print("Pre-extracting numeric tokens...", flush=True)
    addr_num_map = {}
    for df in [s1, s2, s3]:
        for eid, addr in zip(df["entity_id"], df["business_address"]):
            if not is_blank_val(addr):
                addr_num_map[eid] = set(NUMERIC_RE.findall(str(addr).lower()))

    # Counters
    same_country = 0
    cross_country = 0
    missing_country = 0
    cross_examples = []
    missing_examples = []

    b_categories = Counter()
    b_examples = {"agreement": [], "partial": [], "none": [], "conflict": []}

    blank_positive_count = 0
    total_positive_count = 0
    c_examples = []

    d_counts = Counter()

    print("Processing ground truth rows...", flush=True)
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if is_blank_val(matched):
            d_counts[0] += 1
            continue

        m_list = [m.strip() for m in str(matched).split(",") if m.strip()]
        d_counts[len(m_list)] += 1

        c1 = s1_country.get(s1_id)
        c1_blank = is_blank_val(c1)
        a1_raw = s1_addr.get(s1_id)
        a1_blank = is_blank_val(a1_raw)
        n1_num = addr_num_map.get(s1_id, set())

        for mid in m_list:
            total_positive_count += 1
            if mid.startswith("S2-"):
                c2 = s2_country.get(mid)
                a2_raw = s2_addr.get(mid)
                name2 = s2_name.get(mid)
            elif mid.startswith("S3-"):
                c2 = s3_country.get(mid)
                a2_raw = s3_addr.get(mid)
                name2 = s3_name.get(mid)
            else:
                continue

            c2_blank = is_blank_val(c2)
            a2_blank = is_blank_val(a2_raw)

            # Audit A
            if c1_blank or c2_blank:
                missing_country += 1
                if len(missing_examples) < 10:
                    missing_examples.append((s1_id, c1, mid, c2))
            elif str(c1).strip().lower() == str(c2).strip().lower():
                same_country += 1
            else:
                cross_country += 1
                if len(cross_examples) < 25:
                    cross_examples.append((s1_id, c1, mid, c2))

            # Audit B
            n2_num = addr_num_map.get(mid, set())
            if not n1_num or not n2_num:
                cat = "none"
            elif n1_num == n2_num:
                cat = "agreement"
            elif n1_num & n2_num:
                cat = "partial"
            else:
                cat = "conflict"

            b_categories[cat] += 1
            if len(b_examples[cat]) < 10:
                b_examples[cat].append((s1_id, mid, list(n1_num), list(n2_num)))

            # Audit C
            if a1_blank or a2_blank:
                blank_positive_count += 1
                if len(c_examples) < 15:
                    c_examples.append((s1_id, s1_name.get(s1_id), mid, name2))

    sections = [
        "# Full Ground-Truth Audit",
        "",
        f"- S1 rows: {len(s1):,}, S2 rows: {len(s2):,}, S3 rows: {len(s3):,}, "
        f"ground-truth S1 rows: {len(gt):,}",
        "",
        "## Audit A — Cross-country matches",
        "",
    ]

    total_a = same_country + cross_country + missing_country
    if total_a:
        sections.append(f"- same-country positive links: {same_country:,} ({same_country/total_a:.2%})")
        sections.append(f"- cross-country positive links: {cross_country:,} ({cross_country/total_a:.2%})")
        sections.append(f"- missing/unknown country positive links: {missing_country:,} ({missing_country/total_a:.2%})")
    else:
        sections.append("- no positive links found")
    sections.extend([
        "",
        "Cross-country examples (S1 id, S1 country, matched id, matched country):",
    ])
    for ex in cross_examples:
        sections.append(f"  - {ex}")
    sections.extend([
        "",
        "Missing/unknown country examples (S1 id, S1 country, matched id, matched country):",
    ])
    for ex in missing_examples:
        sections.append(f"  - {ex}")
    sections.extend([
        "",
        "**Action**: missing/unknown country is tracked separately from cross-country matches. "
        "If cross-country positives exist above zero, do NOT let blocking/normalization filter on country equality.",
        "",
        "## Audit B — Numeric-overlap categories",
        "",
    ])

    total_b = sum(b_categories.values())
    for cat in ["agreement", "partial", "none", "conflict"]:
        cnt = b_categories[cat]
        pct = f"{cnt/total_b:.2%}" if total_b else "n/a"
        sections.append(f"- {cat}: {cnt:,} ({pct})")
    sections.append("")
    for cat in ["agreement", "partial", "none", "conflict"]:
        sections.append(f"Examples — {cat}:")
        for ex in b_examples[cat]:
            sections.append(f"  - {ex}")
    sections.extend([
        "",
        "**Action**: hand these category counts to Parimarjan for numeric-feature "
        "design. Do not turn a sample observation into an unconditional hard rule.",
        "",
        "## Audit C — Blank-address behavior among TRUE matches",
        "",
    ])

    if total_positive_count:
        sections.append(
            f"- positive links where S1 or matched address is blank: "
            f"{blank_positive_count:,} / {total_positive_count:,} "
            f"({blank_positive_count/total_positive_count:.2%})"
        )
    sections.extend([
        "",
        "Examples (s1_id, s1_name, matched_id, matched_name):",
    ])
    for ex in c_examples:
        sections.append(f"  - {ex}")
    sections.extend([
        "",
        "**Action**: make sure missingness is represented explicitly so a blank "
        "address does not create a false similarity or a false non-match.",
        "",
        "## Audit D — Match-count distribution",
        "",
    ])

    total_d = sum(d_counts.values())
    for k in sorted(d_counts.keys()):
        sections.append(f"- {k} match(es): {d_counts[k]:,} S1 entities ({d_counts[k]/total_d:.2%})")
    sections.extend([
        "",
        "**Action**: confirm the decision layer supports zero, one, or many matches "
        "per S1 entity — do not assume a fixed match count.",
        "",
    ])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
