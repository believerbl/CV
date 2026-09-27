"""
make_sample.py
===============
Builds a SMALL, self-consistent sample of the real training data, so you
don't have to upload the full multi-million-row dataset just to get real
noise examples reviewed.

"Self-consistent" matters: a naive `head -n 3000` on each file
independently would almost certainly NOT share matched entity_ids across
S1/S2/S3, because the files aren't row-aligned. This script instead:
  1. Samples N real S1 entities from the ground truth (a mix of matched
     and singleton entities).
  2. Pulls in every S2/S3 record those entities actually match to.
  3. Writes a small, real TSV bundle that still has genuine positive pairs
     in it — so punctuation/typo/reorder/numeric/blank-address/script
     examples can be found for real, and singleton examples are included.

Usage:
    python3 make_sample.py --train-dir dataset/train --out-dir sample \
        --n-matched 1500 --n-singleton 500

Then zip the `sample/` folder and upload/send just that.
Requires pandas.
"""

import argparse
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--out-dir", default="sample")
    ap.add_argument("--n-matched", type=int, default=1500,
                     help="number of S1 entities WITH matches to sample")
    ap.add_argument("--n-singleton", type=int, default=500,
                     help="number of S1 entities WITHOUT matches to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("This script requires pandas: pip install pandas", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)

    gt = pd.read_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"),
                      sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    matched_rows = gt[gt["matched_entity_ids"].fillna("").str.strip() != ""]
    singleton_rows = gt[gt["matched_entity_ids"].fillna("").str.strip() == ""]

    n_matched = min(args.n_matched, len(matched_rows))
    n_singleton = min(args.n_singleton, len(singleton_rows))

    sampled_matched = matched_rows.sample(n=n_matched, random_state=args.seed)
    sampled_singleton = singleton_rows.sample(n=n_singleton, random_state=args.seed)
    sampled_gt = pd.concat([sampled_matched, sampled_singleton]).reset_index(drop=True)

    s1_ids = set(sampled_gt["source1_entity_id"])
    s2_ids, s3_ids = set(), set()
    for val in sampled_matched["matched_entity_ids"]:
        for mid in str(val).split(","):
            mid = mid.strip()
            if mid.startswith("S2-"):
                s2_ids.add(mid)
            elif mid.startswith("S3-"):
                s3_ids.add(mid)

    def load_and_filter(fname, id_set):
        path = os.path.join(args.train_dir, fname)
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
        return df[df["entity_id"].isin(id_set)]

    s1_df = load_and_filter("train_source1.tsv", s1_ids)
    s2_df = load_and_filter("train_source2.tsv", s2_ids)
    s3_df = load_and_filter("train_source3.tsv", s3_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    sampled_gt.to_csv(os.path.join(args.out_dir, "sample_ground_truth.tsv"), sep="\t", index=False)
    s1_df.to_csv(os.path.join(args.out_dir, "sample_source1.tsv"), sep="\t", index=False)
    s2_df.to_csv(os.path.join(args.out_dir, "sample_source2.tsv"), sep="\t", index=False)
    s3_df.to_csv(os.path.join(args.out_dir, "sample_source3.tsv"), sep="\t", index=False)

    print(f"Wrote sample to {args.out_dir}/")
    print(f"  S1 entities: {len(s1_df)} ({n_matched} matched + {n_singleton} singleton)")
    print(f"  S2 records:  {len(s2_df)}")
    print(f"  S3 records:  {len(s3_df)}")
    print(f"  ground truth rows: {len(sampled_gt)}")
    print("Now zip that folder and send/upload just that — it's small and self-contained.")


if __name__ == "__main__":
    main()