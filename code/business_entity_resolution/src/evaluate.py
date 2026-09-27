"""
evaluate.py
===========
Official Entity-Level Macro F0.5 Evaluator for Amazon ML Challenge 2026.

This module implements the exact evaluation logic described in the official
problem statement and SRS:
    - Entity-level Macro F0.5 metric across complete Source 1 entities.
    - F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - Singletons: If true match set is empty:
        - Predicted empty -> F0.5 = 1.0
        - Predicted non-empty -> F0.5 = 0.0
    - If true match set is non-empty:
        - Predicted empty -> F0.5 = 0.0
    - Returns macro-averaged F0.5, Precision, Recall, singleton stats,
      and diagnostic breakdowns.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Set, List, Optional, Any
import csv
import json
import os
import sys


@dataclass
class EntityScore:
    s1_id: str
    true_count: int
    pred_count: int
    tp_count: int
    fp_count: int
    fn_count: int
    precision: float
    recall: float
    f05: float
    is_singleton: bool
    is_correct_singleton: bool


@dataclass
class EvaluationReport:
    total_s1: int
    macro_f05: float
    macro_precision: float
    macro_recall: float
    
    # Singleton breakdown
    total_singletons: int
    correct_singletons: int
    singleton_accuracy: float
    
    # Non-singleton breakdown
    total_non_singletons: int
    non_singleton_macro_f05: float
    non_singleton_macro_precision: float
    non_singleton_macro_recall: float
    
    # Total link statistics
    total_true_links: int
    total_pred_links: int
    total_tp_links: int
    micro_precision: float
    micro_recall: float
    micro_f05: float

    per_entity_scores: Optional[List[EntityScore]] = None

    def to_dict(self, include_per_entity: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_per_entity:
            data.pop("per_entity_scores", None)
        return data


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> tuple:
    """Compute entity-level precision, recall, and F0.5 for a single S1 entity.

    Returns:
        (precision, recall, f05, tp, fp, fn)
    """
    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # Singleton case: true set is empty
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0
        else:
            return 0.0, 0.0, 0.0, 0, n_pred, 0

    # Non-singleton but predicted empty
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_true

    tp = len(true_ids & pred_ids)
    fp = n_pred - tp
    fn = n_true - tp

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0.0 or (precision == 0.0 and recall == 0.0):
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return precision, recall, f05, tp, fp, fn


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    include_per_entity: bool = False,
) -> EvaluationReport:
    """Evaluate predicted match sets against ground truth match sets.

    Both ground_truth and predictions map:
        source1_entity_id -> set of matching Source 2/3 entity IDs.

    All S1 entities in ground_truth are evaluated. If an S1 entity is missing
    from predictions, it is treated as predicting an empty set (singleton).
    """
    total_s1 = len(ground_truth)
    if total_s1 == 0:
        raise ValueError("Ground truth dictionary is empty.")

    per_entity: List[EntityScore] = []

    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0

    sum_ns_f05 = 0.0
    sum_ns_prec = 0.0
    sum_ns_rec = 0.0

    total_singletons = 0
    correct_singletons = 0

    tot_true = 0
    tot_pred = 0
    tot_tp = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        prec, rec, f05, tp, fp, fn = compute_entity_f05(true_set, pred_set)

        is_singleton = (len(true_set) == 0)
        is_correct_singleton = is_singleton and (len(pred_set) == 0)

        if is_singleton:
            total_singletons += 1
            if is_correct_singleton:
                correct_singletons += 1
        else:
            sum_ns_f05 += f05
            sum_ns_prec += prec
            sum_ns_rec += rec

        sum_f05 += f05
        sum_prec += prec
        sum_rec += rec

        tot_true += len(true_set)
        tot_pred += len(pred_set)
        tot_tp += tp

        if include_per_entity:
            per_entity.append(
                EntityScore(
                    s1_id=s1_id,
                    true_count=len(true_set),
                    pred_count=len(pred_set),
                    tp_count=tp,
                    fp_count=fp,
                    fn_count=fn,
                    precision=round(prec, 4),
                    recall=round(rec, 4),
                    f05=round(f05, 4),
                    is_singleton=is_singleton,
                    is_correct_singleton=is_correct_singleton,
                )
            )

    macro_f05 = sum_f05 / total_s1
    macro_prec = sum_prec / total_s1
    macro_rec = sum_rec / total_s1

    singleton_acc = (
        (correct_singletons / total_singletons) if total_singletons > 0 else 1.0
    )

    total_non_singletons = total_s1 - total_singletons
    if total_non_singletons > 0:
        ns_macro_f05 = sum_ns_f05 / total_non_singletons
        ns_macro_prec = sum_ns_prec / total_non_singletons
        ns_macro_rec = sum_ns_rec / total_non_singletons
    else:
        ns_macro_f05 = ns_macro_prec = ns_macro_rec = 0.0

    micro_prec = (tot_tp / tot_pred) if tot_pred > 0 else 0.0
    micro_rec = (tot_tp / tot_true) if tot_true > 0 else 0.0
    micro_denom = 0.25 * micro_prec + micro_rec
    micro_f05 = (
        (1.25 * micro_prec * micro_rec) / micro_denom if micro_denom > 0 else 0.0
    )

    return EvaluationReport(
        total_s1=total_s1,
        macro_f05=round(macro_f05, 6),
        macro_precision=round(macro_prec, 6),
        macro_recall=round(macro_rec, 6),
        total_singletons=total_singletons,
        correct_singletons=correct_singletons,
        singleton_accuracy=round(singleton_acc, 6),
        total_non_singletons=total_non_singletons,
        non_singleton_macro_f05=round(ns_macro_f05, 6),
        non_singleton_macro_precision=round(ns_macro_prec, 6),
        non_singleton_macro_recall=round(ns_macro_rec, 6),
        total_true_links=tot_true,
        total_pred_links=tot_pred,
        total_tp_links=tot_tp,
        micro_precision=round(micro_prec, 6),
        micro_recall=round(micro_rec, 6),
        micro_f05=round(micro_f05, 6),
        per_entity_scores=per_entity if include_per_entity else None,
    )


def load_tsv_mapping(path: str, col1: str = "source1_entity_id", col2: str = "matched_entity_ids") -> Dict[str, Set[str]]:
    """Load a tab-separated file with S1 ID and comma-separated target IDs."""
    mapping = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return mapping

        # Find column indices
        cols = [c.strip().lower() for c in header]
        idx1 = cols.index(col1.lower()) if col1.lower() in cols else 0
        idx2 = cols.index(col2.lower()) if col2.lower() in cols else 1

        for row in reader:
            if not row or len(row) <= idx1:
                continue
            s1_id = row[idx1].strip()
            if not s1_id:
                continue
            if len(row) > idx2 and row[idx2].strip():
                matches = {m.strip() for m in row[idx2].split(",") if m.strip()}
            else:
                matches = set()
            mapping[s1_id] = matches
    return mapping


def evaluate_tsv_files(
    ground_truth_path: str,
    predictions_path: str,
    include_per_entity: bool = False,
) -> EvaluationReport:
    """Evaluate a predictions TSV against a ground-truth TSV directly."""
    gt = load_tsv_mapping(ground_truth_path, "source1_entity_id", "matched_entity_ids")
    pred = load_tsv_mapping(predictions_path, "source1_entity_id", "matched_entity_ids")
    return evaluate_predictions(gt, pred, include_per_entity=include_per_entity)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Official Entity-Level Macro F0.5 Evaluator for Amazon ML Challenge 2026."
    )
    parser.add_argument("--truth", "-t", required=True, help="Path to ground truth TSV")
    parser.add_argument("--pred", "-p", required=True, help="Path to predictions TSV")
    parser.add_argument("--output", "-o", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--per-entity", action="store_true", help="Include per-entity diagnostics in output"
    )

    args = parser.parse_args()

    report = evaluate_tsv_files(
        args.truth, args.pred, include_per_entity=args.per_entity
    )

    print("\n" + "=" * 50)
    print("EVALUATION REPORT: Entity-Level Macro F0.5")
    print("=" * 50)
    print(f"Total S1 Entities Evaluated: {report.total_s1:,}")
    print(f"Macro F0.5 Score:            {report.macro_f05:.4f}")
    print(f"Macro Precision:             {report.macro_precision:.4f}")
    print(f"Macro Recall:                {report.macro_recall:.4f}")
    print("-" * 50)
    print(f"Singletons:                  {report.total_singletons:,} (Acc: {report.singleton_accuracy:.1%})")
    print(f"Non-Singletons:              {report.total_non_singletons:,} (Macro F0.5: {report.non_singleton_macro_f05:.4f})")
    print(f"Total True Links:            {report.total_true_links:,}")
    print(f"Total Predicted Links:       {report.total_pred_links:,}")
    print(f"True Positives:              {report.total_tp_links:,}")
    print("=" * 50)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(include_per_entity=args.per_entity), f, indent=2)
        print(f"Saved full evaluation report to {args.output}")
