"""
test_evaluate.py
================
Unit tests for the official entity-level Macro F0.5 evaluator.
Covers the exact requirements from Parimarjan's execution guide and SRS:
    - Perfect entity score (1.0)
    - Reduced recall when some true matches missed
    - Reduced precision when false positives present (official PDF example: 0.714)
    - Singleton true empty + pred empty -> 1.0
    - Singleton true empty + pred non-empty -> 0.0
    - Macro average across entities, not pairs
    - Missing S1 in predictions defaults to empty set
"""

import unittest
import os
import tempfile
from evaluate import compute_entity_f05, evaluate_predictions, evaluate_tsv_files


class TestEntityF05Calculation(unittest.TestCase):
    def test_perfect_single_match(self):
        prec, rec, f05, tp, fp, fn = compute_entity_f05({"S2-001"}, {"S2-001"})
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f05, 1.0)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_perfect_multiple_matches(self):
        prec, rec, f05, tp, fp, fn = compute_entity_f05(
            {"S2-001", "S3-002", "S2-003"}, {"S2-001", "S3-002", "S2-003"}
        )
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f05, 1.0)
        self.assertEqual(tp, 3)

    def test_partial_recall_no_fp(self):
        # 2 true matches, only 1 predicted
        prec, rec, f05, tp, fp, fn = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001"})
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 0.5)
        # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333...
        expected_f05 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        self.assertAlmostEqual(f05, expected_f05, places=4)

    def test_official_pdf_example(self):
        # Official example from problem statement PDF (Page 6):
        # GT: [S2-00047, S3-00812]
        # Pred: [S2-00047, S2-00193, S3-00812]
        # Prec = 2/3 = 0.6667, Rec = 2/2 = 1.0
        # F0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
        prec, rec, f05, tp, fp, fn = compute_entity_f05(
            {"S2-00047", "S3-00812"},
            {"S2-00047", "S2-00193", "S3-00812"},
        )
        self.assertAlmostEqual(prec, 2 / 3, places=4)
        self.assertEqual(rec, 1.0)
        self.assertAlmostEqual(f05, 0.7143, places=3)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

    def test_singleton_correct_empty(self):
        # True is empty, pred is empty -> score 1.0
        prec, rec, f05, tp, fp, fn = compute_entity_f05(set(), set())
        self.assertEqual(f05, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    def test_singleton_false_merge(self):
        # True is empty, pred has false matches -> score 0.0
        prec, rec, f05, tp, fp, fn = compute_entity_f05(set(), {"S2-999"})
        self.assertEqual(f05, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

    def test_non_singleton_predicted_empty(self):
        # True has matches, pred is empty -> score 0.0
        prec, rec, f05, tp, fp, fn = compute_entity_f05({"S2-001"}, set())
        self.assertEqual(f05, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)


class TestMacroEvaluation(unittest.TestCase):
    def test_macro_average_across_entities(self):
        # S1-1: perfect match (1.0)
        # S1-2: correct singleton (1.0)
        # S1-3: singleton false merge (0.0)
        # S1-4: missed match (0.0)
        # Expected macro F0.5 = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.500
        truth = {
            "S1-1": {"S2-10"},
            "S1-2": set(),
            "S1-3": set(),
            "S1-4": {"S3-40"},
        }
        pred = {
            "S1-1": {"S2-10"},
            "S1-2": set(),
            "S1-3": {"S2-99"},
            "S1-4": set(),
        }

        report = evaluate_predictions(truth, pred)
        self.assertEqual(report.total_s1, 4)
        self.assertEqual(report.macro_f05, 0.5)
        self.assertEqual(report.total_singletons, 2)
        self.assertEqual(report.correct_singletons, 1)
        self.assertEqual(report.singleton_accuracy, 0.5)

    def test_missing_pred_treated_as_empty(self):
        # S1-2 is in truth as singleton, not in pred dict -> defaults to empty (correct!)
        truth = {"S1-1": {"S2-10"}, "S1-2": set()}
        pred = {"S1-1": {"S2-10"}}  # S1-2 omitted

        report = evaluate_predictions(truth, pred)
        self.assertEqual(report.macro_f05, 1.0)

    def test_tsv_evaluation(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f_gt:
            f_gt.write("source1_entity_id\tmatched_entity_ids\n")
            f_gt.write("S1-01\tS2-100,S3-200\n")
            f_gt.write("S1-02\t\n")
            gt_path = f_gt.name

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f_pred:
            f_pred.write("source1_entity_id\tmatched_entity_ids\n")
            f_pred.write("S1-01\tS2-100,S3-200\n")
            f_pred.write("S1-02\t\n")
            pred_path = f_pred.name

        try:
            report = evaluate_tsv_files(gt_path, pred_path)
            self.assertEqual(report.total_s1, 2)
            self.assertEqual(report.macro_f05, 1.0)
            self.assertEqual(report.correct_singletons, 1)
        finally:
            os.remove(gt_path)
            os.remove(pred_path)


if __name__ == "__main__":
    unittest.main()
