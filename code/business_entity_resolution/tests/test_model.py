"""
test_model.py
=============
Unit tests for model.py (Parimarjan Work Package — Step 4).
Verifies the ML matcher contract against the SRS specification:
    ✓ training works
    ✓ positive/negative labels are generated correctly
    ✓ feature dimension must match
    ✓ deterministic training/prediction
    ✓ save/load produces identical predictions
    ✓ scores are produced for every candidate pair
    ✓ empty candidate input is handled safely
    ✓ malformed feature input fails clearly
"""

import os
import shutil
import tempfile
import unittest

from normalize import normalize_record
from features import FEATURE_NAMES, build_feature_vector
from model import (
    ERModel,
    build_training_dataset,
    train_model,
    predict_scores,
    predict_pair_scores,
    save_model,
    load_model,
    HAS_SKLEARN,
    EXPECTED_FEATURE_DIM,
)

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for model tests")
class TestModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Build synthetic normalized records for testing
        self.s1_1 = normalize_record("S1-1", "Alpha Tech Solutions", "100 Innovation Way", "US")
        self.s1_2 = normalize_record("S1-2", "Beta Retail Store", "200 Market Street", "US")
        self.s1_singleton = normalize_record("S1-3", "Gamma Unique Corp", "300 Solo Blvd", "US")

        self.s2_1 = normalize_record("S2-1", "Alpha Technologies", "100 Innovation Way Suite A", "US")
        self.s2_2 = normalize_record("S2-2", "Alpha Motors", "999 Other Road", "US")
        self.s3_1 = normalize_record("S3-1", "Beta Store LLC", "200 Market St", "US")
        self.s3_2 = normalize_record("S3-2", "Beta Random", "555 Far Away", "US")

        self.s1_lookup = {
            "S1-1": self.s1_1,
            "S1-2": self.s1_2,
            "S1-3": self.s1_singleton,
        }
        self.cand_lookup = {
            "S2-1": self.s2_1,
            "S2-2": self.s2_2,
            "S3-1": self.s3_1,
            "S3-2": self.s3_2,
        }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_synthetic_train_data(self, n_samples: int = 40):
        """Helper to create synthetic 30-dim feature matrix with binary labels."""
        rng = np.random.RandomState(42)
        X = rng.rand(n_samples, EXPECTED_FEATURE_DIM).astype(float)
        # Create correlated labels so model learns meaningful weights
        y = ((X[:, 0] + X[:, 2] + X[:, 16] - X[:, 18]) > 1.0).astype(int)
        # Ensure both classes are present
        y[0] = 1
        y[1] = 0
        return X, y

    # -------------------------------------------------------------------------
    # 1. Positive/Negative Labels Generated Correctly
    # -------------------------------------------------------------------------
    def test_positive_negative_labels_generated_correctly(self):
        # Candidate map contains true and false candidates
        candidate_map = {
            "S1-1": ["S2-1", "S2-2"],
            "S1-2": ["S3-1", "S3-2"],
            "S1-3": ["S2-2"],  # S1-3 is a singleton in truth, so S2-2 is negative
        }
        ground_truth = {
            "S1-1": {"S2-1"},       # S2-1 is positive, S2-2 is negative
            "S1-2": {"S3-1"},       # S3-1 is positive, S3-2 is negative
            "S1-3": set(),          # Singleton: no true matches
        }

        X, y, pair_ids = build_training_dataset(
            candidate_map, ground_truth, self.s1_lookup, self.cand_lookup
        )

        self.assertEqual(len(X), 5)
        self.assertEqual(len(y), 5)
        self.assertEqual(len(pair_ids), 5)

        # Map pair_id to label
        pair_to_label = {pid: lbl for pid, lbl in zip(pair_ids, y)}
        self.assertEqual(pair_to_label[("S1-1", "S2-1")], 1)  # Positive
        self.assertEqual(pair_to_label[("S1-1", "S2-2")], 0)  # Negative
        self.assertEqual(pair_to_label[("S1-2", "S3-1")], 1)  # Positive
        self.assertEqual(pair_to_label[("S1-2", "S3-2")], 0)  # Negative
        self.assertEqual(pair_to_label[("S1-3", "S2-2")], 0)  # Negative (singleton true set is empty)

    def test_training_data_balancing(self):
        # 1 positive and 4 negatives
        candidate_map = {"S1-1": ["S2-1", "S2-2", "S3-1", "S3-2"]}
        ground_truth = {"S1-1": {"S2-1"}}

        # Balance with ratio 2.0 -> at most 2 negatives kept
        X, y, pair_ids = build_training_dataset(
            candidate_map, ground_truth, self.s1_lookup, self.cand_lookup,
            balance_ratio=2.0, random_state=42
        )

        n_pos = sum(1 for label in y if label == 1)
        n_neg = sum(1 for label in y if label == 0)

        self.assertEqual(n_pos, 1)
        self.assertEqual(n_neg, 2)
        self.assertEqual(len(X), 3)

    # -------------------------------------------------------------------------
    # 2. Training Works
    # -------------------------------------------------------------------------
    def test_training_works(self):
        X, y = self._create_synthetic_train_data(50)
        model = train_model(X, y, model_type="hist_gbdt", random_state=42)

        self.assertIsInstance(model, ERModel)
        self.assertEqual(model.expected_dim, 30)

        probs = model.predict_proba(X)
        self.assertEqual(len(probs), 50)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    # -------------------------------------------------------------------------
    # 3. Feature Dimension Must Match
    # -------------------------------------------------------------------------
    def test_feature_dimension_must_match(self):
        X, y = self._create_synthetic_train_data(20)

        # 29 features instead of 30 on train
        X_29 = X[:, :29]
        with self.assertRaises(ValueError) as ctx:
            train_model(X_29, y)
        self.assertIn("30", str(ctx.exception))

        # 31 features instead of 30 on train
        X_31 = np.hstack([X, np.zeros((20, 1))])
        with self.assertRaises(ValueError) as ctx:
            train_model(X_31, y)
        self.assertIn("30", str(ctx.exception))

        # Predict with wrong dimension
        model = train_model(X, y, random_state=42)
        with self.assertRaises(ValueError) as ctx:
            model.predict_proba(X_29)
        self.assertIn("30", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Deterministic Training and Prediction
    # -------------------------------------------------------------------------
    def test_deterministic_training_and_prediction(self):
        X, y = self._create_synthetic_train_data(60)
        X_test, _ = self._create_synthetic_train_data(20)

        m1 = train_model(X, y, model_type="hist_gbdt", random_state=42)
        m2 = train_model(X, y, model_type="hist_gbdt", random_state=42)

        preds1 = m1.predict_proba(X_test)
        preds2 = m2.predict_proba(X_test)

        np.testing.assert_array_equal(preds1, preds2)

    # -------------------------------------------------------------------------
    # 5. Save/Load Produces Identical Predictions
    # -------------------------------------------------------------------------
    def test_save_load_produces_identical_predictions(self):
        X, y = self._create_synthetic_train_data(40)
        X_test, _ = self._create_synthetic_train_data(15)

        model = train_model(X, y, model_type="hist_gbdt", random_state=42)
        preds_original = model.predict_proba(X_test)

        save_path = os.path.join(self.temp_dir, "test_model.joblib")
        save_model(model, save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_model = load_model(save_path)
        preds_loaded = loaded_model.predict_proba(X_test)

        np.testing.assert_array_equal(preds_original, preds_loaded)
        self.assertEqual(loaded_model.feature_names, FEATURE_NAMES)

    def test_load_nonexistent_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model(os.path.join(self.temp_dir, "does_not_exist.joblib"))

    # -------------------------------------------------------------------------
    # 6. Scores Are Produced for Every Candidate Pair
    # -------------------------------------------------------------------------
    def test_scores_produced_for_every_candidate_pair(self):
        X, y = self._create_synthetic_train_data(30)
        model = train_model(X, y, random_state=42)

        candidate_map = {
            "S1-1": ["S2-1", "S2-2"],
            "S1-2": ["S3-1"],
            "S1-3": ["S3-2"],
        }

        scores = predict_scores(model, candidate_map, self.s1_lookup, self.cand_lookup)

        # Check all S1 entities present
        self.assertEqual(set(scores.keys()), {"S1-1", "S1-2", "S1-3"})

        # Check all candidate IDs present
        self.assertEqual(set(scores["S1-1"].keys()), {"S2-1", "S2-2"})
        self.assertEqual(set(scores["S1-2"].keys()), {"S3-1"})
        self.assertEqual(set(scores["S1-3"].keys()), {"S3-2"})

        # Check all scores are floats in [0.0, 1.0]
        for s1_id, cand_scores in scores.items():
            for cand_id, score in cand_scores.items():
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    # -------------------------------------------------------------------------
    # 7. Empty Candidate Input Is Handled Safely
    # -------------------------------------------------------------------------
    def test_empty_candidate_input_handled_safely(self):
        X, y = self._create_synthetic_train_data(20)
        model = train_model(X, y, random_state=42)

        # Completely empty candidate map
        scores_empty = predict_scores(model, {}, self.s1_lookup, self.cand_lookup)
        self.assertEqual(scores_empty, {})

        # S1 entity with 0 candidates
        scores_no_cands = predict_scores(
            model, {"S1-1": []}, self.s1_lookup, self.cand_lookup
        )
        self.assertEqual(scores_no_cands, {"S1-1": {}})

        # Empty feature array
        empty_preds = predict_pair_scores(model, np.empty((0, EXPECTED_FEATURE_DIM)))
        self.assertEqual(len(empty_preds), 0)

    # -------------------------------------------------------------------------
    # 8. Malformed Feature Input Fails Clearly
    # -------------------------------------------------------------------------
    def test_malformed_feature_input_fails_clearly(self):
        X, y = self._create_synthetic_train_data(20)
        model = train_model(X, y, random_state=42)

        # None input to predict
        with self.assertRaises(ValueError):
            model.predict_proba(None)

        # String input to predict
        with self.assertRaises(ValueError):
            model.predict_proba([["not", "a", "number"] * 10])

        # 1D scalar-like array
        with self.assertRaises(ValueError):
            model.predict_proba(np.array(5.0))

        # Empty train data
        with self.assertRaises(ValueError):
            train_model(np.empty((0, EXPECTED_FEATURE_DIM)), np.empty((0,)))

        # Mismatched lengths
        with self.assertRaises(ValueError):
            train_model(X, y[:10])


if __name__ == "__main__":
    unittest.main()
