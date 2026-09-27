"""
test_pipeline.py
================
Unit tests for pipeline.py (Parimarjan Work Package — Step 6).
Verifies the end-to-end entity resolution pipeline:
    ✓ End-to-end execution from raw records to final matches
    ✓ Candidate pair generation + subset invariant (M subseteq C)
    ✓ Tab-separated submission format compliance
    ✓ Singleton empty match handling
"""

import os
import shutil
import tempfile
import unittest

from normalize import normalize_record
from features import FEATURE_NAMES
from model import ERModel
from decision import DecisionConfig
from pipeline import run_pipeline, write_submission_files

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn required for pipeline tests")
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Build synthetic records
        self.s1_records = [
            {"entity_id": "S1-1", "business_name": "Apex Tech Solutions", "business_address": "100 Market St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Lone Star Retail", "business_address": "500 Desert Rd", "country": "India"},
        ]
        self.s2_records = [
            {"entity_id": "S2-1", "business_name": "Apex Tech Solutions LLC", "business_address": "100 Market St Suite 1", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Unrelated Store", "business_address": "999 Far Away", "country": "India"},
        ]
        self.s3_records = [
            {"entity_id": "S3-1", "business_name": "Apex Tech Corp", "business_address": "100 Market St", "country": "US"},
        ]

        # Train a mock model that predicts high score for high name overlap
        rng = np.random.RandomState(42)
        X = rng.rand(40, len(FEATURE_NAMES))
        y = (X[:, 0] > 0.5).astype(int)
        y[0], y[1] = 1, 0
        estimator = HistGradientBoostingClassifier(max_iter=20, random_state=42)
        estimator.fit(X, y)
        self.model = ERModel(estimator=estimator, feature_names=FEATURE_NAMES)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_pipeline_execution(self):
        matching_results, candidate_pairs = run_pipeline(
            s1_records=self.s1_records,
            s2_records=self.s2_records,
            s3_records=self.s3_records,
            model=self.model,
            decision_config=DecisionConfig(threshold=0.2, suppress_numeric_conflict=True),
        )

        # 1. Every S1 entity has an entry
        self.assertIn("S1-1", matching_results)
        self.assertIn("S1-2", matching_results)
        self.assertIn("S1-1", candidate_pairs)
        self.assertIn("S1-2", candidate_pairs)

        # 2. Invariant: M subseteq C
        for sid, matches in matching_results.items():
            cands = set(candidate_pairs.get(sid, []))
            for m in matches:
                self.assertIn(m, cands, f"Match {m} for {sid} is not in candidate set {cands}")

    def test_write_submission_files_format(self):
        matching_results = {
            "S1-1": ["S2-1", "S3-1"],
            "S1-2": [],  # Singleton
        }
        candidate_pairs = {
            "S1-1": ["S2-1", "S2-2", "S3-1"],
            "S1-2": ["S2-2"],
        }

        m_path = os.path.join(self.temp_dir, "matching_results.tsv")
        c_path = os.path.join(self.temp_dir, "candidate_pairs.tsv")

        write_submission_files(matching_results, candidate_pairs, m_path, c_path)

        # Verify matching_results.tsv
        with open(m_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]

        self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
        self.assertEqual(lines[1], "S1-1\tS2-1,S3-1")
        self.assertEqual(lines[2], "S1-2\t")  # Empty match set has empty string after tab

        # Verify candidate_pairs.tsv
        with open(c_path, "r", encoding="utf-8") as f:
            c_lines = [line.rstrip("\r\n") for line in f]

        self.assertEqual(c_lines[0], "source1_entity_id\tcandidate_entity_ids")
        self.assertEqual(c_lines[1], "S1-1\tS2-1,S2-2,S3-1")
        self.assertEqual(c_lines[2], "S1-2\tS2-2")


if __name__ == "__main__":
    unittest.main()
