"""
test_decision.py
================
Unit tests for decision.py (Parimarjan Work Package — Step 5).
Verifies the set-decision layer contract against the SRS specification:
    ✓ threshold keeps correct candidates
    ✓ threshold removes weak candidates
    ✓ zero-match output works
    ✓ one-match output works
    ✓ many-match output works
    ✓ no duplicate IDs
    ✓ deterministic ordering
    ✓ numeric-conflict rule behaves correctly
    ✓ optional cap behaves correctly
    ✓ empty candidate input works
    ✓ malformed scores fail clearly
"""

import math
import unittest

from decision import (
    select_matches_for_entity,
    make_decisions,
    sweep_thresholds,
    DecisionConfig,
)


class TestDecision(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Threshold Keeps Correct Candidates
    # -------------------------------------------------------------------------
    def test_threshold_keeps_correct_candidates(self):
        scores = {"S2-1": 0.85, "S2-2": 0.60, "S3-1": 0.35}
        matches = select_matches_for_entity(scores, threshold=0.50)
        self.assertEqual(matches, ["S2-1", "S2-2"])

    # -------------------------------------------------------------------------
    # 2. Threshold Removes Weak Candidates
    # -------------------------------------------------------------------------
    def test_threshold_removes_weak_candidates(self):
        scores = {"S2-1": 0.45, "S2-2": 0.30, "S3-1": 0.15}
        matches = select_matches_for_entity(scores, threshold=0.50)
        self.assertEqual(matches, [])

    # -------------------------------------------------------------------------
    # 3. Zero-Match Output Works
    # -------------------------------------------------------------------------
    def test_zero_match_output_works(self):
        # All below threshold
        scores = {"S2-1": 0.2, "S3-1": 0.1}
        self.assertEqual(select_matches_for_entity(scores, threshold=0.3), [])

        # Singleton gate suppresses low confidence candidates
        scores2 = {"S2-1": 0.45, "S2-2": 0.40}
        self.assertEqual(
            select_matches_for_entity(scores2, threshold=0.40, singleton_threshold=0.50),
            [],
        )

    # -------------------------------------------------------------------------
    # 4. One-Match Output Works
    # -------------------------------------------------------------------------
    def test_one_match_output_works(self):
        scores = {"S2-1": 0.92, "S2-2": 0.35, "S3-1": 0.15}
        matches = select_matches_for_entity(scores, threshold=0.50)
        self.assertEqual(matches, ["S2-1"])
        self.assertEqual(len(matches), 1)

    # -------------------------------------------------------------------------
    # 5. Many-Match Output Works
    # -------------------------------------------------------------------------
    def test_many_match_output_works(self):
        # Multi-match entity with 7 matches (supports multi-match training distribution)
        scores = {
            "S2-1": 0.95,
            "S2-2": 0.88,
            "S2-3": 0.75,
            "S2-4": 0.82,
            "S3-1": 0.91,
            "S3-2": 0.65,
            "S3-3": 0.70,
            "S3-distractor": 0.20,
        }
        matches = select_matches_for_entity(scores, threshold=0.50)
        expected = ["S2-1", "S2-2", "S2-3", "S2-4", "S3-1", "S3-2", "S3-3"]
        self.assertEqual(matches, sorted(expected))
        self.assertEqual(len(matches), 7)

    # -------------------------------------------------------------------------
    # 6. No Duplicate IDs
    # -------------------------------------------------------------------------
    def test_no_duplicate_ids(self):
        scores = {"S2-1": 0.9, "S2-2": 0.8}
        matches = select_matches_for_entity(scores, threshold=0.5)
        self.assertEqual(len(matches), len(set(matches)))

    # -------------------------------------------------------------------------
    # 7. Deterministic Ordering
    # -------------------------------------------------------------------------
    def test_deterministic_ordering(self):
        # Equal scores
        scores = {"S2-Gamma": 0.8, "S2-Alpha": 0.8, "S3-Beta": 0.8}
        matches1 = select_matches_for_entity(scores, threshold=0.5, output_order="id_asc")
        matches2 = select_matches_for_entity(scores, threshold=0.5, output_order="id_asc")
        self.assertEqual(matches1, matches2)
        self.assertEqual(matches1, ["S2-Alpha", "S2-Gamma", "S3-Beta"])

        # Score desc order with tie-breaking
        matches_desc = select_matches_for_entity(scores, threshold=0.5, output_order="score_desc")
        self.assertEqual(matches_desc, ["S2-Alpha", "S2-Gamma", "S3-Beta"])

    # -------------------------------------------------------------------------
    # 8. Numeric-Conflict Rule Behaves Correctly
    # -------------------------------------------------------------------------
    def test_numeric_conflict_rule_behaves_correctly(self):
        scores = {"S2-1": 0.90, "S2-Conflict": 0.85}
        conflicts = {"S2-Conflict"}

        # Without suppression: both are kept
        matches_no_suppress = select_matches_for_entity(
            scores,
            threshold=0.5,
            numeric_conflicts=conflicts,
            suppress_numeric_conflict=False,
        )
        self.assertEqual(matches_no_suppress, ["S2-1", "S2-Conflict"])

        # With suppression: conflicting candidate is excluded
        matches_suppressed = select_matches_for_entity(
            scores,
            threshold=0.5,
            numeric_conflicts=conflicts,
            suppress_numeric_conflict=True,
        )
        self.assertEqual(matches_suppressed, ["S2-1"])

    # -------------------------------------------------------------------------
    # 9. Optional Cap Behaves Correctly
    # -------------------------------------------------------------------------
    def test_optional_cap_behaves_correctly(self):
        scores = {
            "S2-Low": 0.60,
            "S2-High": 0.95,
            "S2-Mid": 0.80,
            "S2-MedLow": 0.70,
        }
        # Cap at top 2 matches
        matches_capped = select_matches_for_entity(
            scores, threshold=0.5, max_matches=2, output_order="score_desc"
        )
        self.assertEqual(matches_capped, ["S2-High", "S2-Mid"])
        self.assertEqual(len(matches_capped), 2)

    # -------------------------------------------------------------------------
    # 10. Empty Candidate Input Works
    # -------------------------------------------------------------------------
    def test_empty_candidate_input_works(self):
        # Empty dict to single entity
        self.assertEqual(select_matches_for_entity({}), [])

        # Batch interface with empty dict
        self.assertEqual(make_decisions({}), {})

        # Batch interface with entity having no candidates
        batch_input = {"S1-1": {}, "S1-2": {"S2-1": 0.9}}
        decisions = make_decisions(batch_input, threshold=0.5)
        self.assertEqual(decisions, {"S1-1": [], "S1-2": ["S2-1"]})

    # -------------------------------------------------------------------------
    # 11. Malformed Scores Fail Clearly
    # -------------------------------------------------------------------------
    def test_malformed_scores_fail_clearly(self):
        # None scores
        with self.assertRaises(ValueError):
            select_matches_for_entity(None)

        # None batch scores
        with self.assertRaises(ValueError):
            make_decisions(None)

        # Non-numeric score string
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": "high"})

        # NaN score
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": float("nan")})

        # Out-of-bounds scores
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": 1.5})
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": -0.1})

        # Invalid threshold
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": 0.8}, threshold=-0.1)
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": 0.8}, threshold=1.5)
        with self.assertRaises(ValueError):
            select_matches_for_entity({"S2-1": 0.8}, threshold="bad")

    # -------------------------------------------------------------------------
    # 12. DecisionConfig Integration & Batch Sweep
    # -------------------------------------------------------------------------
    def test_decision_config_and_threshold_sweep(self):
        config = DecisionConfig(threshold=0.6, output_order="id_asc")
        scores = {
            "S1-1": {"S2-1": 0.8, "S2-2": 0.4},
            "S1-2": {"S3-1": 0.3},
        }
        decisions = make_decisions(scores, config=config)
        self.assertEqual(decisions, {"S1-1": ["S2-1"], "S1-2": []})

        # Test threshold sweep
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),  # true singleton
        }
        sweep = sweep_thresholds(scores, gt, thresholds=[0.3, 0.5, 0.7])
        self.assertEqual(len(sweep), 3)
        for row in sweep:
            self.assertIn("threshold", row)
            self.assertIn("macro_f05", row)
            self.assertIn("empty_set_rate", row)
            self.assertIn("singleton_accuracy", row)


if __name__ == "__main__":
    unittest.main()
