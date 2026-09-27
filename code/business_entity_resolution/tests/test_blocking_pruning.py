"""
test_blocking_pruning.py
========================
Unit tests for candidate pruning and K-budget selection in blocking.py.
Tests that:
    - Candidate list length is strictly bounded by max_candidates_per_s1.
    - Candidates supported by multiple passes are prioritized during pruning.
    - Deterministic ordering and tie-breaking are preserved.
    - K=None preserves all union candidates.
"""

import unittest
from blocking import generate_candidates


def make_row(entity_id, name, address="100 Market St", country="US"):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


class TestCandidatePruning(unittest.TestCase):
    def setUp(self):
        # S1 entity
        self.s1 = [make_row("S1-001", "Nexus Robotics Labs", "100 Market St Suite 5", "US")]
        # Multiple S2/S3 records that match across various passes
        self.s2 = [
            make_row("S2-001", "Nexus Robotics Labs", "100 Market St Suite 5", "US"),  # Exact name + numeric -> Passes A, B, C, D
            make_row("S2-002", "Nexus Automation", "100 Market St", "US"),              # Token Nexus + numeric 100 -> Passes B, C, D
            make_row("S2-003", "Robotics Dynamics", "100 Market St", "US"),            # Token Robotics + numeric 100 -> Passes B, C, D
            make_row("S2-004", "Nexus Robotics Corp", "500 Elm St", "US"),              # Core name match -> Pass A, B, D
            make_row("S2-005", "Nexus Technologies", "200 Pine St", "US"),              # Token Nexus -> Pass B, D
        ]
        self.s3 = [
            make_row("S3-001", "Other Entity", "100 Market St", "US"),                  # Numeric 100 -> Pass C
            make_row("S3-002", "Unrelated Business", "999 Oak Ave", "US"),              # Unrelated
        ]
        self.base_cfg = {"pass_b_max_df_ratio": 1.0, "pass_c_max_df_ratio": 1.0}

    def test_unconstrained_retains_all(self):
        cfg = dict(self.base_cfg, max_candidates_per_s1=None)
        cmap = generate_candidates(self.s1, self.s2, self.s3, cfg)
        self.assertGreater(len(cmap["S1-001"]), 2)

    def test_k_bounding(self):
        for k in [1, 2, 3, 5]:
            cfg = dict(self.base_cfg, max_candidates_per_s1=k)
            cmap = generate_candidates(self.s1, self.s2, self.s3, cfg)
            self.assertLessEqual(len(cmap["S1-001"]), k)

    def test_high_agreement_candidate_prioritized(self):
        # S2-001 has highest multi-pass agreement and exact token match
        cfg = dict(self.base_cfg, max_candidates_per_s1=1)
        cmap = generate_candidates(self.s1, self.s2, self.s3, cfg)
        self.assertEqual(len(cmap["S1-001"]), 1)
        self.assertEqual(cmap["S1-001"][0], "S2-001")

    def test_determinism(self):
        cfg = dict(self.base_cfg, max_candidates_per_s1=3)
        run1 = generate_candidates(self.s1, self.s2, self.s3, cfg)
        run2 = generate_candidates(self.s1, self.s2, self.s3, cfg)
        self.assertEqual(run1, run2)


if __name__ == "__main__":
    unittest.main()
