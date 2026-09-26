"""
test_blocking_pass_b.py
========================
Synthetic tests for blocking.py Pass B (rare/IDF token blocking).
Uses raw row dicts as input (the shape generate_candidates() accepts),
so it also exercises normalize_dataframe_rows() indirectly.

Pass-B-specific assertions call _pass_b_rare_token() directly so that
Passes A/C/D cannot mask the result; integration assertions call
generate_candidates() to verify union behavior.

Run with: python -m unittest test_blocking_pass_b -v
"""

import unittest

from blocking import (
    generate_candidates,
    _pass_a_exact_core_name,
    _pass_b_rare_token,
)
from normalize import normalize_dataframe_rows


def row(entity_id, name, address="1 Main St", country="India"):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


def _norm(s1, s2, s3):
    return (
        normalize_dataframe_rows(s1),
        normalize_dataframe_rows(s2),
        normalize_dataframe_rows(s3),
    )


BASE_OFF = {
    "pass_b_enabled": False,
    "pass_c_enabled": False,
    "pass_d_enabled": False,
    "pass_e_enabled": False,
}


class TestRareTokenMatch(unittest.TestCase):
    def test_shared_rare_token_blocks_together(self):
        s1 = [row("S1-001", "Zxqwv Traders")]
        s2 = [row("S2-001", "Zxqwv Foods")]
        s3 = []
        s1n, s2n, s3n = _norm(s1, s2, s3)
        # Pass A misses (different cores); Pass B hits via "zxqwv".
        self.assertEqual(_pass_a_exact_core_name(s1n, s2n, s3n, {})["S1-001"], [])
        self.assertIn("S2-001", _pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"])
        # Union also retains it.
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertIn("S2-001", cmap["S1-001"])


class TestCommonTokenSuppression(unittest.TestCase):
    def test_ubiquitous_token_produces_no_candidates(self):
        s2 = [row(f"S2-{i:04d}", "Global Traders Hub") for i in range(100)]
        s3 = []
        s1 = [row("S1-001", "Global Traders Hub")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        # Every token has DF=100 > threshold (min(50, max(2, 1)) = 2).
        self.assertEqual(_pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"], [])


class TestTypoTokenBehavior(unittest.TestCase):
    def test_single_typo_token_does_not_match_via_b(self):
        # Pass B is token-exact: "continental" != "continantal".
        s1 = [row("S1-001", "Continental")]
        s2 = [row("S2-001", "Continantal")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"], [])
        # The typo pair is Pass D's job: union still finds it.
        cmap = generate_candidates(s1, s2, [], config={})
        self.assertIn("S2-001", cmap["S1-001"])


class TestMultipleTokens(unittest.TestCase):
    def test_two_rare_tokens_reach_two_records(self):
        s1 = [row("S1-001", "Zxqwv Qwerty")]
        s2 = [row("S2-001", "Zxqwv Alpha"), row("S2-002", "Qwerty Beta")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        out = _pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"]
        self.assertEqual(set(out), {"S2-001", "S2-002"})


class TestMaxCandidateCap(unittest.TestCase):
    def test_cap_bounds_fan_out_deterministically(self):
        s2 = [row(f"S2-{i:04d}", "Zxqwv Traders") for i in range(10)]
        s1 = [row("S1-001", "Zxqwv Foods")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        cfg = {
            "pass_b_max_df_floor": 500,
            "pass_b_max_df_ratio": 1.0,
            "pass_b_max_candidates": 5,
        }
        out = _pass_b_rare_token(s1n, s2n, s3n, cfg)["S1-001"]
        self.assertEqual(len(out), 5)
        self.assertEqual(out, sorted(out))
        self.assertEqual(
            out, ["S2-0000", "S2-0001", "S2-0002", "S2-0003", "S2-0004"]
        )


class TestOnlyS2S3IdsReturned(unittest.TestCase):
    def test_no_s1_ids_leak(self):
        s1 = [row("S1-001", "Zxqwv Traders"), row("S1-002", "Zxqwv Traders")]
        s2 = [row("S2-001", "Zxqwv Foods")]
        s3 = [row("S3-001", "Zxqwv Goods")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        bmap = _pass_b_rare_token(s1n, s2n, s3n, {})
        for s1_id, cands in bmap.items():
            for cid in cands:
                self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"))
                self.assertNotEqual(cid, s1_id)


class TestEveryS1GetsEntry(unittest.TestCase):
    def test_all_s1_keys_preserved(self):
        s1 = [
            row("S1-001", "Zxqwv Traders"),
            row("S1-002", "Totally Different Words Here"),
            row("S1-003", None),
        ]
        s2 = [row("S2-001", "Zxqwv Foods")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        bmap = _pass_b_rare_token(s1n, s2n, s3n, {})
        self.assertEqual(set(bmap.keys()), {"S1-001", "S1-002", "S1-003"})
        self.assertIn("S2-001", bmap["S1-001"])
        self.assertEqual(bmap["S1-003"], [])


class TestDeterministicOutput(unittest.TestCase):
    def test_repeated_runs_identical_and_sorted(self):
        s1 = [row("S1-001", "Zxqwv Qwerty Traders")]
        s2 = [
            row("S2-002", "Qwerty Beta"),
            row("S2-001", "Zxqwv Alpha"),
            row("S2-003", "Traders Gamma"),
        ]
        s1n, s2n, s3n = _norm(s1, s2, [])
        outs = [_pass_b_rare_token(s1n, s2n, s3n, {}) for _ in range(5)]
        for o in outs[1:]:
            self.assertEqual(o, outs[0])
        self.assertEqual(outs[0]["S1-001"], sorted(outs[0]["S1-001"]))


class TestBlankNames(unittest.TestCase):
    def test_blank_s1_gets_no_candidates(self):
        s1 = [row("S1-001", None)]
        s2 = [row("S2-001", ""), row("S2-002", "   ")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"], [])

    def test_blank_s2_tokens_never_indexed(self):
        s1 = [row("S1-001", None)]
        s2 = [row("S2-001", None), row("S2-002", "")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"], [])


class TestNoBruteForceScaling(unittest.TestCase):
    def test_selective_retrieval_over_2000_records(self):
        s2 = [row(f"S2-{i:04d}", f"Unique Name {i}") for i in range(2000)]
        s1 = [row("S1-001", "Unique Name 500")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # "unique"/"name" are suppressed (DF=2000); "500" selects S2-0500.
        self.assertEqual(
            _pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"], ["S2-0500"]
        )


class TestConfigDisabling(unittest.TestCase):
    def test_all_off_equals_pure_pass_a(self):
        s1 = [row("S1-001", "Zxqwv Traders")]
        s2 = [row("S2-001", "Zxqwv Foods")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        pure_a = _pass_a_exact_core_name(s1n, s2n, s3n, {})
        self.assertEqual(generate_candidates(s1, s2, [], config=dict(BASE_OFF)), pure_a)

    def test_b_flag_isolates_b_contribution(self):
        s1 = [row("S1-001", "Zxqwv Traders")]
        s2 = [row("S2-001", "Zxqwv Foods")]
        on = dict(BASE_OFF)
        off = dict(BASE_OFF)
        on["pass_b_enabled"] = True
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config=on)["S1-001"])
        self.assertNotIn(
            "S2-001", generate_candidates(s1, s2, [], config=off)["S1-001"]
        )


class TestCrossCountryRetained(unittest.TestCase):
    def test_france_india_same_business_retained(self):
        s1 = [row("S1-001", "Zxqwv Traders", country="France")]
        s2 = [row("S2-001", "Zxqwv Foods", country="India")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertIn("S2-001", _pass_b_rare_token(s1n, s2n, s3n, {})["S1-001"])
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config={})["S1-001"])


if __name__ == "__main__":
    unittest.main()
