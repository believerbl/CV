"""
test_blocking_pass_e.py
========================
Synthetic tests for blocking.py Pass E (script-aware blocking).
Uses raw row dicts as input (the shape generate_candidates() accepts),
so it also exercises normalize_dataframe_rows() indirectly.

Pass-E-specific assertions call _pass_e_script_aware() directly so that
Passes A/B/C/D cannot mask the result; integration assertions call
generate_candidates() to verify union behavior.

Run with: python -m unittest test_blocking_pass_e -v
"""

import unittest

from blocking import (
    generate_candidates,
    _pass_a_exact_core_name,
    _pass_e_script_aware,
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


class TestSameScriptDevanagariMatch(unittest.TestCase):
    def test_a_misses_e_hits(self):
        s1 = [row("S1-001", "शर्मा टेक्सटाइल्स")]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_a_exact_core_name(s1n, s2n, s3n, {})["S1-001"], [])
        self.assertIn("S2-001", _pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"])
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config={})["S1-001"])


class TestCrossScriptNonMatch(unittest.TestCase):
    def test_latin_vs_devanagari_does_not_cross_match(self):
        s1 = [row("S1-001", "Sharma Textiles")]
        s2 = [row("S2-001", "शर्मा टेक्सटाइल्स")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"], [])


class TestMixedScriptExpansion(unittest.TestCase):
    def test_mixed_s1_reaches_both_partitions(self):
        s1 = [row("S1-001", "Sharma शर्मा Textiles")]
        s2 = [row("S2-001", "Sharma Traders")]
        s3 = [row("S3-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        out = _pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"]
        self.assertIn("S2-001", out)  # via (latin, sharma)
        self.assertIn("S3-001", out)  # via (devanagari, शर्मा)

    def test_no_expansion_queries_primary_only(self):
        s1 = [row("S1-001", "Sharma शर्मा Textiles")]
        s2 = [row("S2-001", "Sharma Traders")]
        s3 = [row("S3-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        # Latin dominates ("Sharma"+"Textiles" = 14 latin chars vs 5
        # Devanagari), so primary is latin: only S2-001 reachable.
        cfg = {"pass_e_use_mixed_expansion": False}
        out = _pass_e_script_aware(s1n, s2n, s3n, cfg)["S1-001"]
        self.assertIn("S2-001", out)
        self.assertNotIn("S3-001", out)


class TestLatinCommonTokenSuppression(unittest.TestCase):
    def test_ubiquitous_latin_token_suppressed(self):
        s2 = [row(f"S2-{i:04d}", "Global Traders Hub") for i in range(100)]
        s1 = [row("S1-001", "Global Traders Hub")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # Each token DF=100 > threshold (min(50, max(2, 1)) = 2).
        self.assertEqual(_pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"], [])


class TestBlankNameSafety(unittest.TestCase):
    def test_none_name_no_candidates(self):
        s1 = [row("S1-001", None)]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"], [])

    def test_whitespace_names_never_cross_match(self):
        s1 = [row("S1-001", "   ")]
        s2 = [row("S2-001", ""), row("S2-002", None)]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"], [])


class TestMissingScriptSafety(unittest.TestCase):
    def test_digit_only_names_form_no_bucket(self):
        # Digits are non-alpha: detect_scripts() yields no script.
        s1 = [row("S1-001", "12345")]
        s2 = [row("S2-001", "12345"), row("S2-002", "12345")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(_pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"], [])


class TestMaxCandidateCap(unittest.TestCase):
    def test_cap_bounds_fan_out_deterministically(self):
        s2 = [row(f"S2-{i:04d}", "शर्मा टेक्सटाइल्स") for i in range(10)]
        s1 = [row("S1-001", "शर्मा Foods")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        cfg = {
            "pass_e_max_df_floor": 500,
            "pass_e_max_df_ratio": 1.0,
            "pass_e_max_candidates": 5,
        }
        out = _pass_e_script_aware(s1n, s2n, s3n, cfg)["S1-001"]
        self.assertEqual(len(out), 5)
        self.assertEqual(out, sorted(out))
        self.assertEqual(
            out, ["S2-0000", "S2-0001", "S2-0002", "S2-0003", "S2-0004"]
        )


class TestOnlyS2S3IdsReturned(unittest.TestCase):
    def test_no_s1_ids_leak(self):
        s1 = [row("S1-001", "शर्मा टेक्सटाइल्स"), row("S1-002", "शर्मा टेक्सटाइल्स")]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        s3 = [row("S3-001", "शर्मा एंटरप्राइजेज")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        emap = _pass_e_script_aware(s1n, s2n, s3n, {})
        for s1_id, cands in emap.items():
            for cid in cands:
                self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"))
                self.assertNotEqual(cid, s1_id)


class TestEveryS1GetsEntry(unittest.TestCase):
    def test_all_s1_keys_preserved(self):
        s1 = [
            row("S1-001", "शर्मा टेक्सटाइल्स"),
            row("S1-002", "Totally Different Words Here"),
            row("S1-003", None),
        ]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        emap = _pass_e_script_aware(s1n, s2n, s3n, {})
        self.assertEqual(set(emap.keys()), {"S1-001", "S1-002", "S1-003"})
        self.assertIn("S2-001", emap["S1-001"])
        self.assertEqual(emap["S1-003"], [])


class TestDeterministicOutput(unittest.TestCase):
    def test_repeated_runs_identical_and_sorted(self):
        s1 = [row("S1-001", "Sharma शर्मा Textiles")]
        s2 = [
            row("S2-002", "शर्मा ट्रेडर्स"),
            row("S2-001", "Sharma Traders"),
        ]
        s3 = [row("S3-001", "Textiles Sharma")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        outs = [_pass_e_script_aware(s1n, s2n, s3n, {}) for _ in range(5)]
        for o in outs[1:]:
            self.assertEqual(o, outs[0])
        self.assertEqual(outs[0]["S1-001"], sorted(outs[0]["S1-001"]))


class TestConfigDisabling(unittest.TestCase):
    def test_all_off_equals_pure_pass_a(self):
        s1 = [row("S1-001", "शर्मा टेक्सटाइल्स")]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        pure_a = _pass_a_exact_core_name(s1n, s2n, s3n, {})
        self.assertEqual(generate_candidates(s1, s2, [], config=dict(BASE_OFF)), pure_a)
        self.assertEqual(
            generate_candidates(s1, s2, [], config=dict(BASE_OFF))["S1-001"], []
        )

    def test_e_flag_isolates_e_contribution(self):
        s1 = [row("S1-001", "शर्मा टेक्सटाइल्स")]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स")]
        on = dict(BASE_OFF)
        off = dict(BASE_OFF)
        on["pass_e_enabled"] = True
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config=on)["S1-001"])
        self.assertNotIn(
            "S2-001", generate_candidates(s1, s2, [], config=off)["S1-001"]
        )


class TestCrossCountryRetained(unittest.TestCase):
    def test_france_india_same_script_token_retained(self):
        s1 = [row("S1-001", "शर्मा टेक्सटाइल्स", country="France")]
        s2 = [row("S2-001", "शर्मा ट्रेडर्स", country="India")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertIn("S2-001", _pass_e_script_aware(s1n, s2n, s3n, {})["S1-001"])
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config={})["S1-001"])


class TestUnionWithABCD(unittest.TestCase):
    def test_e_candidate_appears_alongside_a_candidate(self):
        s1 = [
            row("S1-001", "Acme Corp"),
            row("S1-002", "शर्मा टेक्सटाइल्स"),
        ]
        s2 = [
            row("S2-001", "Acme Corp"),
            row("S2-002", "शर्मा ट्रेडर्स"),
        ]
        cmap = generate_candidates(s1, s2, [], config={})
        self.assertIn("S2-001", cmap["S1-001"])  # via Pass A
        self.assertIn("S2-002", cmap["S1-002"])  # via Pass E


if __name__ == "__main__":
    unittest.main()
