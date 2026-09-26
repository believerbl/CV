"""
test_blocking_pass_c.py
========================
Synthetic tests for blocking.py Pass C (numeric/address anchor blocking).
Uses raw row dicts as input (the shape generate_candidates() accepts),
so it also exercises normalize_dataframe_rows() indirectly.

Pass-C-specific assertions call _pass_c_numeric_address() directly so
that Passes A/B/D cannot mask the result; integration assertions call
generate_candidates() to verify union behavior.

Run with: python -m unittest test_blocking_pass_c -v
"""

import unittest

from blocking import (
    generate_candidates,
    _pass_a_exact_core_name,
    _pass_c_numeric_address,
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


class TestPostalTokenMatch(unittest.TestCase):
    def test_shared_postal_blocks_together(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704")]
        s2 = [row("S2-001", "Red Mountain Foods", "10 Other Rd, Shelbyville, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertIn("S2-001", _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"])
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config={})["S1-001"])


class TestNumericTokenMatch(unittest.TestCase):
    def test_shared_house_number_blocks_together(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "221B Baker Street, London")]
        s2 = [row("S2-001", "Red Mountain Foods", "Flat 221B, Baker Street, London")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertIn("S2-001", _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"])


class TestAddressAnchorMatch(unittest.TestCase):
    def test_distinctive_word_matches_when_enabled(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "Shop near Willowbrook Market")]
        s2 = [row("S2-001", "Red Mountain Foods", "Stall opposite Willowbrook Market")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # Default config ignores address words: no match.
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], []
        )
        # Opt-in address tokens: match on "willowbrook".
        cfg = {"pass_c_use_address_tokens": True}
        self.assertIn("S2-001", _pass_c_numeric_address(s1n, s2n, s3n, cfg)["S1-001"])

    def test_short_address_tokens_ignored(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "12 Main Rd")]
        s2 = [row("S2-001", "Red Mountain Foods", "12 Main Rd")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # "12" is filtered by min_numeric_len=3; short words need opt-in.
        cfg = {"pass_c_use_address_tokens": True, "pass_c_min_address_token_len": 10}
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, cfg)["S1-001"], []
        )


class TestMissingAddress(unittest.TestCase):
    def test_none_address_gets_no_candidates(self):
        s1 = [row("S1-001", "Blue Ocean Traders", None)]
        s2 = [row("S2-001", "Red Mountain Foods", "500 Market St, Springfield, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], []
        )

    def test_blank_addresses_never_cross_match(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "   ")]
        s2 = [row("S2-001", "Red Mountain Foods", ""), row("S2-002", "Silver Star Co", None)]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], []
        )


class TestCommonAnchorSuppression(unittest.TestCase):
    def test_ubiquitous_anchor_produces_no_candidates(self):
        s2 = [
            row(f"S2-{i:04d}", f"Business Number {i}", "12345 Main Street, Town")
            for i in range(100)
        ]
        s1 = [row("S1-001", "Completely Different Name Here", "12345 Main Street, Town")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # "12345" has DF=100 > threshold (min(50, max(2, 0)) = 2).
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], []
        )


class TestMaxCandidateCap(unittest.TestCase):
    def test_cap_bounds_fan_out_deterministically(self):
        s2 = [
            row(f"S2-{i:04d}", f"Business Number {i}", "500 Market St, Springfield, 62704")
            for i in range(10)
        ]
        s1 = [row("S1-001", "Completely Different Name Here", "500 Market St, Springfield, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        cfg = {
            "pass_c_max_df_floor": 500,
            "pass_c_max_df_ratio": 1.0,
            "pass_c_max_candidates": 5,
        }
        out = _pass_c_numeric_address(s1n, s2n, s3n, cfg)["S1-001"]
        self.assertEqual(len(out), 5)
        self.assertEqual(out, sorted(out))
        self.assertEqual(
            out, ["S2-0000", "S2-0001", "S2-0002", "S2-0003", "S2-0004"]
        )


class TestOnlyS2S3IdsReturned(unittest.TestCase):
    def test_no_s1_ids_leak(self):
        s1 = [
            row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704"),
            row("S1-002", "Blue Ocean Traders", "500 Market St, Springfield, 62704"),
        ]
        s2 = [row("S2-001", "Other Name One", "500 Market St, Springfield, 62704")]
        s3 = [row("S3-001", "Other Name Two", "500 Market St, Springfield, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        cmap = _pass_c_numeric_address(s1n, s2n, s3n, {})
        for s1_id, cands in cmap.items():
            for cid in cands:
                self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"))
                self.assertNotEqual(cid, s1_id)


class TestEveryS1GetsEntry(unittest.TestCase):
    def test_all_s1_keys_preserved(self):
        s1 = [
            row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704"),
            row("S1-002", "Unmatched Name Here", "99 Nowhere Lane, Villagetown"),
            row("S1-003", "No Address Co", None),
        ]
        s2 = [row("S2-001", "Red Mountain Foods", "10 Other Rd, Shelbyville, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        cmap = _pass_c_numeric_address(s1n, s2n, s3n, {})
        self.assertEqual(set(cmap.keys()), {"S1-001", "S1-002", "S1-003"})
        self.assertIn("S2-001", cmap["S1-001"])
        self.assertEqual(cmap["S1-003"], [])


class TestDeterministicOutput(unittest.TestCase):
    def test_repeated_runs_identical_and_sorted(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704")]
        s2 = [
            row("S2-002", "Other Name Two", "10 Other Rd, Shelbyville, 62704"),
            row("S2-001", "Other Name One", "500 Market St, Springfield, 62704"),
        ]
        s1n, s2n, s3n = _norm(s1, s2, [])
        outs = [_pass_c_numeric_address(s1n, s2n, s3n, {}) for _ in range(5)]
        for o in outs[1:]:
            self.assertEqual(o, outs[0])
        self.assertEqual(outs[0]["S1-001"], sorted(outs[0]["S1-001"]))


class TestConfigDisabling(unittest.TestCase):
    def test_all_off_equals_pure_pass_a(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704")]
        s2 = [row("S2-001", "Red Mountain Foods", "10 Other Rd, Shelbyville, 62704")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        pure_a = _pass_a_exact_core_name(s1n, s2n, s3n, {})
        self.assertEqual(generate_candidates(s1, s2, [], config=dict(BASE_OFF)), pure_a)

    def test_c_flag_isolates_c_contribution(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704")]
        s2 = [row("S2-001", "Red Mountain Foods", "10 Other Rd, Shelbyville, 62704")]
        on = dict(BASE_OFF)
        off = dict(BASE_OFF)
        on["pass_c_enabled"] = True
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config=on)["S1-001"])
        self.assertNotIn(
            "S2-001", generate_candidates(s1, s2, [], config=off)["S1-001"]
        )


class TestMultipleAnchors(unittest.TestCase):
    def test_postal_and_numeric_reach_two_records(self):
        s1 = [row("S1-001", "Completely Different Name", "221B Baker Street, London 62704")]
        s2 = [row("S2-001", "Another Business One", "Flat 221B, Baker Street, London 99999")]
        s3 = [row("S3-001", "Another Business Two", "77 Other Road, Town 62704")]
        s1n, s2n, s3n = _norm(s1, s2, s3)
        out = _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"]
        self.assertIn("S2-001", out)  # via "221b"
        self.assertIn("S3-001", out)  # via "62704"


class TestCrossCountryRetained(unittest.TestCase):
    def test_france_india_same_postal_retained(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "500 Market St, Springfield, 62704", "France")]
        s2 = [row("S2-001", "Red Mountain Foods", "10 Other Rd, Shelbyville, 62704", "India")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertIn("S2-001", _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"])
        self.assertIn("S2-001", generate_candidates(s1, s2, [], config={})["S1-001"])


class TestBlankSafeBehavior(unittest.TestCase):
    def test_whitespace_address_safe(self):
        s1 = [row("S1-001", "Blue Ocean Traders", "   ")]
        s2 = [row("S2-001", "Red Mountain Foods", "   ")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], []
        )


class TestNoBruteForceScan(unittest.TestCase):
    def test_selective_retrieval_over_2000_records(self):
        s2 = [
            row(f"S2-{i:04d}", f"Business Number {i}", f"Street {i}, Town, {90000 + i}")
            for i in range(2000)
        ]
        s1 = [row("S1-001", "Completely Different Name Here", "Street 500, Town, 90500")]
        s1n, s2n, s3n = _norm(s1, s2, [])
        # Postal "90500" is unique (DF=1); only S2-0500 matches.
        self.assertEqual(
            _pass_c_numeric_address(s1n, s2n, s3n, {})["S1-001"], ["S2-0500"]
        )


if __name__ == "__main__":
    unittest.main()
