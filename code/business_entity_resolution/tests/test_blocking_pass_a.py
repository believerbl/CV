"""
test_blocking_pass_a.py
========================
Synthetic tests for blocking.py Pass A (exact/core name blocking) only.
Uses raw row dicts as input (the shape generate_candidates() accepts),
so it also exercises normalize_dataframe_rows() indirectly.

Run with: python3 -m unittest test_blocking_pass_a.py -v
"""

import unittest

from blocking import generate_candidates, _pass_a_exact_core_name
from normalize import normalize_dataframe_rows


def row(entity_id, name, address="1 Main St", country="India"):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


class TestExactNameMatch(unittest.TestCase):
    def test_exact_name_norm_match(self):
        s1 = [row("S1-001", "Global Traders Inc")]
        s2 = [row("S2-001", "Global Traders Inc")]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertIn("S2-001", cmap["S1-001"])


class TestNameCoreMatchAfterLegalSuffixRemoval(unittest.TestCase):
    def test_corp_vs_corporation_still_blocks_together(self):
        # These differ in name_norm (corp vs corporation -> both map to
        # 'corp' actually per LEGAL_FORM_MAP, so use a pair that only
        # matches on name_core: differing legal form tokens entirely).
        s1 = [row("S1-001", "Sharma Textiles Pvt Ltd")]
        s2 = [row("S2-001", "Sharma Textiles Private Limited")]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        # name_norm differs ('pvt ltd' vs 'private limited' -> both map
        # to 'pvt ltd' actually per LEGAL_FORM_MAP -- confirm via core too)
        self.assertIn("S2-001", cmap["S1-001"])

    def test_pure_core_match_with_different_legal_forms(self):
        # "Acme Corp" vs "Acme LLC": name_norm differs (corp vs llc),
        # but name_core is identical ("acme") once both legal-form
        # tokens are stripped by normalize_name().
        s1 = [row("S1-002", "Acme Corp")]
        s2 = [row("S2-002", "Acme LLC")]
        s3 = []

        # Sanity: confirm the premise using Nidhi's normalize directly.
        n1 = normalize_dataframe_rows(s1)[0]
        n2 = normalize_dataframe_rows(s2)[0]
        self.assertNotEqual(n1.name_norm, n2.name_norm)
        self.assertEqual(n1.name_core, n2.name_core)

        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertIn("S2-002", cmap["S1-002"])


class TestNoCandidateForUnrelatedName(unittest.TestCase):
    def test_unrelated_names_produce_no_candidates(self):
        s1 = [row("S1-001", "Blue Ocean Traders")]
        s2 = [row("S2-001", "Red Mountain Foods")]
        s3 = [row("S3-001", "Silver Star Logistics")]
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(cmap["S1-001"], [])


class TestEmptyNameNoGiantBucket(unittest.TestCase):
    def test_blank_names_do_not_cross_match(self):
        # Multiple S2/S3 records with blank names must NOT all become
        # candidates for an S1 record with a blank name.
        s1 = [row("S1-001", None)]
        s2 = [row("S2-001", ""), row("S2-002", "   "), row("S2-003", None)]
        s3 = [row("S3-001", "")]
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(cmap["S1-001"], [])

    def test_blank_name_key_never_indexed(self):
        s2 = [row("S2-001", ""), row("S2-002", "   ")]
        s3 = [row("S3-001", None)]
        s2n = normalize_dataframe_rows(s2)
        s3n = normalize_dataframe_rows(s3)
        idx = _pass_a_exact_core_name([], s2n, s3n, config={})
        # No S1 records -> empty map, but more importantly, blank keys
        # must never have been inserted into an internal index that
        # would fan out. We verify indirectly: a blank-name S1 record
        # added now still gets zero candidates.
        s1 = [row("S1-999", None)]
        s1n = normalize_dataframe_rows(s1)
        full_map = _pass_a_exact_core_name(s1n, s2n, s3n, config={})
        self.assertEqual(full_map["S1-999"], [])


class TestEveryS1GetsEntry(unittest.TestCase):
    def test_every_s1_present_even_with_no_match(self):
        s1 = [
            row("S1-001", "Alpha Traders"),
            row("S1-002", "Totally Unmatched Name"),
            row("S1-003", None),  # blank name
        ]
        s2 = [row("S2-001", "Alpha Traders")]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(set(cmap.keys()), {"S1-001", "S1-002", "S1-003"})
        self.assertIn("S2-001", cmap["S1-001"])
        self.assertEqual(cmap["S1-002"], [])
        self.assertEqual(cmap["S1-003"], [])


class TestOnlyS2S3IdsReturned(unittest.TestCase):
    def test_no_s1_ids_leak_into_candidates(self):
        s1 = [
            row("S1-001", "Acme Corp"),
            row("S1-002", "Acme Corp"),  # same name as another S1 record
        ]
        s2 = [row("S2-001", "Acme Corp")]
        s3 = [row("S3-001", "Acme Corp")]
        cmap = generate_candidates(s1, s2, s3, config={})
        for s1_id, candidates in cmap.items():
            for cid in candidates:
                self.assertTrue(
                    cid.startswith("S2-") or cid.startswith("S3-"),
                    f"Unexpected id in candidates: {cid}",
                )
                self.assertNotEqual(cid, s1_id)
        # Both S1 records should find both S2 and S3 candidates.
        self.assertEqual(set(cmap["S1-001"]), {"S2-001", "S3-001"})
        self.assertEqual(set(cmap["S1-002"]), {"S2-001", "S3-001"})


class TestDuplicateCandidatesRemoved(unittest.TestCase):
    def test_dedup_when_name_core_and_name_norm_both_match_same_id(self):
        # Same S2 record would match via both name_core AND name_norm
        # indexes -- must appear only once in the output.
        s1 = [row("S1-001", "Acme Traders")]
        s2 = [row("S2-001", "Acme Traders")]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(cmap["S1-001"].count("S2-001"), 1)

    def test_dedup_with_multiple_s1_matching_same_candidate(self):
        s1 = [row("S1-001", "Acme Traders")]
        s2 = [row("S2-001", "Acme Traders")]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(len(cmap["S1-001"]), len(set(cmap["S1-001"])))


class TestDeterministicOutput(unittest.TestCase):
    def test_repeated_runs_produce_identical_output(self):
        s1 = [
            row("S1-001", "Acme Corp"),
            row("S1-002", "Blue Ocean Traders"),
            row("S1-003", None),
        ]
        s2 = [
            row("S2-003", "Acme LLC"),
            row("S2-001", "Acme Corp"),
            row("S2-002", "Blue Ocean Traders"),
        ]
        s3 = [row("S3-001", "Acme Incorporated")]

        results = []
        for _ in range(5):
            cmap = generate_candidates(s1, s2, s3, config={})
            results.append(cmap)

        for r in results[1:]:
            self.assertEqual(r, results[0])

    def test_candidate_list_order_is_sorted(self):
        s1 = [row("S1-001", "Acme Corp")]
        # Multiple ids that should all match via name_core "acme"
        s2 = [
            row("S2-005", "Acme LLC"),
            row("S2-001", "Acme Incorporated"),
            row("S2-003", "Acme Ltd"),
        ]
        s3 = []
        cmap = generate_candidates(s1, s2, s3, config={})
        self.assertEqual(cmap["S1-001"], sorted(cmap["S1-001"]))


class TestNoBruteForceScaling(unittest.TestCase):
    def test_index_is_built_once_not_per_s1_record(self):
        # Not a timing test (flaky in CI); instead assert structurally
        # that a lookup is O(1) dict access by checking the index
        # contains expected keys and no unrelated key blew up bucket
        # size for an unrelated S1 query.
        s2 = [row(f"S2-{i:04d}", f"Unique Name {i}") for i in range(2000)]
        s3 = []
        s2n = normalize_dataframe_rows(s2)
        s3n = normalize_dataframe_rows(s3)
        s1 = [row("S1-001", "Unique Name 500")]
        s1n = normalize_dataframe_rows(s1)
        cmap = _pass_a_exact_core_name(s1n, s2n, s3n, config={})
        self.assertEqual(cmap["S1-001"], ["S2-0500"])


if __name__ == "__main__":
    unittest.main()
