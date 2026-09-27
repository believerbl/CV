"""
test_features.py
================
Unit tests for features.py (Parimarjan Work Package — Step 3).
Verifies the complete feature contract against the SRS specification:
    - Name lexical, structural, and edit similarities
    - Legal suffix invariance (name_core_exact)
    - Address token Jaccard and postal code matching
    - Mutually exclusive numeric evidence (perfect, partial, conflict, none)
    - Missingness handling (blank address, missing fields)
    - Open-set country agreement (France, India, US)
    - Script mismatch awareness (Latin vs Devanagari)
    - Schema consistency and determinism
"""

import unittest
from normalize import normalize_record
from features import (
    FEATURE_NAMES,
    build_pair_features,
    build_feature_vector,
    build_pair_features_batch,
)


class TestPairFeatures(unittest.TestCase):
    def test_schema_length(self):
        s1 = normalize_record("S1-1", "Acme Corp", "100 Market St", "US")
        s2 = normalize_record("S2-1", "Acme Corp", "100 Market St", "US")
        vec = build_feature_vector(s1, s2)
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertEqual(len(vec), 30)

    def test_identical_entity_pair(self):
        s1 = normalize_record("S1-1", "Acme Technologies Inc", "100 Market St Suite 5", "US")
        s2 = normalize_record("S2-1", "Acme Technologies Inc", "100 Market St Suite 5", "US")
        feats = build_pair_features(s1, s2)

        self.assertEqual(feats["name_core_exact"], 1.0)
        self.assertEqual(feats["name_norm_exact"], 1.0)
        self.assertEqual(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["name_char_edit_similarity"], 1.0)
        self.assertEqual(feats["address_norm_exact"], 1.0)
        self.assertEqual(feats["address_token_jaccard"], 1.0)
        self.assertEqual(feats["numeric_perfect"], 1.0)
        self.assertEqual(feats["numeric_conflict"], 0.0)
        self.assertEqual(feats["country_equal"], 1.0)
        self.assertEqual(feats["name_missing_s1"], 0.0)
        self.assertEqual(feats["address_missing_cand"], 0.0)

    def test_legal_suffix_difference(self):
        # S1 has 'Pvt Ltd', S2 has 'Corporation'
        s1 = normalize_record("S1-1", "Global Synergy Pvt Ltd", "500 MG Road", "India")
        s2 = normalize_record("S2-1", "Global Synergy Corporation", "500 MG Road", "India")
        feats = build_pair_features(s1, s2)

        self.assertEqual(feats["name_core_exact"], 1.0)  # Core name 'global synergy' matches!
        self.assertEqual(feats["name_norm_exact"], 0.0)  # Suffix differs
        self.assertGreaterEqual(feats["name_token_jaccard"], 0.4)

    def test_word_order_transposition(self):
        s1 = normalize_record("S1-1", "Solutions Apex Global", "100 Market St", "US")
        s2 = normalize_record("S2-1", "Apex Global Solutions", "100 Market St", "US")
        feats = build_pair_features(s1, s2)

        self.assertEqual(feats["name_token_jaccard"], 1.0)  # Same tokens!
        self.assertEqual(feats["name_first_token_match"], 0.0)  # Transposed first word
        self.assertGreater(feats["name_char_3gram_jaccard"], 0.7)

    def test_numeric_evidence_categories(self):
        # Case A: Perfect match (same numbers: 100, 5)
        s1 = normalize_record("S1-1", "Shop", "100 Market St #5", "US")
        s2_perf = normalize_record("S2-1", "Shop", "100 Market St Unit 5", "US")
        feats_perf = build_pair_features(s1, s2_perf)
        self.assertEqual(feats_perf["numeric_perfect"], 1.0)
        self.assertEqual(feats_perf["numeric_partial"], 0.0)
        self.assertEqual(feats_perf["numeric_conflict"], 0.0)
        self.assertEqual(feats_perf["numeric_none"], 0.0)

        # Case B: Partial match (shares 100, but s2 has extra 20)
        s2_part = normalize_record("S2-2", "Shop", "100 Market St Suite 20", "US")
        feats_part = build_pair_features(s1, s2_part)
        self.assertEqual(feats_part["numeric_perfect"], 0.0)
        self.assertEqual(feats_part["numeric_partial"], 1.0)
        self.assertEqual(feats_part["numeric_conflict"], 0.0)

        # Case C: Conflict (100 vs 200)
        s2_conf = normalize_record("S2-3", "Shop", "200 Market St", "US")
        feats_conf = build_pair_features(s1, s2_conf)
        self.assertEqual(feats_conf["numeric_conflict"], 1.0)
        self.assertEqual(feats_conf["numeric_perfect"], 0.0)
        self.assertEqual(feats_conf["numeric_partial"], 0.0)

        # Case D: None (no numbers in address)
        s2_none = normalize_record("S2-4", "Shop", "Market Street Near Central Park", "US")
        feats_none = build_pair_features(s1, s2_none)
        self.assertEqual(feats_none["numeric_none"], 1.0)
        self.assertEqual(feats_conf["numeric_perfect"], 0.0)

    def test_blank_address_handling(self):
        s1 = normalize_record("S1-1", "Omega Retail", "100 Main St", "US")
        s2 = normalize_record("S2-1", "Omega Retail", "", "US")  # Blank address in auxiliary source
        feats = build_pair_features(s1, s2)

        self.assertEqual(feats["address_missing_cand"], 1.0)
        self.assertEqual(feats["address_norm_exact"], 0.0)
        self.assertEqual(feats["address_token_jaccard"], 0.0)
        self.assertEqual(feats["address_char_edit_similarity"], 0.0)
        self.assertEqual(feats["numeric_none"], 1.0)

    def test_open_set_country_france(self):
        # France entity
        s1 = normalize_record("S1-1", "Boulangerie Paul", "12 Rue de Rivoli", "France")
        s2 = normalize_record("S2-1", "Boulangerie Paul", "12 Rue de Rivoli", "France")
        s3 = normalize_record("S3-1", "Boulangerie Paul", "12 Rue de Rivoli", "India")

        feats_fr = build_pair_features(s1, s2)
        feats_diff = build_pair_features(s1, s3)

        self.assertEqual(feats_fr["country_equal"], 1.0)
        self.assertEqual(feats_diff["country_equal"], 0.0)

    def test_script_mismatch(self):
        # Latin S1 vs Devanagari S2
        s1 = normalize_record("S1-1", "Ram Marketing", "Delhi", "India")
        s2 = normalize_record("S2-1", "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917", "Delhi", "India")
        feats = build_pair_features(s1, s2)

        self.assertEqual(feats["script_equal"], 0.0)
        self.assertEqual(feats["script_mismatch"], 1.0)

    def test_batch_feature_generation(self):
        s1 = normalize_record("S1-1", "Acme", "100 St", "US")
        s2_1 = normalize_record("S2-1", "Acme", "100 St", "US")
        s2_2 = normalize_record("S2-2", "Beta", "200 St", "US")

        cand_map = {"S1-1": ["S2-1", "S2-2"]}
        s1_lookup = {"S1-1": s1}
        cand_lookup = {"S2-1": s2_1, "S2-2": s2_2}

        X, pair_ids = build_pair_features_batch(cand_map, s1_lookup, cand_lookup)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(pair_ids), 2)
        self.assertEqual(pair_ids[0], ("S1-1", "S2-1"))
        self.assertEqual(pair_ids[1], ("S1-1", "S2-2"))

    def test_determinism(self):
        s1 = normalize_record("S1-1", "Solar Energy Tech", "77 Broadway Blvd", "US")
        s2 = normalize_record("S2-1", "Solar Energy Technologies LLC", "77 Broadway Boulevard", "US")

        run1 = build_feature_vector(s1, s2)
        run2 = build_feature_vector(s1, s2)
        self.assertEqual(run1, run2)


if __name__ == "__main__":
    unittest.main()
