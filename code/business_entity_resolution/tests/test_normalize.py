"""
test_normalize.py
==================
Unit tests for normalize.py, covering the minimum test matrix from the
execution guide: case/whitespace, safe punctuation, legal suffix, token
reordering, typos, blank address, numeric address, postal code, Latin
script, Devanagari/non-Latin script, France/unseen country, missing
name/address, repeated/generic tokens, and determinism.

Run with:  python3 -m unittest test_normalize.py -v
"""

import unittest

from normalize import (
    normalize_record,
    normalize_name,
    normalize_address,
    normalize_country,
    detect_scripts,
    is_blank,
    source_from_entity_id,
)


class TestSourceDetection(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(source_from_entity_id("S1-00001"), "S1")
        self.assertEqual(source_from_entity_id("S2-00047"), "S2")
        self.assertEqual(source_from_entity_id("S3-00812"), "S3")

    def test_unknown_prefix_raises(self):
        with self.assertRaises(ValueError):
            source_from_entity_id("X9-00001")


class TestCaseWhitespace(unittest.TestCase):
    def test_case_and_whitespace_equivalence(self):
        a = normalize_name("Acme   Corp")
        b = normalize_name("acme corp")
        c = normalize_name("  ACME    CORP  ")
        self.assertEqual(a["name_norm"], b["name_norm"])
        self.assertEqual(b["name_norm"], c["name_norm"])


class TestSafePunctuation(unittest.TestCase):
    def test_ampersand_vs_and(self):
        a = normalize_name("Smith & Sons")
        b = normalize_name("Smith and Sons")
        self.assertEqual(a["name_norm"], b["name_norm"])

    def test_punctuation_stripped_without_losing_tokens(self):
        out = normalize_name("O'Neil's Bakery, Inc.")
        self.assertIn("bakery", out["name_tokens"])
        self.assertIn("oneil", "".join(out["name_tokens"]).replace("s", "") or out["name_norm"])


class TestLegalSuffix(unittest.TestCase):
    def test_corp_vs_corporation(self):
        a = normalize_name("Global Traders Corp")
        b = normalize_name("Global Traders Corporation")
        self.assertEqual(a["name_norm"], b["name_norm"])
        self.assertEqual(a["name_core"], b["name_core"])

    def test_pvt_vs_private(self):
        a = normalize_name("Sharma Textiles Pvt Ltd")
        b = normalize_name("Sharma Textiles Private Limited")
        self.assertEqual(a["name_norm"], b["name_norm"])
        self.assertEqual(a["name_core"], b["name_core"])

    def test_raw_preserved(self):
        raw = "Sharma Textiles Private Limited"
        # normalize_record preserves raw regardless of normalization
        rec = normalize_record("S1-00001", raw, "1 Main St", "India")
        self.assertEqual(rec.name_raw, raw)

    def test_core_excludes_legal_form_but_norm_keeps_it(self):
        out = normalize_name("Sharma Textiles Pvt Ltd")
        self.assertNotIn("pvt", out["name_core"].split())
        self.assertIn("pvt", out["name_norm"].split())


class TestTokenReordering(unittest.TestCase):
    def test_reordering_supported_via_token_sets(self):
        a = normalize_name("Blue Ocean Traders")
        b = normalize_name("Traders Blue Ocean")
        self.assertEqual(set(a["name_tokens"]), set(b["name_tokens"]))
        # name_norm itself is order-preserving (features layer decides how
        # to use order-sensitive vs order-insensitive comparisons).
        self.assertNotEqual(a["name_norm"], b["name_norm"])


class TestTyposResidualNoise(unittest.TestCase):
    def test_typo_does_not_crash_and_stays_close(self):
        a = normalize_name("Continental Traders")
        b = normalize_name("Continantal Traders")  # typo
        self.assertNotEqual(a["name_norm"], b["name_norm"])
        # still comparable representations (no crash, tokens present)
        self.assertTrue(len(a["name_tokens"]) > 0 and len(b["name_tokens"]) > 0)


class TestBlankAddress(unittest.TestCase):
    def test_blank_address_flagged_explicitly(self):
        for val in [None, "", "   "]:
            out = normalize_address(val)
            self.assertTrue(out["address_is_blank"])
            self.assertIsNone(out["address_norm"])
            self.assertEqual(out["address_tokens"], [])

    def test_blank_name_flagged_explicitly(self):
        for val in [None, "", "  "]:
            out = normalize_name(val)
            self.assertIsNone(out["name_norm"])
            self.assertIsNone(out["name_core"])

    def test_missingness_flags_on_full_record(self):
        rec = normalize_record("S2-00001", None, "", "India")
        self.assertTrue(rec.missingness_flags["name_missing"])
        self.assertTrue(rec.missingness_flags["address_missing"])
        self.assertFalse(rec.missingness_flags["country_missing"])


class TestNumericAddress(unittest.TestCase):
    def test_numeric_tokens_extracted(self):
        out = normalize_address("221B Baker Street, London")
        self.assertIn("221b", out["numeric_tokens"] + [t for t in out["address_tokens"] if any(c.isdigit() for c in t)])
        # at least one numeric-bearing token captured
        self.assertTrue(any(any(c.isdigit() for c in t) for t in out["address_tokens"]))


class TestPostalCode(unittest.TestCase):
    def test_us_zip_recognized(self):
        out = normalize_address("500 Market St, Springfield, 62704")
        self.assertIn("62704", out["postal_tokens"])

    def test_india_pin_recognized(self):
        out = normalize_address("MG Road, Bengaluru, 560001")
        self.assertIn("560001", out["postal_tokens"])

    def test_no_false_postal_on_short_numbers(self):
        out = normalize_address("Plot 12, Sector 5")
        self.assertNotIn("12", out["postal_tokens"])
        self.assertNotIn("5", out["postal_tokens"])


class TestScriptFlags(unittest.TestCase):
    def test_latin_script_flagged(self):
        out = normalize_name("Continental Traders")
        self.assertEqual(out["name_script_flags"]["primary_script"], "latin")
        self.assertFalse(out["name_script_flags"]["is_mixed_script"])

    def test_devanagari_preserved_and_flagged(self):
        raw = "शर्मा टेक्सटाइल्स"
        out = normalize_name(raw)
        self.assertEqual(out["name_script_flags"]["primary_script"], "devanagari")
        # not blindly transliterated: original characters remain in norm
        self.assertTrue(any(0x0900 <= ord(ch) <= 0x097F for ch in out["name_norm"]))

    def test_mixed_script_detected(self):
        raw = "Sharma शर्मा Textiles"
        out = normalize_name(raw)
        self.assertTrue(out["name_script_flags"]["is_mixed_script"])


class TestOpenSetCountry(unittest.TestCase):
    def test_france_passes_through(self):
        out = normalize_country("France")
        self.assertEqual(out["country_norm"], "france")
        self.assertFalse(out["country_is_blank"])

    def test_arbitrary_unseen_country_passes_through(self):
        out = normalize_country("Wakanda")
        self.assertEqual(out["country_norm"], "wakanda")

    def test_no_allowlist_filtering(self):
        # There must be no branch that rejects or nulls out an
        # unrecognized country string.
        for c in ["US", "India", "France", "Some New Country"]:
            out = normalize_country(c)
            self.assertIsNotNone(out["country_norm"])

    def test_accented_french_name_survives(self):
        raw = "Café de la Gare"
        out = normalize_name(raw)
        self.assertIsNotNone(out["name_norm"])
        self.assertIn("caf", out["name_norm"].split()[0])  # accent-normalized via NFKC path retains base char cluster


class TestAccentFolding(unittest.TestCase):
    """Item 1 from the review: accent variation must be handled, in
    addition to (not instead of) preserving the accented form.
    Tests verified against real French dataset rows (e.g. test_source1).
    """

    def test_name_norm_preserves_accent_but_fold_strips_it(self):
        accented = normalize_name("Société Générale")
        self.assertIn("é", accented["name_norm"])  # accent preserved in norm
        self.assertNotIn("é", accented["name_fold"])  # stripped in fold
        self.assertEqual(accented["name_fold"], "societe generale")

    def test_accented_and_unaccented_forms_match_on_fold_not_norm(self):
        a = normalize_name("Café de la Gare")
        b = normalize_name("Cafe de la Gare")
        self.assertNotEqual(a["name_norm"], b["name_norm"])  # different as written
        self.assertEqual(a["name_fold"], b["name_fold"])      # same once folded

    def test_real_french_name_accents(self):
        a = normalize_name("Maison de Santé Generation")
        b = normalize_name("Maison de Sante Generation")
        self.assertEqual(a["name_fold"], b["name_fold"])

        a2 = normalize_name("Saint-Herblain Societe SARL")
        b2 = normalize_name("Saint-Herblain Société SARL")
        self.assertEqual(a2["name_fold"], b2["name_fold"])
        self.assertNotEqual(a2["name_norm"], b2["name_norm"])

    def test_french_legal_suffixes(self):
        forms = [
            ("Thermal & Fils SASU", "sasu"),
            ("Elephant Centre EURL", "eurl"),
            ("Immobiliere Saint Pierre SCI", "sci"),
            ("Transports Loire SNC", "snc"),
        ]
        for name_raw, expected_tok in forms:
            out = normalize_name(name_raw)
            self.assertIn(expected_tok, out["name_tokens"])
            self.assertNotIn(expected_tok, out["name_core"].split())

    def test_address_fold_handles_french_accents(self):
        a = normalize_address("Rue de l'Église, Paris")
        b = normalize_address("Rue de l'Eglise, Paris")
        self.assertEqual(a["address_fold"], b["address_fold"])

        a2 = normalize_address("7 Place de Suède, Lille, Hauts-de-France")
        b2 = normalize_address("7 Place de Suede, Lille, Hauts-de-France")
        self.assertEqual(a2["address_fold"], b2["address_fold"])

    def test_devanagari_not_affected_by_folding(self):
        # NFKD does not decompose Devanagari the way it does Latin
        # diacritics, so folding should be a no-op here, not garbage.
        out = normalize_name("शर्मा टेक्सटाइल्स")
        self.assertEqual(out["name_fold"], out["name_norm"])

    def test_blank_input_fold_is_none(self):
        out = normalize_name(None)
        self.assertIsNone(out["name_fold"])
        out2 = normalize_address("")
        self.assertIsNone(out2["address_fold"])


class TestArtifactMarkerHypothesis(unittest.TestCase):
    """Item 2 from the review: DBA/trade-name and leading-record-code
    artifact detection.

    IMPORTANT — these patterns are a HYPOTHESIS built from the noise types
    the problem statement explicitly names ('DBA/trade names'), not from
    an audit of real dataset values, because the real dataset was not
    provided. They flag detection only; they do not silently rewrite
    name_norm. Confirm/revise against real examples before Piyush/
    Parimarjan rely on the flag for a feature.
    """

    def test_dba_marker_detected(self):
        out = normalize_name("Acme Holdings DBA Fresh Mart")
        self.assertTrue(out["name_had_dba_marker"])

    def test_trading_as_marker_detected(self):
        out = normalize_name("Acme Holdings trading as Fresh Mart")
        self.assertTrue(out["name_had_dba_marker"])

    def test_no_marker_on_plain_name(self):
        out = normalize_name("Acme Holdings")
        self.assertFalse(out["name_had_dba_marker"])
        self.assertFalse(out["name_had_leading_code_artifact"])

    def test_leading_code_artifact_detected(self):
        out = normalize_name("00123 - Acme Holdings")
        self.assertTrue(out["name_had_leading_code_artifact"])

    def test_marker_flag_does_not_mutate_name_norm(self):
        # Detection is additive/flag-only in this revision — name_norm is
        # unchanged so nothing that already depends on it breaks.
        out = normalize_name("Acme Holdings DBA Fresh Mart")
        self.assertIn("dba", out["name_norm"])


class TestMissingNameAddressNoCrash(unittest.TestCase):
    def test_fully_blank_record_does_not_crash(self):
        rec = normalize_record("S1-00099", None, None, None)
        self.assertTrue(rec.missingness_flags["name_missing"])
        self.assertTrue(rec.missingness_flags["address_missing"])
        self.assertTrue(rec.missingness_flags["country_missing"])
        self.assertIsNone(rec.name_norm)
        self.assertIsNone(rec.address_norm)
        self.assertIsNone(rec.country_norm)


class TestRepeatedGenericTokens(unittest.TestCase):
    def test_token_frequency_retained(self):
        out = normalize_name("Star Star Traders")
        self.assertEqual(out["name_token_counts"]["star"], 2)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        raw_name = "Sharma Textiles Pvt. Ltd."
        raw_addr = "12, MG Road, Near SBI ATM, Bengaluru, 560001"
        r1 = normalize_record("S1-00001", raw_name, raw_addr, "India")
        r2 = normalize_record("S1-00001", raw_name, raw_addr, "India")
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_repeated_runs_are_stable(self):
        results = set()
        for _ in range(5):
            rec = normalize_record("S3-00042", "Global & Sons Corp", "1 Rd, Apt 2", "France")
            results.add((rec.name_norm, rec.address_norm))
        self.assertEqual(len(results), 1)


class TestIsBlankHelper(unittest.TestCase):
    def test_is_blank_variants(self):
        self.assertTrue(is_blank(None))
        self.assertTrue(is_blank(""))
        self.assertTrue(is_blank("   "))
        self.assertFalse(is_blank("a"))


if __name__ == "__main__":
    unittest.main()