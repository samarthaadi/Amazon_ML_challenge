"""Small checks for the scoring and numeric diagnostics used in the report."""
import unittest

import numpy as np

from eda_first_pass import clean, expanded, clean_targeted, expanded_targeted, number_status, paired_cos, trigram_matrix, TARGETED_NORMALIZATION
from full_pool_baseline import pair_fbeta, score_references, cleanup_row_audit, feature_row
import pandas as pd


class DiagnosticTests(unittest.TestCase):
    def test_unicode_and_punctuation(self):
        self.assertEqual(clean("  ＡＣＭＥ, Inc. "), "acme inc")
        self.assertEqual(clean("École"), "école")
        self.assertEqual(clean("राम"), "राम")

    def test_legal_abbreviations(self):
        self.assertEqual(expanded("Acme Pvt. Ltd."), expanded("ACME PRIVATE LIMITED"))

    def test_numeric_categories(self):
        self.assertEqual(number_status("Main Road", "Main Rd"), "neither_has_numbers")
        self.assertEqual(number_status("12 Main", "Main"), "one_side_has_no_numbers")
        self.assertEqual(number_status("12 Main, unit 3", "3 / 12 Main"), "same_numeric_tokens")
        self.assertEqual(number_status("101 Main", "1-101 Main"), "partial_overlap")
        self.assertEqual(number_status("101 Main", "202 Main"), "disjoint_numeric_tokens")
        self.assertEqual(number_status("101 Main", "0101 Main", strip_zeroes=True), "same_numeric_tokens")

    def test_count_cosine_and_empty(self):
        matrix = trigram_matrix(["abcd", "abce", "abcd", "", "wxyz"])
        result = paired_cos(matrix, np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4]))
        np.testing.assert_allclose(result, [0.5, 1.0, 0.0, 0.0], atol=1e-6)

    def test_entity_f0_5_including_singletons(self):
        self.assertEqual(pair_fbeta(set(), set()), 1.0)
        self.assertEqual(pair_fbeta(set(), {"false"}), 0.0)
        self.assertEqual(pair_fbeta({"a", "b"}, set()), 0.0)
        self.assertAlmostEqual(pair_fbeta({"a", "b"}, {"a"}), 1.25 / 1.5)
        truth = {"matched": {"a", "b"}, "singleton": set()}
        predictions = {"matched": {"a"}, "singleton": set()}
        self.assertAlmostEqual(score_references(["matched", "singleton"], truth, predictions), (1.25 / 1.5 + 1) / 2)

    def test_targeted_scripts_and_symbols(self):
        self.assertEqual(clean_targeted("  ÉCOLE ★ राम தமிழ் বাংলা 中文 ＡＣＭＥ  "), "école राम தமிழ் বাংলা 中文 acme")
        self.assertEqual(clean_targeted("Cafe\u0301"), "café")
        self.assertEqual(clean_targeted("North@South_Inc"), "north south inc")

    def test_targeted_numeric_structure(self):
        values = ["123", "12-3", "12/3", "12.3", "12,3", "12:3", "12 3"]
        self.assertEqual(len({clean_targeted(v) for v in values}), len(values))
        self.assertEqual(clean_targeted("0012-003 B007/004A 000"), "12-3 b7/4a 0")
        self.assertEqual(clean_targeted("0012.003 000,005 000.000"), "12.003 0,005 0.000")
        self.assertEqual(clean_targeted("१२–०३ 12 / 03 0012 − 003"), "१२-३ 12/3 12-3")

    def test_targeted_websites(self):
        self.assertEqual(clean_targeted("Acme (https://www.acme.com/about?a=1&b=2)", field="name"), "acme")
        self.assertEqual(clean_targeted("www.aarvyasupply.com", field="name"), "aarvyasupply")
        self.assertEqual(clean_targeted("HTTPS://shop.acme.co.in/products", field="name"), "acme")
        self.assertEqual(clean_targeted("École www.école.fr", field="name"), "école")
        self.assertEqual(clean_targeted("Acme example.com", field="name"), "acme")
        self.assertEqual(clean_targeted("A.B. Company", field="name"), "a b company")
        self.assertEqual(expanded_targeted("Acme & Co https://acme.com?q=1&b=2"), "acme and company")

    def test_targeted_legal_and_empty(self):
        self.assertEqual(expanded_targeted("ACME L.L.C."), "acme llc")
        self.assertEqual(expanded_targeted("Acme P.L.L.C. Pvt. Ltd."), "acme pllc private limited")
        self.assertEqual(clean_targeted(""), "")
        self.assertEqual(clean_targeted("★ ()", field="name"), "")

    def test_targeted_idempotence(self):
        for text in ["राम", "ÉCOLE", "000", "0012-003", "000.005", "12 / 03", "Acme L.L.C.", "www.acme.com", "Acme & Co", "ABC (https://acme.com)"]:
            for field in ("name", "address"):
                result = clean_targeted(text, field=field)
                self.assertEqual(clean_targeted(result, field=field), result)
            self.assertEqual(expanded_targeted(expanded_targeted(text)), expanded_targeted(text))

    def test_rule_counts_use_unique_records(self):
        anchors = pd.DataFrame([{"entity_id": "S1-a", "business_name": "ACME www.acme.com", "business_address": "0012-003"}])
        candidates = pd.DataFrame([{"candidate_entity_id": "S2-a", "candidate_name": "acme", "candidate_address": "12-3"}] * 2)
        counts, totals, changed = cleanup_row_audit(anchors, candidates)
        self.assertEqual({r["population"]: r["rows"] for r in totals}, {"sampled_references": 1, "retrieved_candidate_records": 1})
        website = next(r for r in counts if r["population"] == "sampled_references" and r["field"] == "name" and r["rule"] == "website")
        self.assertEqual(website["changed_rows"], 1)
        self.assertEqual(website["rate"], 1.0)
        self.assertEqual(changed, {"S1-a", "S2-a"})

    def test_feature_version_is_explicit(self):
        reference = {"business_name": "Acme LLC", "business_address": "0012-003"}
        candidate = {"business_name": "Acme L.L.C. www.acme.com", "business_address": "12-3"}
        old = feature_row(reference, candidate)
        new = feature_row(reference, candidate, normalization=TARGETED_NORMALIZATION)
        self.assertLess(old["expanded_name_similarity"], 1)
        self.assertAlmostEqual(new["expanded_name_similarity"], 1)
        self.assertAlmostEqual(new["address_similarity"], 1)
        self.assertEqual(feature_row(reference, candidate), old)

    def test_ablation_rules_are_isolated(self):
        from eda_first_pass import CLEANUP_ABLATIONS
        raw = "Acme L.L.C. www.acme.com 0012-003"
        expected = {
            "url_only": "acme l l c 0012 003",
            "dotted_suffix_only": "acme llc www acme com 0012 003",
            "leading_zero_only": "acme l l c www acme com 12 3",
            "numeric_separator_only": "acme l l c www acme com 0012-003",
        }
        for variant, rules in CLEANUP_ABLATIONS.items():
            self.assertEqual(clean_targeted(raw, field="name", rules=rules), expected[variant])
        self.assertEqual(clean_targeted("000.005", rules={"leading_zero"}), "0 005")
        for text in [raw, "Acme & Co", "राम", "ÉCOLE", "0012.003", "", "www.acme.com"]:
            self.assertEqual(clean_targeted(text, rules=set()), clean(text))
            self.assertEqual(expanded_targeted(text, rules=set()), expanded(text))


if __name__ == "__main__":
    unittest.main()
