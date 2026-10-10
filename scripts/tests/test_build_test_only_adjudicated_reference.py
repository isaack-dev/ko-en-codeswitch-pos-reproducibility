"""
Unit tests (synthetic fixtures) + integration tests (real repo data) for
scripts/build_test_only_adjudicated_reference.py.

Run from repo root:
  python scripts/tests/test_build_test_only_adjudicated_reference.py
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import build_test_only_adjudicated_reference as m  # noqa: E402
from read_conllu import iter_sentences, F_UPOS  # noqa: E402

import openpyxl  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: pure-logic functions against synthetic fixtures
# ---------------------------------------------------------------------------

class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement_gives_kappa_one(self):
        pairs = [("NOUN", "NOUN"), ("VERB", "VERB"), ("ADJ", "ADJ")] * 10
        self.assertAlmostEqual(m.cohens_kappa(pairs), 1.0)

    def test_empty_pairs_returns_zero(self):
        self.assertEqual(m.cohens_kappa([]), 0.0)

    def test_chance_level_agreement_near_zero(self):
        pairs = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")] * 5
        self.assertAlmostEqual(m.cohens_kappa(pairs), 0.0, places=6)

    def test_known_disagreement_kappa_is_less_than_one(self):
        # Both labels appear on both sides so pe < po strictly (a fixture
        # where gold is always "NOUN" makes pe == po by construction, giving
        # a spurious kappa == 0 that doesn't test what this test is for).
        pairs = [("NOUN", "NOUN")] * 8 + [("VERB", "VERB")] + [("NOUN", "VERB")]
        k = m.cohens_kappa(pairs)
        self.assertLess(k, 1.0)
        self.assertGreater(k, 0.0)


class TestPerClassMetrics(unittest.TestCase):
    def test_precision_recall_f1_simple_case(self):
        pairs = [("NOUN", "NOUN"), ("NOUN", "VERB"), ("VERB", "VERB"), ("VERB", "VERB")]
        per_class = m.compute_per_class(pairs)
        self.assertAlmostEqual(per_class["NOUN"]["recall"], 0.5)
        self.assertAlmostEqual(per_class["NOUN"]["precision"], 1.0)
        self.assertAlmostEqual(per_class["VERB"]["recall"], 1.0)
        self.assertAlmostEqual(per_class["VERB"]["precision"], 2 / 3)

    def test_macro_and_weighted_f1_differ_with_imbalanced_support(self):
        pairs = [("NOUN", "NOUN")] * 9 + [("NOUN", "VERB")] + [("VERB", "ADJ")]
        per_class = m.compute_per_class(pairs)
        macro = m.macro_f1(per_class)
        weighted = m.weighted_f1(per_class)
        self.assertNotAlmostEqual(macro, weighted, places=3)

    def test_zero_support_class_excluded_from_macro(self):
        per_class = {"NOUN": {"f1": 0.8, "support": 5}, "PUNCT": {"f1": 0.0, "support": 0}}
        self.assertAlmostEqual(m.macro_f1(per_class), 0.8)


class TestDecisionClassification(unittest.TestCase):
    def test_p1_selected(self):
        self.assertEqual(m.classify_decision_source("ADJ", "ADJ", "NOUN"), "P1_selected")

    def test_p2_selected(self):
        self.assertEqual(m.classify_decision_source("NOUN", "ADJ", "NOUN"), "P2_selected")

    def test_third_label(self):
        self.assertEqual(m.classify_decision_source("VERB", "ADJ", "NOUN"), "third_label")

    def test_agreed(self):
        self.assertEqual(m.classify_decision_source("NOUN", "NOUN", "NOUN"), "agreed")


class TestSentenceBootstrap(unittest.TestCase):
    def test_deterministic_with_fixed_seed(self):
        correct = [8, 5, 10, 3, 7]
        total = [10, 10, 10, 10, 10]
        r1 = m.sentence_bootstrap_ci(correct, total, seed=42, n_boot=500)
        r2 = m.sentence_bootstrap_ci(correct, total, seed=42, n_boot=500)
        self.assertEqual(r1, r2)

    def test_all_correct_gives_ci_at_one(self):
        correct = [10] * 20
        total = [10] * 20
        lo, hi, mean = m.sentence_bootstrap_ci(correct, total, seed=42, n_boot=200)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)
        self.assertAlmostEqual(mean, 1.0)


class TestCheckExpected(unittest.TestCase):
    def test_no_mismatch_when_within_tolerance(self):
        report: list[str] = []
        m.check_expected({"agreement": 0.5000001}, {"agreement": 0.5}, "x", report)
        self.assertEqual(report, [])

    def test_mismatch_reported(self):
        report: list[str] = []
        m.check_expected({"agreement": 0.4}, {"agreement": 0.5}, "x", report)
        self.assertEqual(len(report), 1)
        self.assertIn("x.agreement", report[0])

    def test_nested_dict_checked_recursively(self):
        report: list[str] = []
        m.check_expected({"a": {"b": 5}}, {"a": {"b": 6}}, "root", report)
        self.assertEqual(len(report), 1)
        self.assertIn("root.a.b", report[0])


# ---------------------------------------------------------------------------
# Integration tests against the real, already-generated pipeline output
# ---------------------------------------------------------------------------

RESULTS_DIR = m.OUT_DIR


@unittest.skipUnless(m.P1_XLSX.exists() and m.P2_XLSX.exists(), "completed P1/P2 workbooks not present")
class TestIntegrationRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokens, cls.diagnostics = m.build_final_reference()
        cls.rep = [t for t in cls.tokens if t.set_name == "Representative"]
        cls.diag = [t for t in cls.tokens if t.set_name == "Diagnostic"]
        cls.pre_adj = {
            "Representative": m.compute_pre_adjudication_metrics(cls.rep),
            "Diagnostic": m.compute_pre_adjudication_metrics(cls.diag),
            "Combined": m.compute_pre_adjudication_metrics(cls.tokens),
        }
        cls.adj_summary = m.compute_adjudication_summary(cls.tokens)
        cls.human_canon = {
            "Representative": m.compute_human_canonical_metrics(cls.rep),
            "Diagnostic": m.compute_human_canonical_metrics(cls.diag),
            "Combined": m.compute_human_canonical_metrics(cls.tokens),
        }

    def test_total_token_count(self):
        self.assertEqual(len(self.tokens), 1879)

    def test_representative_and_diagnostic_token_counts(self):
        self.assertEqual(len(self.rep), 1063)
        self.assertEqual(len(self.diag), 816)

    def test_representative_and_diagnostic_sentence_counts(self):
        self.assertEqual(len(set(t.sample_id for t in self.rep)), 60)
        self.assertEqual(len(set(t.sample_id for t in self.diag)), 20)

    def test_no_duplicate_sample_id_token_id(self):
        keys = [(t.sample_id, t.token_id) for t in self.tokens]
        self.assertEqual(len(keys), len(set(keys)))

    def test_all_upos_valid(self):
        for t in self.tokens:
            self.assertIn(t.final_upos, m.UPOS17, msg=f"{t.sample_id}#{t.token_id}")
            self.assertIn(t.p1_upos, m.UPOS17)
            self.assertIn(t.p2_upos, m.UPOS17)

    def test_all_sentences_in_authoritative_test_conllu(self):
        conllu_sent_ids = set()
        for sent in iter_sentences(m.TEST_CONLLU):
            for c in sent["comments"]:
                if c.startswith("# sent_id = "):
                    conllu_sent_ids.add(c[len("# sent_id = "):])
        for t in self.tokens:
            self.assertIn(t.source_sent_id, conllu_sent_ids)

    def test_no_alignment_errors(self):
        self.assertEqual(self.diagnostics["alignment_errors"], [])

    def test_69_disagreements_all_resolved(self):
        decided = [t for t in self.tokens if t.decision_source in ("P1_selected", "P2_selected", "third_label")]
        self.assertEqual(len(decided), 69)
        for t in decided:
            self.assertTrue(t.final_upos)
            self.assertIn(t.final_upos, m.UPOS17)

    def test_da_post_audit_correction_incorporated(self):
        da = next(t for t in self.tokens if t.sample_id == "REP-TEST-055" and t.token_id == 5)
        self.assertEqual(da.token, "da")
        self.assertEqual(da.decision_source, "post_audit_correction")
        self.assertEqual(da.final_upos, "ADP")
        self.assertEqual(da.p1_upos, "PROPN")
        self.assertEqual(da.p2_upos, "PROPN")

    def test_p1_p2_source_workbooks_byte_unchanged_across_two_loads(self):
        h1 = m.sha256_of(m.P1_XLSX)
        h2 = m.sha256_of(m.P2_XLSX)
        m.load_p_workbook(m.P1_XLSX)
        m.load_p_workbook(m.P2_XLSX)
        self.assertEqual(m.sha256_of(m.P1_XLSX), h1)
        self.assertEqual(m.sha256_of(m.P2_XLSX), h2)

    def test_pre_adjudication_metrics_match_expected(self):
        report: list[str] = []
        m.check_expected(self.pre_adj, m.EXPECTED["pre_adjudication"], "pre_adjudication", report)
        self.assertEqual(report, [], msg="\n".join(report))

    def test_adjudication_summary_matches_expected(self):
        report: list[str] = []
        m.check_expected(self.adj_summary, m.EXPECTED["adjudication_summary"], "adjudication_summary", report)
        self.assertEqual(report, [], msg="\n".join(report))

    def test_human_canonical_metrics_match_expected(self):
        report: list[str] = []
        m.check_expected(self.human_canon, m.EXPECTED["human_canonical"], "human_canonical", report)
        self.assertEqual(report, [], msg="\n".join(report))

    def test_fixed_seed_rerun_reproduces_identical_tokens(self):
        tokens2, _ = m.build_final_reference()
        rows1 = [m.token_to_row(t) for t in self.tokens]
        rows2 = [m.token_to_row(t) for t in tokens2]
        self.assertEqual(rows1, rows2)

    def test_fixed_seed_bootstrap_reproducible(self):
        r1 = m.compute_human_canonical_metrics(self.rep)
        r2 = m.compute_human_canonical_metrics(self.rep)
        self.assertEqual(r1["bootstrap_ci_low"], r2["bootstrap_ci_low"])
        self.assertEqual(r1["bootstrap_ci_high"], r2["bootstrap_ci_high"])


@unittest.skipUnless((RESULTS_DIR / "final_human_reference_tokens.csv").exists(), "pipeline has not been run yet")
class TestGeneratedOutputsReopen(unittest.TestCase):
    def test_tokens_csv_reopens_with_1879_rows(self):
        with (RESULTS_DIR / "final_human_reference_tokens.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1879)
        for r in rows:
            self.assertIn(r["Final_Human_UPOS"], m.UPOS17)

    def test_sentences_jsonl_reopens_with_80_sentences(self):
        lines = (RESULTS_DIR / "final_human_reference_sentences.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 80)
        for line in lines:
            rec = json.loads(line)
            self.assertIn("tokens", rec)

    def test_final_conllu_reopens_and_has_final_upos_in_upos_field(self):
        sentences = list(iter_sentences(RESULTS_DIR / "final_human_reference.conllu"))
        self.assertEqual(len(sentences), 80)
        total_tokens = 0
        for sent in sentences:
            for tok in sent["tokens"]:
                self.assertIn(tok["fields"][F_UPOS], m.UPOS17)
                total_tokens += 1
        self.assertEqual(total_tokens, 1879)

    def test_results_workbook_reopens(self):
        wb = openpyxl.load_workbook(RESULTS_DIR / "Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx")
        self.assertIn("Summary", wb.sheetnames)
        self.assertIn("Final_Reference_1879", wb.sheetnames)

    def test_manifest_reopens_and_has_no_discrepancies(self):
        manifest = json.loads((RESULTS_DIR / "reference_build_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["expected_value_discrepancies"], [])
        self.assertEqual(manifest["seed"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
