"""
Unit tests (synthetic fixtures) + integration tests (real repo data) for
scripts/evaluate_models_on_test_only_human_reference.py.

Run from repo root:
  python scripts/tests/test_evaluate_models_on_test_only_human_reference.py
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import evaluate_models_on_test_only_human_reference as m  # noqa: E402

import openpyxl  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: pure-logic functions against synthetic fixtures
# ---------------------------------------------------------------------------

class TestGetPredUpos(unittest.TestCase):
    def test_extracts_pred_upos(self):
        self.assertEqual(m.get_pred_upos("Lang=KO|PredUPOS=NOUN"), "NOUN")

    def test_returns_none_when_absent(self):
        self.assertIsNone(m.get_pred_upos("Lang=KO"))

    def test_handles_underscore_misc(self):
        self.assertIsNone(m.get_pred_upos("_"))


class TestComputeModelMetrics(unittest.TestCase):
    def _make_rows(self):
        return [
            {"Sample_ID": "S1", "Token_ID": 1, "Set": "Representative", "Final_Human_UPOS": "NOUN",
             "Pred_UPOS": "NOUN", "Final_Tokenization_Issue": "No", "Lang": "KO", "Is_Boundary": False},
            {"Sample_ID": "S1", "Token_ID": 2, "Set": "Representative", "Final_Human_UPOS": "VERB",
             "Pred_UPOS": "NOUN", "Final_Tokenization_Issue": "No", "Lang": "EN", "Is_Boundary": True},
            {"Sample_ID": "S2", "Token_ID": 1, "Set": "Representative", "Final_Human_UPOS": "PUNCT",
             "Pred_UPOS": "PUNCT", "Final_Tokenization_Issue": "Yes", "Lang": "OTHER", "Is_Boundary": False},
        ]

    def test_accuracy_and_non_punct_accuracy(self):
        rows = self._make_rows()
        result = m.compute_model_metrics(rows)
        self.assertEqual(result["n_tokens"], 3)
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["non_punct_tokens"], 2)
        self.assertEqual(result["non_punct_correct"], 1)

    def test_tokenization_stratum_split(self):
        rows = self._make_rows()
        result = m.compute_model_metrics(rows)
        self.assertEqual(result["tokenization_issue_yes"][1], 1)
        self.assertEqual(result["tokenization_issue_no"][1], 2)

    def test_language_and_boundary_strata(self):
        rows = self._make_rows()
        result = m.compute_model_metrics(rows)
        self.assertEqual(result["ko_accuracy"][1], 1)
        self.assertEqual(result["en_accuracy"][1], 1)
        self.assertEqual(result["boundary_accuracy"][1], 1)
        self.assertEqual(result["non_boundary_accuracy"][1], 2)

    def test_by_key_lookup_matches_correctness(self):
        rows = self._make_rows()
        result = m.compute_model_metrics(rows)
        self.assertTrue(result["_by_key"][("S1", 1)])
        self.assertFalse(result["_by_key"][("S1", 2)])


class TestRankComparison(unittest.TestCase):
    def test_rank_and_score_change_computed(self):
        silver_acc = {"a": 0.99, "b": 0.60}
        rep_metrics = {"a": {"accuracy": 0.60}, "b": {"accuracy": 0.65}}
        available = ["a", "b"]
        silver_ranked = sorted(available, key=lambda k: -silver_acc[k])
        human_ranked = sorted(available, key=lambda k: -rep_metrics[k]["accuracy"])
        self.assertEqual(silver_ranked, ["a", "b"])
        self.assertEqual(human_ranked, ["b", "a"])


class TestPairwiseMcNemar(unittest.TestCase):
    def test_identical_models_have_zero_discordant(self):
        by_key = {("S1", 1): True, ("S1", 2): False, ("S2", 1): True}
        rep_metrics = {
            "a": {"_by_key": dict(by_key), "correct_by_sentence": [1, 1], "total_by_sentence": [2, 1]},
            "b": {"_by_key": dict(by_key), "correct_by_sentence": [1, 1], "total_by_sentence": [2, 1]},
        }
        orig_order, orig_models = m.MODEL_ORDER, m.MODELS
        try:
            m.MODEL_ORDER = ["a", "b"]
            m.MODELS = {"a": {"display": "A"}, "b": {"display": "B"}}
            mcnemar_rows, boot_rows = m.pairwise_mcnemar_and_bootstrap(rep_metrics)
        finally:
            m.MODEL_ORDER = orig_order
            m.MODELS = orig_models
        self.assertEqual(len(mcnemar_rows), 1)
        self.assertEqual(mcnemar_rows[0]["discordant"], 0)
        self.assertAlmostEqual(mcnemar_rows[0]["mcnemar_exact_p"], 1.0)


# ---------------------------------------------------------------------------
# Integration tests against the real pipeline output
# ---------------------------------------------------------------------------

RESULTS_DIR = m.OUT_DIR


@unittest.skipUnless(m.HUMAN_REF_CSV.exists(), "Phase 2 human reference not built yet")
class TestIntegrationRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.human_rows = m.load_human_reference()
        cls.lang_boundary = m.load_lang_and_boundary()

    def test_human_reference_has_1879_rows(self):
        self.assertEqual(len(self.human_rows), 1879)

    def test_all_models_align_to_1879_tokens(self):
        for k in m.MODEL_ORDER:
            with self.subTest(model=k):
                aligned, errors = m.align_model(k, self.human_rows, self.lang_boundary)
                self.assertEqual(errors, [], msg=f"{k}: {errors[:5]}")
                self.assertEqual(len(aligned), 1879)

    def test_ud_transfer_no_mapping_uses_corrected_not_prefix_predictions(self):
        cfg = m.MODELS["ud_transfer_no_mapping"]
        self.assertIn("corrected_ud_transfer", str(cfg["pred_path"]))
        self.assertTrue(cfg["pred_path"].exists())

    def test_silver_accuracy_loads_for_all_models(self):
        for k, cfg in m.MODELS.items():
            with self.subTest(model=k):
                acc = m.load_silver_test_accuracy(cfg["silver_csv"], cfg["silver_key_col"], cfg["silver_key"])
                self.assertGreater(acc, 0.0)
                self.assertLessEqual(acc, 1.0)

    def test_representative_and_diagnostic_split_sizes(self):
        aligned, _ = m.align_model("joint_perceptron", self.human_rows, self.lang_boundary)
        rep = [r for r in aligned if r["Set"] == "Representative"]
        diag = [r for r in aligned if r["Set"] == "Diagnostic"]
        self.assertEqual(len(rep), 1063)
        self.assertEqual(len(diag), 816)


@unittest.skipUnless((RESULTS_DIR / "model_metrics.json").exists(), "pipeline has not been run yet")
class TestGeneratedOutputsReopen(unittest.TestCase):
    def test_aligned_predictions_csv_has_1879_rows_and_all_model_columns(self):
        with (RESULTS_DIR / "aligned_token_predictions.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1879)
        for k in m.MODEL_ORDER:
            self.assertIn(f"Pred_{k}", rows[0])
            self.assertIn(f"Correct_{k}", rows[0])

    def test_metrics_csvs_have_one_row_per_model(self):
        for fname in ("model_metrics_representative.csv", "model_metrics_diagnostic.csv"):
            with (RESULTS_DIR / fname).open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(m.MODELS))

    def test_representative_token_count_is_1063_in_metrics(self):
        with (RESULTS_DIR / "model_metrics_representative.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            self.assertEqual(int(r["n_tokens"]), 1063)

    def test_diagnostic_token_count_is_816_in_metrics(self):
        with (RESULTS_DIR / "model_metrics_diagnostic.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            self.assertEqual(int(r["n_tokens"]), 816)

    def test_workbook_reopens_with_required_sheets(self):
        wb = openpyxl.load_workbook(RESULTS_DIR / "Human_POS_Tagging_test_v1_Model_Evaluation.xlsx")
        required = {
            "Executive_Summary", "Representative_Metrics", "Diagnostic_Metrics", "Silver_vs_Human_Ranks",
            "Pairwise_McNemar", "Bootstrap_Differences", "Per_UPOS", "Diagnostic_Categories",
            "Tokenization_Strata", "Confusions", "Aligned_Predictions",
        }
        self.assertTrue(required.issubset(set(wb.sheetnames)), msg=set(wb.sheetnames))

    def test_manifest_reopens_and_all_models_present(self):
        manifest = json.loads((RESULTS_DIR / "model_evaluation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["models"].keys()), set(m.MODEL_ORDER))
        self.assertEqual(manifest["seed"], 42)

    def test_no_nan_or_inf_in_model_metrics_json(self):
        text = (RESULTS_DIR / "model_metrics.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_mcnemar_csv_has_28_pairs_for_8_models(self):
        with (RESULTS_DIR / "representative_pairwise_exact_mcnemar.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        n = len(m.MODELS)
        self.assertEqual(len(rows), n * (n - 1) // 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
