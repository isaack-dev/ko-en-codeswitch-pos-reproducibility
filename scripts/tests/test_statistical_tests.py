"""
test_statistical_tests.py - Unit tests for scripts/statistical_tests.py.

Uses only the standard library (unittest), since pytest is not installed in
this environment. Run from repo root:

  python scripts/tests/test_statistical_tests.py

The exact McNemar implementation is cross-checked against an independent
reference implementation (_independent_exact_binomial_two_sided below) that
uses exact rational arithmetic (fractions.Fraction + math.comb) rather than
the log-space (lgamma) approach used in statistical_tests.exact_mcnemar_pvalue,
so the two do not share a code path.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import statistical_tests as st  # noqa: E402


# ---------------------------------------------------------------------------
# Independent reference implementation for exact McNemar (rational arithmetic)
# ---------------------------------------------------------------------------

def _independent_exact_binomial_two_sided(a_only: int, b_only: int) -> float:
    """Reference two-sided exact McNemar p-value using exact Fraction math.

    Deliberately does not share any code with st.exact_mcnemar_pvalue
    (which works in log-space via math.lgamma). Only used to independently
    verify correctness on small/moderate examples where math.comb(n, i) is
    still cheap to compute exactly.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    denom = Fraction(2) ** n
    tail = sum(Fraction(math.comb(n, i)) for i in range(k + 1))
    p = 2 * tail / denom
    return float(min(p, Fraction(1)))


class TestExactMcNemar(unittest.TestCase):
    def test_zero_discordant_pairs_returns_one(self):
        self.assertEqual(st.exact_mcnemar_pvalue(0, 0), 1.0)

    def test_symmetric_in_a_and_b(self):
        self.assertAlmostEqual(
            st.exact_mcnemar_pvalue(10, 20), st.exact_mcnemar_pvalue(20, 10), places=12
        )

    def test_hand_computed_single_sided_example(self):
        # a_only=0, b_only=7: p = 2 * C(7,0) * 0.5^7 = 2/128 = 0.015625
        self.assertAlmostEqual(st.exact_mcnemar_pvalue(0, 7), 0.015625, places=12)

    def test_a_equals_b_always_saturates_to_one(self):
        # When a_only == b_only, k = n/2 and P(X <= n/2) >= 0.5 by symmetry,
        # so the two-sided p-value always caps at 1.0.
        for k in (1, 5, 50, 500):
            self.assertEqual(st.exact_mcnemar_pvalue(k, k), 1.0)

    def test_matches_independent_reference_small_examples(self):
        cases = [(10, 20), (45, 55), (5, 95), (1, 1), (0, 1), (3, 17), (33, 67)]
        for a_only, b_only in cases:
            with self.subTest(a_only=a_only, b_only=b_only):
                expected = _independent_exact_binomial_two_sided(a_only, b_only)
                actual = st.exact_mcnemar_pvalue(a_only, b_only)
                self.assertAlmostEqual(actual, expected, places=9)

    def test_matches_independent_reference_moderate_n(self):
        # n=300 discordant pairs: still cheap for exact Fraction arithmetic,
        # large enough to exercise the log-space summation path meaningfully.
        for a_only, b_only in [(120, 180), (100, 200), (10, 290)]:
            with self.subTest(a_only=a_only, b_only=b_only):
                expected = _independent_exact_binomial_two_sided(a_only, b_only)
                actual = st.exact_mcnemar_pvalue(a_only, b_only)
                self.assertAlmostEqual(actual, expected, places=9)

    def test_matches_scipy_if_available(self):
        try:
            from scipy.stats import binomtest
        except ImportError:
            self.skipTest("scipy not installed in this environment")
        for a_only, b_only in [(10, 20), (5, 95), (0, 7)]:
            n = a_only + b_only
            k = min(a_only, b_only)
            expected = binomtest(k, n, 0.5, alternative="two-sided").pvalue
            actual = st.exact_mcnemar_pvalue(a_only, b_only)
            self.assertAlmostEqual(actual, expected, places=9)

    def test_large_discordant_count_does_not_crash_or_over_underflow(self):
        # Mirrors real data scale (results/diagnostic_analysis/paired_model_comparison.csv:
        # majority_lang vs perceptron_pipeline_ud_transfer test split has
        # a_only=28882, b_only=17596).
        p = st.exact_mcnemar_pvalue(28882, 17596)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        # Such a large imbalance (46478 discordant pairs) must be significant.
        self.assertLess(p, 1e-100)


class TestHolmBonferroni(unittest.TestCase):
    def test_hand_computed_example(self):
        # p = [0.01, 0.04, 0.03, 0.005], alpha = 0.05
        # sorted ascending: idx3(.005), idx0(.01), idx2(.03), idx1(.04)
        # raw adj: idx3=.005*4=.02, idx0=.01*3=.03, idx2=.03*2=.06, idx1=.04*1=.04
        # cumulative max (monotonic): idx3=.02, idx0=.03, idx2=.06, idx1=.06
        pvalues = [0.01, 0.04, 0.03, 0.005]
        adjusted, reject = st.holm_bonferroni(pvalues, alpha=0.05)
        expected_adjusted = [0.03, 0.06, 0.06, 0.02]
        for a, e in zip(adjusted, expected_adjusted):
            self.assertAlmostEqual(a, e, places=9)
        self.assertEqual(reject, [True, False, False, True])

    def test_adjusted_pvalues_never_below_raw_pvalues(self):
        pvalues = [0.001, 0.2, 0.03, 0.049, 0.5]
        adjusted, _ = st.holm_bonferroni(pvalues, alpha=0.05)
        for raw, adj in zip(pvalues, adjusted):
            self.assertGreaterEqual(adj, raw - 1e-12)

    def test_adjusted_pvalues_capped_at_one(self):
        pvalues = [0.9, 0.95, 0.99, 0.999]
        adjusted, _ = st.holm_bonferroni(pvalues, alpha=0.05)
        for adj in adjusted:
            self.assertLessEqual(adj, 1.0)

    def test_single_pvalue_unadjusted(self):
        adjusted, reject = st.holm_bonferroni([0.03], alpha=0.05)
        self.assertAlmostEqual(adjusted[0], 0.03, places=12)
        self.assertTrue(reject[0])


class TestDocLevelBootstrap(unittest.TestCase):
    def test_reproducible_with_same_seed(self):
        ca = st.np.array([10, 20, 5, 8], dtype=st.np.int64)
        cb = st.np.array([8, 18, 6, 7], dtype=st.np.int64)
        n = st.np.array([12, 22, 8, 10], dtype=st.np.int64)
        r1 = st.doc_level_bootstrap(ca, cb, n, n_boot=500, seed=42)
        r2 = st.doc_level_bootstrap(ca, cb, n, n_boot=500, seed=42)
        self.assertTrue(st.np.array_equal(r1["distribution"], r2["distribution"]))

    def test_ci_is_exactly_zero_when_models_identical(self):
        ca = st.np.array([10, 20, 5, 8], dtype=st.np.int64)
        n = st.np.array([12, 22, 8, 10], dtype=st.np.int64)
        r = st.doc_level_bootstrap(ca, ca.copy(), n, n_boot=200, seed=42)
        self.assertEqual(r["observed_diff"], 0.0)
        self.assertEqual(r["ci_low"], 0.0)
        self.assertEqual(r["ci_high"], 0.0)
        self.assertTrue(r["ci_contains_zero"])

    def test_observed_diff_is_a_minus_b(self):
        # Model A strictly better on every document -> diff should be
        # positive and the 95% CI should exclude zero.
        ca = st.np.array([10, 20, 8, 9, 15], dtype=st.np.int64)
        cb = st.np.array([2, 5, 1, 2, 3], dtype=st.np.int64)
        n = st.np.array([10, 20, 8, 9, 15], dtype=st.np.int64)
        r = st.doc_level_bootstrap(ca, cb, n, n_boot=500, seed=42)
        self.assertGreater(r["observed_diff"], 0.0)
        self.assertGreater(r["ci_low"], 0.0)
        self.assertFalse(r["ci_contains_zero"])

        # Swapping A and B should flip the sign exactly.
        r_swapped = st.doc_level_bootstrap(cb, ca, n, n_boot=500, seed=42)
        self.assertAlmostEqual(r_swapped["observed_diff"], -r["observed_diff"], places=12)

    def test_percentile_ci_matches_numpy_percentile_definition(self):
        ca = st.np.array([10, 20, 5, 8, 30], dtype=st.np.int64)
        cb = st.np.array([8, 18, 6, 7, 25], dtype=st.np.int64)
        n = st.np.array([12, 22, 8, 10, 35], dtype=st.np.int64)
        r = st.doc_level_bootstrap(ca, cb, n, n_boot=1000, seed=42)
        expected_low, expected_high = st.np.percentile(r["distribution"], [2.5, 97.5])
        self.assertAlmostEqual(r["ci_low"], float(expected_low), places=12)
        self.assertAlmostEqual(r["ci_high"], float(expected_high), places=12)


# ---------------------------------------------------------------------------
# CoNLL-U loading / alignment validation (duplicate, missing, mismatched)
# ---------------------------------------------------------------------------

def _write_conllu(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestLoadModelTokensValidation(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        st.load_model_tokens.cache_clear()

    def tearDown(self):
        st.load_model_tokens.cache_clear()
        self._tmpdir.cleanup()

    def test_duplicate_token_key_raises(self):
        # Same doc_id/sent_id appears twice -> token index collides.
        conllu = (
            "# doc_id = D1\n"
            "# sent_id = S1\n"
            "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
            "\n"
            "# doc_id = D1\n"
            "# sent_id = S1\n"
            "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
        )
        fpath = self.tmp_path / "dup.conllu"
        _write_conllu(fpath, conllu)
        with patch.object(st, "get_pred_path", return_value=fpath):
            with self.assertRaisesRegex(ValueError, "duplicate token key"):
                st.load_model_tokens("fake_dup_model", "fake_split")

    def test_missing_doc_id_raises(self):
        conllu = (
            "# sent_id = S1\n"
            "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
        )
        fpath = self.tmp_path / "nodoc.conllu"
        _write_conllu(fpath, conllu)
        with patch.object(st, "get_pred_path", return_value=fpath):
            with self.assertRaisesRegex(ValueError, "doc_id"):
                st.load_model_tokens("fake_nodoc_model", "fake_split")

    def test_missing_predupos_raises(self):
        conllu = (
            "# doc_id = D1\n"
            "# sent_id = S1\n"
            "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"
        )
        fpath = self.tmp_path / "nopred.conllu"
        _write_conllu(fpath, conllu)
        with patch.object(st, "get_pred_path", return_value=fpath):
            with self.assertRaisesRegex(ValueError, "PredUPOS"):
                st.load_model_tokens("fake_nopred_model", "fake_split")

    def test_get_pred_path_missing_file_raises_with_checked_paths(self):
        with self.assertRaisesRegex(FileNotFoundError, "Checked:"):
            st.get_pred_path("definitely_not_a_real_model", "test")


class TestValidateAlignment(unittest.TestCase):
    def test_missing_key_raises(self):
        records_a = {
            ("D1", "S1", 0): ("D1", "NOUN", "NOUN"),
            ("D1", "S1", 1): ("D1", "VERB", "VERB"),
        }
        records_b = {
            ("D1", "S1", 0): ("D1", "NOUN", "NOUN"),
        }
        with self.assertRaisesRegex(ValueError, "missing from"):
            st.validate_alignment(records_a, records_b, "model_a", "model_b", "test")

    def test_gold_mismatch_raises(self):
        records_a = {("D1", "S1", 0): ("D1", "NOUN", "NOUN")}
        records_b = {("D1", "S1", 0): ("D1", "VERB", "NOUN")}
        with self.assertRaisesRegex(ValueError, "Gold UPOS mismatch"):
            st.validate_alignment(records_a, records_b, "model_a", "model_b", "test")

    def test_success_returns_sorted_shared_keys(self):
        records_a = {
            ("D2", "S1", 0): ("D2", "NOUN", "NOUN"),
            ("D1", "S1", 0): ("D1", "VERB", "VERB"),
        }
        records_b = {
            ("D2", "S1", 0): ("D2", "NOUN", "PUNCT"),
            ("D1", "S1", 0): ("D1", "VERB", "NOUN"),
        }
        keys = st.validate_alignment(records_a, records_b, "model_a", "model_b", "test")
        self.assertEqual(keys, sorted(records_a.keys()))


# ---------------------------------------------------------------------------
# End-to-end integration test with a small, fully hand-verified fixture
# ---------------------------------------------------------------------------

MODEL_A_FIXTURE = (
    "# doc_id = D1\n"
    "# sent_id = S1\n"
    "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
    "2\ttok1\t_\tVERB\t_\t_\t0\troot\t_\tPredUPOS=VERB\n"
    "\n"
    "# doc_id = D1\n"
    "# sent_id = S2\n"
    "1\ttok0\t_\tADP\t_\t_\t0\troot\t_\tPredUPOS=VERB\n"
    "\n"
    "# doc_id = D2\n"
    "# sent_id = S3\n"
    "1\ttok0\t_\tPUNCT\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
    "2\ttok1\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
)

# Same gold (col 4), different PredUPOS (MISC), by design:
#   D1/S1/tok0: both correct        -> both_correct
#   D1/S1/tok1: A correct, B wrong  -> a_only
#   D1/S2/tok0: both wrong          -> both_wrong
#   D2/S3/tok0: A wrong, B correct  -> b_only
#   D2/S3/tok1: both correct        -> both_correct
MODEL_B_FIXTURE = (
    "# doc_id = D1\n"
    "# sent_id = S1\n"
    "1\ttok0\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
    "2\ttok1\t_\tVERB\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
    "\n"
    "# doc_id = D1\n"
    "# sent_id = S2\n"
    "1\ttok0\t_\tADP\t_\t_\t0\troot\t_\tPredUPOS=VERB\n"
    "\n"
    "# doc_id = D2\n"
    "# sent_id = S3\n"
    "1\ttok0\t_\tPUNCT\t_\t_\t0\troot\t_\tPredUPOS=PUNCT\n"
    "2\ttok1\t_\tNOUN\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
)


class TestComparePairIntegration(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        self.path_a = tmp_path / "model_a.conllu"
        self.path_b = tmp_path / "model_b.conllu"
        _write_conllu(self.path_a, MODEL_A_FIXTURE)
        _write_conllu(self.path_b, MODEL_B_FIXTURE)
        st.load_model_tokens.cache_clear()

    def tearDown(self):
        st.load_model_tokens.cache_clear()
        self._tmpdir.cleanup()

    def _fake_get_pred_path(self, model, split):
        return {"fixture_model_a": self.path_a, "fixture_model_b": self.path_b}[model]

    def test_counts_and_mcnemar_and_bootstrap_use_same_token_population(self):
        with patch.object(st, "get_pred_path", side_effect=self._fake_get_pred_path):
            r = st.compare_pair(
                "fixture_model_a", "fixture_model_b", "fixture_split", n_boot=200, seed=42
            )

        self.assertEqual(r["total_tokens"], 5)
        self.assertEqual(r["both_correct"], 2)
        self.assertEqual(r["both_wrong"], 1)
        self.assertEqual(r["a_only_correct"], 1)
        self.assertEqual(r["b_only_correct"], 1)
        self.assertEqual(r["discordant_pairs"], 2)

        # a_only == b_only == 1 -> p-value always saturates to 1.0 (see
        # TestExactMcNemar.test_a_equals_b_always_saturates_to_one).
        self.assertEqual(r["mcnemar_exact_p"], 1.0)
        self.assertEqual(r["mcnemar_exact_p"], st.exact_mcnemar_pvalue(1, 1))

        # Doc-level bootstrap must be built from the SAME aligned token
        # population as the McNemar counts: total tokens across documents
        # must equal total_tokens, and correct_a - correct_b must match
        # both_correct/both_wrong/a_only/b_only bookkeeping.
        boot = r["bootstrap"]
        self.assertEqual(boot["n_documents"], 2)  # D1, D2
        self.assertEqual(boot["total_tokens"], 5)
        # correct_a total = both_correct + a_only = 3; correct_b total =
        # both_correct + b_only = 3; observed diff = (3-3)/5 = 0.0.
        self.assertAlmostEqual(boot["observed_diff"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
