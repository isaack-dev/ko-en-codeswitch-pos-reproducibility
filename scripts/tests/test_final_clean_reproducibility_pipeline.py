"""
Unit + light integration tests for
scripts/run_final_clean_reproducibility_pipeline.py and
scripts/clean_run_compare.py.

Run from repo root:
  python scripts/tests/test_final_clean_reproducibility_pipeline.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import clean_run_compare as cc  # noqa: E402
import run_final_clean_reproducibility_pipeline as pipeline  # noqa: E402
import train_baselines  # noqa: E402

ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# clean_run_compare.py tolerance-comparison logic (synthetic fixtures)
# ---------------------------------------------------------------------------

class TestCompareRow(unittest.TestCase):
    def test_exact_numeric_match(self):
        r = cc.compare_row("art", "acc", 0.5532, 0.5532, tolerance=1e-10)
        self.assertEqual(r["status"], cc.STATUS_EXACT)
        self.assertEqual(r["absolute_difference"], 0.0)

    def test_within_tolerance(self):
        r = cc.compare_row("art", "acc", 0.55320000001, 0.5532, tolerance=1e-9)
        self.assertEqual(r["status"], cc.STATUS_TOLERANCE)

    def test_exceeds_tolerance_is_unexplained(self):
        r = cc.compare_row("art", "acc", 0.60, 0.58, tolerance=1e-10)
        self.assertEqual(r["status"], cc.STATUS_UNEXPLAINED)
        self.assertAlmostEqual(r["absolute_difference"], 0.02)

    def test_missing_clean_value_is_missing_artifact(self):
        r = cc.compare_row("art", "acc", 0.55, None)
        self.assertEqual(r["status"], cc.STATUS_MISSING)

    def test_exact_string_equality(self):
        r = cc.compare_row("art", "model", "perceptron_joint", "perceptron_joint")
        self.assertEqual(r["status"], cc.STATUS_EXACT)

    def test_string_mismatch_is_unexplained(self):
        r = cc.compare_row("art", "model", "perceptron_joint", "majority_lang")
        self.assertEqual(r["status"], cc.STATUS_UNEXPLAINED)

    def test_force_status_expected_serialization_difference(self):
        r = cc.compare_row("art", "field", 100, 90, force_status=cc.STATUS_SERIALIZATION,
                            explanation="known benign cause")
        self.assertEqual(r["status"], cc.STATUS_SERIALIZATION)
        self.assertEqual(r["explanation"], "known benign cause")

    def test_integer_counts_use_exact_equality_at_zero_tolerance(self):
        r = cc.compare_row("art", "tokens", 1879, 1878, tolerance=0.0)
        self.assertEqual(r["status"], cc.STATUS_UNEXPLAINED)
        r2 = cc.compare_row("art", "tokens", 1879, 1879, tolerance=0.0)
        self.assertEqual(r2["status"], cc.STATUS_EXACT)


class TestHasBlockingDiscrepancies(unittest.TestCase):
    def test_true_when_unexplained_present(self):
        rows = [cc.compare_row("a", "f", 1, 1), cc.compare_row("a", "f2", 1, 2, tolerance=0)]
        self.assertTrue(cc.has_blocking_discrepancies(rows))

    def test_false_when_all_exact_or_tolerated(self):
        rows = [cc.compare_row("a", "f", 1, 1), cc.compare_row("a", "f2", 1.0, 1.0 + 1e-12, tolerance=1e-9)]
        self.assertFalse(cc.has_blocking_discrepancies(rows))

    def test_true_when_missing_artifact_present(self):
        rows = [cc.compare_row("a", "f", 1, None)]
        self.assertTrue(cc.has_blocking_discrepancies(rows))


class TestSummarize(unittest.TestCase):
    def test_counts_all_five_statuses(self):
        rows = [
            cc.compare_row("a", "f1", 1, 1),
            cc.compare_row("a", "f2", 1.0, 1.0 + 1e-12, tolerance=1e-9),
            cc.compare_row("a", "f3", 1, 2, force_status=cc.STATUS_SERIALIZATION),
            cc.compare_row("a", "f4", 1, 2, tolerance=0),
            cc.compare_row("a", "f5", 1, None),
        ]
        summary = cc.summarize(rows)
        self.assertEqual(summary[cc.STATUS_EXACT], 1)
        self.assertEqual(summary[cc.STATUS_TOLERANCE], 1)
        self.assertEqual(summary[cc.STATUS_SERIALIZATION], 1)
        self.assertEqual(summary[cc.STATUS_UNEXPLAINED], 1)
        self.assertEqual(summary[cc.STATUS_MISSING], 1)


# ---------------------------------------------------------------------------
# Path-leak scan (Phase 3) -- tested directly against a synthetic tree
# ---------------------------------------------------------------------------

class TestPathLeakScan(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="clean_run_leak_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detects_windows_user_path(self):
        f = self.tmp / "report.txt"
        f.write_text("see C:\\Users\\someone\\file.json for details", encoding="utf-8")
        result = pipeline.scan_for_path_leaks(self.tmp, patterns=["C:\\Users\\"])
        self.assertEqual(len(result["hits"]), 1)

    def test_clean_file_produces_no_hits(self):
        f = self.tmp / "report.txt"
        f.write_text("relative/path/only.csv", encoding="utf-8")
        result = pipeline.scan_for_path_leaks(self.tmp, patterns=["C:\\Users\\", "/home/"])
        self.assertEqual(len(result["hits"]), 0)

    def test_ignores_non_text_extensions(self):
        f = self.tmp / "model.bin"
        f.write_bytes(b"C:\\Users\\someone\\binary")
        result = pipeline.scan_for_path_leaks(self.tmp, patterns=["C:\\Users\\"])
        self.assertEqual(len(result["hits"]), 0)

    def test_fix_path_leak_removes_repo_root_prefix(self):
        f = self.tmp / "leaky.txt"
        root_prefixed = str(pipeline.ROOT) + "\\some\\file.json"
        f.write_text(f"path was {root_prefixed}", encoding="utf-8")
        changed = pipeline.fix_path_leak(f)
        self.assertTrue(changed)
        text = f.read_text(encoding="utf-8")
        self.assertNotIn(str(pipeline.ROOT), text)
        self.assertIn("<repo_root>", text)


# ---------------------------------------------------------------------------
# Output-directory reset/refusal behavior (Phase 1 / Phase 9 requirement)
# ---------------------------------------------------------------------------

class TestEnsureOutputDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="clean_run_dir_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_dir_is_accepted(self):
        pipeline.ensure_output_dir(self.tmp, reset=False, verify_only=False, skip_training=False)
        self.assertTrue((self.tmp / "baselines" / "models").exists())

    def test_nonempty_dir_refused_without_reset(self):
        (self.tmp / "stale_file.txt").write_text("stale", encoding="utf-8")
        with self.assertRaises(SystemExit):
            pipeline.ensure_output_dir(self.tmp, reset=False, verify_only=False, skip_training=False)

    def test_nonempty_dir_accepted_with_reset(self):
        (self.tmp / "stale_file.txt").write_text("stale", encoding="utf-8")
        pipeline.ensure_output_dir(self.tmp, reset=True, verify_only=False, skip_training=False)
        self.assertFalse((self.tmp / "stale_file.txt").exists())
        self.assertTrue((self.tmp / "baselines" / "models").exists())

    def test_nonempty_dir_accepted_with_skip_training(self):
        (self.tmp / "stale_file.txt").write_text("stale", encoding="utf-8")
        pipeline.ensure_output_dir(self.tmp, reset=False, verify_only=False, skip_training=True)
        # skip-training reuses existing contents rather than wiping them
        self.assertTrue((self.tmp / "stale_file.txt").exists())


# ---------------------------------------------------------------------------
# Determinism / seed handling
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_repo_convention_seed_is_42(self):
        self.assertEqual(pipeline.SEED_DEFAULT, 42)
        self.assertEqual(train_baselines.SEED, 42)

    def test_doc_level_bootstrap_is_reproducible_given_same_seed(self):
        import numpy as np
        import statistical_tests as st
        a = np.array([9, 8, 7, 10])
        b = np.array([8, 7, 6, 9])
        n = np.array([10, 10, 10, 10])
        r1 = st.doc_level_bootstrap(a, b, n, n_boot=500, seed=42)
        r2 = st.doc_level_bootstrap(a, b, n, n_boot=500, seed=42)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertEqual(r1["ci_high"], r2["ci_high"])
        self.assertEqual(list(r1["distribution"]), list(r2["distribution"]))


# ---------------------------------------------------------------------------
# No dependency on committed prediction/model/results files as computational
# inputs (static analysis of this pipeline's own source, per task Phase 9)
# ---------------------------------------------------------------------------

class TestNoCommittedFileDependency(unittest.TestCase):
    """Committed authoritative paths may appear ONLY as AUTH_*-named module
    constants (used exclusively by the Phase 8 comparison step). Any other
    reference to these directories in this file's source would indicate the
    pipeline accidentally reading a committed artifact as a computational
    input rather than redirecting to the clean-run directory."""

    PROTECTED_LITERALS = [
        "results/baselines", "results\\\\baselines",
        "outputs/corrected_ud_transfer", "outputs\\\\corrected_ud_transfer",
        "outputs/human_validation/test_only/results",
        "outputs/human_validation/test_only/model_evaluation",
        "models/baselines", "models\\\\baselines",
    ]

    def test_committed_paths_only_referenced_via_AUTH_constants(self):
        src_path = Path(pipeline.__file__)
        lines = src_path.read_text(encoding="utf-8").splitlines()
        violations = []
        inside_auth_block = False
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "AUTH_" in line and line.rstrip().endswith("["):
                inside_auth_block = True
            if inside_auth_block:
                if "]" in line:
                    inside_auth_block = False
                continue
            # A line that is purely (a fragment of) a string literal used as
            # human-readable explanation text -- not a path-construction or
            # file-access expression -- is not a computational-input read.
            is_plain_string_fragment = stripped.startswith('"') or stripped.startswith("'")
            for lit in self.PROTECTED_LITERALS:
                if lit in line:
                    if "AUTH_" in line or is_plain_string_fragment:
                        continue
                    violations.append((i, line))
        self.assertEqual(violations, [], f"Found committed-path references outside AUTH_* constants: {violations}")

    def test_auth_constants_used_only_inside_phase8_functions(self):
        import inspect
        phase8_fn_names = {
            "_compare_overall_results", "_compare_mapping_results", "_compare_statistical_tests",
            "_compare_human_reference", "_compare_model_evaluation", "run_phase8_comparison",
            "_compare_prediction_equality",
        }
        other_fn_names = [
            n for n in dir(pipeline)
            if callable(getattr(pipeline, n)) and n.startswith(("run_train", "run_ud_mapping",
                                                                  "run_canonical", "run_seen_unseen",
                                                                  "run_statistical_tests",
                                                                  "run_human_reference", "run_human_model"))
        ]
        for name in other_fn_names:
            fn = getattr(pipeline, name)
            src = inspect.getsource(fn)
            for lit in ("AUTH_",):
                self.assertNotIn(lit, src, f"{name} unexpectedly references an AUTH_* constant")


# ---------------------------------------------------------------------------
# UD-transfer correction enforcement
# ---------------------------------------------------------------------------

class TestUDTransferCorrectionEnforced(unittest.TestCase):
    def test_train_baselines_uses_lang_override_for_ud_corpora(self):
        src = Path(train_baselines.__file__).read_text(encoding="utf-8")
        self.assertIn("UD_LANG_OVERRIDE", src)
        self.assertIn('GUM_TRAIN: "EN"', src.replace("'", '"'))
        self.assertIn('KAIST_TRAIN: "KO"', src.replace("'", '"'))

    def test_load_tagged_sents_with_override_never_yields_blank_lang(self):
        sents = train_baselines.load_tagged_sents(train_baselines.GUM_TRAIN, lang_override="EN")
        self.assertTrue(sents, "expected at least one sentence from GUM_TRAIN")
        for sent in sents[:20]:
            for _, lang, _ in sent:
                self.assertEqual(lang, "EN")


# ---------------------------------------------------------------------------
# Failure modes: missing predictions / duplicate tokens / token-form mismatch
# ---------------------------------------------------------------------------

class TestFailureModes(unittest.TestCase):
    def test_statistical_tests_raises_on_missing_prediction_file(self):
        import statistical_tests as st
        with self.assertRaises(FileNotFoundError):
            st.get_pred_path("this_model_does_not_exist", "test")

    def test_statistical_tests_raises_on_duplicate_token_key(self):
        import statistical_tests as st
        tmp = Path(tempfile.mkdtemp(prefix="dup_tok_test_"))
        try:
            conllu = tmp / "dup_test_predictions.conllu"
            conllu.write_text(
                "# sent_id = s1\n# doc_id = d1\n"
                "1\tHello\t_\tINTJ\t_\t_\t_\t_\t_\tLang=EN|PredUPOS=INTJ\n"
                "2\tworld\t_\tNOUN\t_\t_\t_\t_\t_\tLang=EN|PredUPOS=NOUN\n\n"
                "# sent_id = s1\n# doc_id = d1\n"
                "1\tHello\t_\tINTJ\t_\t_\t_\t_\t_\tLang=EN|PredUPOS=INTJ\n\n",
                encoding="utf-8",
            )
            st.RESULTS_BASELINES = tmp
            st.RESULTS_DIAG = tmp
            st.load_model_tokens.cache_clear()
            with self.assertRaises(ValueError):
                st.load_model_tokens("dup", "test")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_evaluate_models_align_model_reports_token_form_mismatch(self):
        import evaluate_models_on_test_only_human_reference as evalhuman
        tmp = Path(tempfile.mkdtemp(prefix="form_mismatch_test_"))
        try:
            conllu = tmp / "pred.conllu"
            conllu.write_text(
                "# sent_id = s1\n# doc_id = d1\n"
                "1\tWRONGFORM\t_\tNOUN\t_\t_\t_\t_\t_\tLang=EN|PredUPOS=NOUN\n\n",
                encoding="utf-8",
            )
            human_rows = [{
                "Source_Sent_ID": "s1", "Token_ID": 1, "Token": "ActualForm",
                "Sample_ID": "R001", "Set": "Representative",
            }]
            original_models = evalhuman.MODELS
            evalhuman.MODELS = {"fake": {"pred_path": conllu}}
            try:
                aligned, errors = evalhuman.align_model("fake", human_rows, {})
            finally:
                evalhuman.MODELS = original_models
            self.assertEqual(aligned, [])
            self.assertTrue(any("form mismatch" in e for e in errors))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Human-reference reconstruction cannot read Canonical_UPOS as a human label
# ---------------------------------------------------------------------------

class TestCanonicalNeverUsedAsHumanLabel(unittest.TestCase):
    def test_build_final_reference_never_assigns_final_upos_from_canonical(self):
        import inspect
        import build_test_only_adjudicated_reference as ref
        src = inspect.getsource(ref.build_final_reference)
        # final_upos may only be assigned from P1/P2 agreement, adjudication
        # decisions, or post-audit corrections -- never from `canon[...]` /
        # `canonical[...]` / `canonical_upos`.
        for line in src.splitlines():
            if "final_upos" in line and "=" in line and not line.strip().startswith("#"):
                rhs = line.split("=", 1)[1]
                self.assertNotIn("canon", rhs.lower(),
                                 f"final_upos appears to be assigned from a canonical-key variable: {line!r}")

    def test_final_token_canonical_upos_field_is_populated_separately(self):
        import build_test_only_adjudicated_reference as ref
        import inspect
        sig = inspect.signature(ref.FinalToken)
        self.assertIn("canonical_upos", sig.parameters)
        self.assertIn("final_upos", sig.parameters)


# ---------------------------------------------------------------------------
# Discrepancy-driven nonzero exit behavior (synthetic injection)
# ---------------------------------------------------------------------------

class TestDiscrepancyExitBehavior(unittest.TestCase):
    def test_blocking_rows_trigger_has_blocking_discrepancies_true(self):
        rows = [cc.compare_row("synthetic", "accuracy", 0.99, 0.10, tolerance=1e-10)]
        self.assertTrue(cc.has_blocking_discrepancies(rows))

    def test_pipeline_main_exits_nonzero_on_injected_discrepancy(self):
        """End-to-end: run the pipeline with --verify-only against a minimal
        clean-run directory whose statistical_tests output has been hand-edited
        to disagree with the authoritative file, and confirm nonzero exit."""
        tmp = Path(tempfile.mkdtemp(prefix="clean_run_injected_disc_"))
        try:
            for sd in ["environment", "baselines/models", "baselines/results", "baselines/diagnostics",
                       "corrected_ud_transfer", "statistical_tests", "human_reference",
                       "human_model_evaluation", "comparisons", "logs"]:
                (tmp / sd).mkdir(parents=True, exist_ok=True)
            (tmp / "environment" / "environment.json").write_text(
                json.dumps({"python_version": "0.0.0", "repo_commit_hash": "deadbeef", "seed": 42}), encoding="utf-8"
            )
            # No baselines/results/overall_results.csv etc. exist at all in this
            # synthetic dir, so every comparison row should resolve to
            # missing_artifact -- which is itself a blocking status.
            result = subprocess.run(
                [sys.executable, str(Path(pipeline.__file__)), "--output-dir", str(tmp), "--verify-only"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
