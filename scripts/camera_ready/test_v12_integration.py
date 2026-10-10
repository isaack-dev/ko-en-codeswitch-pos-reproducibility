"""Regression edge cases and byte-repeatability of the gated v1.2 pipeline."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import audit_v12_reference as audit
import evaluate_v12_reference as evaluation


def fixture():
    rows, predictions = [], []
    for s in range(1, 6):
        for i in range(s+1):
            rows.append({"sample_id": f"fixture-{s}", "source_sentence_id": f"sid-{s}",
                "token_ordinal": str(i+1), "stratum": str(s), "human_upos_v12": "NOUN" if i == 0 else "PUNCT",
                "reference_status": "RESOLVED"})
            predictions.append("NOUN" if i == 0 else "VERB")
    return rows, predictions


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = patch.object(audit, "OUT", Path(self.temp.name))
        self.out.start()
        self.addCleanup(self.out.stop)
        self.rows, self.pred = fixture()

    def test_exact_micro_and_stratum_counts(self):
        results = evaluation.evaluate(self.rows, {"fixture": self.pred})
        self.assertEqual(results["v12_primary_results.csv"][0]["exact_accuracy"], "5/20")
        self.assertEqual([(r["correct"], r["eligible_tokens"]) for r in results["v12_stratum_results.csv"]], [(1, n) for n in (2,3,4,5,6)])

    def test_equal_mean_is_not_micro(self):
        result = evaluation.evaluate(self.rows, {"fixture": self.pred})
        mean = result["v12_equal_stratum_summary.csv"][0]
        self.assertEqual(Fraction(mean["exact_mean_numerator"], mean["exact_mean_denominator"]), Fraction(29, 100))
        self.assertNotEqual(mean["equal_stratum_accuracy"], result["v12_primary_results.csv"][0]["accuracy"])

    def test_nonpunct_keeps_erroneous_punct_prediction(self):
        self.pred[0] = "PUNCT"
        result = evaluation.evaluate(self.rows, {"fixture": self.pred})["v12_nonpunct_results.csv"][0]
        self.assertEqual((result["correct"], result["eligible_tokens"]), (4,5))

    def test_unresolved_common_coverage_and_full_sample_bounds(self):
        self.rows[1].update(reference_status="UNRESOLVED", human_upos_v12="")
        result = evaluation.evaluate(self.rows, {"first": self.pred, "second": self.pred})
        for r in result["v12_primary_results.csv"]:
            self.assertEqual((r["eligible_tokens"], r["unresolved_tokens"], r["reference_coverage"]), (19,1,.95))
            self.assertEqual((r["full_sample_lower_correct"],r["full_sample_upper_correct"],r["full_sample_bound_denominator"]),(5,6,20))
        first_stratum = result["v12_stratum_results.csv"][0]
        self.assertEqual(first_stratum["unresolved_tokens"],1)

    def test_missing_prediction_cannot_reduce_denominator(self):
        with self.assertRaisesRegex(audit.ValidationFailure, "denominator cannot shrink"):
            evaluation.evaluate(self.rows, {"broken": self.pred[:-1]})

    def test_supported_macro_and_absent_reference_classes(self):
        self.pred[1] = "SYM"  # prediction-only class must not enter macro mean
        result = evaluation.evaluate(self.rows, {"fixture": self.pred})
        classes = evaluation.compute_per_class(list(zip((r["human_upos_v12"] for r in self.rows),self.pred)))
        expected = (classes["NOUN"]["f1"]+classes["PUNCT"]["f1"])/2
        self.assertEqual(result["v12_primary_results.csv"][0]["macro_f1_positive_reference_support"], expected)
        sym = next(r for r in result["v12_upos_metrics.csv"] if r["upos"] == "SYM")
        punct = next(r for r in result["v12_upos_metrics.csv"] if r["upos"] == "PUNCT")
        self.assertEqual((sym["support"],sym["included_in_macro"]),(0,False))
        self.assertEqual(punct["f1"],0)
        self.assertEqual(result["v12_primary_results.csv"][0]["support_weighted_f1"],.25)

    def test_full_confusion_matrix_totals(self):
        result = evaluation.evaluate(self.rows, {"fixture": self.pred})
        matrix = result["v12_confusion_matrices.csv"]
        self.assertEqual(len(matrix),17*17)
        self.assertEqual(sum(r["tokens"] for r in matrix),20)
        self.assertEqual(sum(r["tokens"] for r in matrix if r["human_upos"]==r["predicted_upos"]),5)

    def test_paired_system_denominators(self):
        result = evaluation.evaluate(self.rows, {"first": self.pred, "second": list(reversed(self.pred))})
        self.assertEqual({r["eligible_tokens"] for r in result["v12_primary_results.csv"]},{20})

    def test_duplicate_sample_token_and_source_keys(self):
        with self.assertRaises(audit.ValidationFailure):
            evaluation.scored(self.rows+[self.rows[0]])
        changed = copy.deepcopy(self.rows)
        changed[1]["source_sentence_id"] = changed[0]["source_sentence_id"]
        changed[1]["token_ordinal"] = changed[0]["token_ordinal"]
        with self.assertRaises(audit.ValidationFailure):
            evaluation.scored(changed)

    def test_conservative_leaves_adjectives_unchanged(self):
        self.assertEqual(evaluation.apply_map("ADJ","KO",1,True),"ADJ")
        self.assertEqual(evaluation.apply_map("PART","KO",1,True),"ADP")


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = audit.csv_rows(audit.REFERENCE)
        cls.manifest = audit.csv_rows(audit.SAMPLE/"sampling_manifest.csv")
        cls.test = {s.sid:s for s in audit.sampling.load_split(audit.sampling.SPLITS["test"])}

    def test_full_frozen_source_manifest(self):
        self.assertEqual(len(audit.validate_source(self.rows,self.manifest,self.test)),180)

    def test_source_form_mismatch_detected_even_if_manifest_altered(self):
        rows, manifest = copy.deepcopy(self.rows), copy.deepcopy(self.manifest)
        rows[0]["token"] = manifest[0]["token"] = "ALTERED"
        with self.assertRaisesRegex(audit.ValidationFailure,"Original source"):
            audit.validate_source(rows,manifest,self.test)

    def test_source_ordinal_mismatch_detected(self):
        rows, manifest = copy.deepcopy(self.rows), copy.deepcopy(self.manifest)
        rows[0]["token_ordinal"] = manifest[0]["token_ordinal"] = "9999"
        with self.assertRaisesRegex(audit.ValidationFailure,"Original source"):
            audit.validate_source(rows,manifest,self.test)

    def test_duplicate_sample_identity_detected(self):
        rows, manifest = copy.deepcopy(self.rows), copy.deepcopy(self.manifest)
        first, second = rows[0]["sample_id"], next(r["sample_id"] for r in rows if r["sample_id"] != rows[0]["sample_id"])
        for seq in (rows,manifest):
            for r in seq:
                if r["sample_id"] == second:
                    r["sample_id"] = first
        with self.assertRaises(audit.ValidationFailure):
            audit.validate_source(rows,manifest,self.test)

    def test_alignment_detects_form_id_context_and_missing_token(self):
        reference = self.rows[:1]
        k = evaluation.source_key(reference[0])
        token = {f:reference[0][f] for f in ("original_token_id","token","sentence","source_document_id")}
        token["prediction"] = "NOUN"
        self.assertEqual(evaluation.align(reference,{k:token}),["NOUN"])
        for field in ("original_token_id","token","sentence","source_document_id"):
            bad = {**token,field:"MISMATCH"}
            with self.assertRaises(audit.ValidationFailure):
                evaluation.align(reference,{k:bad})
        with self.assertRaises(audit.ValidationFailure):
            evaluation.align(reference,{})

    def test_predupos_reader_excludes_nonreal_and_never_uses_silver_column(self):
        text = "# doc_id = doc1\n# split = test\n# sent_id = sid1\n# text = word\n"
        text += "1-2\tignored\t_\t_\t_\t_\t_\t_\t_\t_\n"
        text += "1\tword\t_\tVERB\t_\t_\t0\troot\t_\tPredUPOS=NOUN\n"
        text += "1.1\tempty\t_\tVERB\t_\t_\t_\t_\t_\t_\n\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fixture.conllu"
            path.write_text(text,encoding="utf-8")
            stream, detail = evaluation.load_stream(path,{"sid1"})
            self.assertEqual(stream["sid1",1]["prediction"],"NOUN")
            self.assertEqual(detail["file_real_tokens"],1)
            path.write_text(text.replace("PredUPOS=NOUN","_"),encoding="utf-8")
            with self.assertRaisesRegex(audit.ValidationFailure,"Missing/invalid prediction"):
                evaluation.load_stream(path,{"sid1"})

    def test_frozen_checksums_and_legacy_preservation(self):
        for path,expected in audit.EXPECTED.items():
            self.assertEqual(audit.sha(path),expected)
        baseline = audit.preserve()
        self.assertEqual(len(baseline["files"]),786)
        legacy = "scripts/evaluate_models_on_test_only_human_reference.py"
        self.assertEqual(audit.sha(audit.ROOT/legacy),baseline["files"][legacy])


def main():
    log = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (MetricTests,SourceTests)])
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not result.wasSuccessful():
        (audit.OUT/"test_report.txt").write_text(log.getvalue(),encoding="utf-8")
        print(log.getvalue())
        raise SystemExit(1)
    # These are full integration repetitions: reference/annotation gates and
    # prediction provenance execute each time. No source/model is regenerated.
    evaluation.run()
    names = sorted(p.name for p in audit.OUT.iterdir() if p.is_file() and p.suffix in (".json",".csv",".md",".txt")
        and p.name not in ("test_report.txt","repeatability_validation.json","evaluation_blocker.json","validation_blocker.json"))
    first = {name:audit.sha(audit.OUT/name) for name in names}
    evaluation.run()
    second = {name:audit.sha(audit.OUT/name) for name in names}
    mismatches = [name for name in names if first[name] != second[name]]
    baseline = audit.preserve()
    verification = {"unit_regression_tests_run":result.testsRun,"unit_regression_tests_passed":result.wasSuccessful(),
        "full_integration_runs":2,"byte_identical_files":len(names),"mismatches":mismatches,
        "output_sha256":second,"preexisting_files_verified_unchanged":len(baseline["files"]),
        "command":"python -B scripts/camera_ready/test_v12_integration.py",
        "script":evaluation.record(Path(__file__))}
    (audit.OUT/"repeatability_validation.json").write_bytes(audit.dump(verification))
    log.write(f"\nFull frozen-input integration runs: 2\nByte-identical generated files: {len(names)}\nMismatches: {mismatches}\nPreexisting files verified unchanged: {len(baseline['files'])}\n")
    if mismatches:
        log.write("FAIL: serialized outputs not repeatable\n")
    else:
        log.write("PASS: all regression tests, integration gates, paired denominators, source preservation and byte-repeatability\n")
    (audit.OUT/"test_report.txt").write_text(log.getvalue(),encoding="utf-8")
    print(log.getvalue())
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
