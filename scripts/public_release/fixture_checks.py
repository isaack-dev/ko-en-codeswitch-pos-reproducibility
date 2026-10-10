"""Run only synthetic camera-ready regression fixtures; no private corpus access."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts/camera_ready"), str(ROOT/"scripts"), str(ROOT/"scripts/public_release")]
import test_v12_integration as human
import test_additional_diagnostics as diagnostics


def suite():
    loader = unittest.TestLoader()
    tests = unittest.TestSuite([loader.loadTestsFromTestCase(human.MetricTests)])
    # The two remaining RegressionTests require withheld original split and
    # preservation manifests; excluding them is explicit, never a silent pass.
    excluded = {"test_partition_hashes_and_document_split_preserved", "test_all_preexisting_files_unchanged"}
    for name in loader.getTestCaseNames(diagnostics.RegressionTests):
        if name not in excluded:
            tests.addTest(diagnostics.RegressionTests(name))
    # This one source-reader test uses only its own temporary synthetic file.
    # SourceTests.setUpClass otherwise requires private real-data reference rows.
    tests.addTest(unittest.FunctionTestCase(
        lambda: human.SourceTests.test_predupos_reader_excludes_nonreal_and_never_uses_silver_column(unittest.TestCase()),
        description="synthetic prediction reader excludes nonreal nodes and requires explicit predictions"))
    return tests


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if result.wasSuccessful() else 1)
