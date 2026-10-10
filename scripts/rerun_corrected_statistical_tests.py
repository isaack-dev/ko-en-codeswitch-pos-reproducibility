"""
rerun_corrected_statistical_tests.py - Rerun scripts/statistical_tests.py
(commit c4560bf, unmodified) against the corrected UD-transfer predictions.

Does NOT edit scripts/statistical_tests.py. Instead, builds a mixed input
tree at runtime in a temporary directory containing:
  - byte-identical copies of the 4 unaffected model prediction files
    (majority_global, majority_lang, perceptron_joint, perceptron_pipeline_panyo)
    from results/baselines/,
  - the corrected perceptron_pipeline_ud_transfer predictions,
  - the corrected ud_transfer_panyo_coarse / ud_transfer_ko_oriented mapping predictions,
    both from outputs/corrected_ud_transfer/predictions/,
points statistical_tests.py's two module-level input-directory constants at
that temporary tree so all 6 predefined pairs and the Holm-Bonferroni family
structure stay identical to the original run (only the 3 pairs involving
UD-transfer see different input data), then deletes the temporary directory
automatically on exit. Nothing under outputs/corrected_ud_transfer/ other
than statistical_tests/ is created or left behind by this script.

Run from repo root (after scripts/rerun_corrected_ud_transfer.py):
  python scripts/rerun_corrected_statistical_tests.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import statistical_tests as st  # noqa: E402  (unmodified, commit c4560bf)

ROOT = Path(__file__).parent.parent
RESULTS_BASELINES_ORIG = ROOT / "results" / "baselines"          # unchanged in-domain models
CORRECTED_PRED = ROOT / "outputs" / "corrected_ud_transfer" / "predictions"  # corrected UD-transfer
OUT_DIR = ROOT / "outputs" / "corrected_ud_transfer" / "statistical_tests"

UNCHANGED_MODELS = [
    "majority_global", "majority_lang", "perceptron_joint", "perceptron_pipeline_panyo",
]
SPLITS = ["dev", "test"]


def build_mixed_input_tree(tmp_root: Path) -> tuple[Path, Path]:
    """Populate tmp_root/{baselines,diagnostic} and return their paths."""
    baselines_dir = tmp_root / "baselines"
    diag_dir = tmp_root / "diagnostic"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    for model in UNCHANGED_MODELS:
        for split in SPLITS:
            fname = f"{model}_{split}_predictions.conllu"
            shutil.copy2(RESULTS_BASELINES_ORIG / fname, baselines_dir / fname)

    for split in SPLITS:
        fname = f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        shutil.copy2(CORRECTED_PRED / fname, baselines_dir / fname)
        for scheme in ("panyo_coarse", "ko_oriented"):
            mname = f"ud_transfer_{scheme}_{split}_predictions.conllu"
            shutil.copy2(CORRECTED_PRED / mname, diag_dir / mname)

    return baselines_dir, diag_dir


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="statistical_tests_input_") as tmp:
        baselines_dir, diag_dir = build_mixed_input_tree(Path(tmp))

        st.load_model_tokens.cache_clear()
        st.RESULTS_BASELINES = baselines_dir
        st.RESULTS_DIAG = diag_dir

        print(f"Patched st.RESULTS_BASELINES -> {st.RESULTS_BASELINES}")
        print(f"Patched st.RESULTS_DIAG      -> {st.RESULTS_DIAG}")

        st.run(
            splits=st.SPLITS,
            seed=st.SEED,
            n_boot=st.N_BOOTSTRAP,
            alpha=st.ALPHA,
            out_dir=OUT_DIR,
        )
    # tempfile.TemporaryDirectory removes tmp_root (and everything copied into
    # it) automatically on context-manager exit -- no manual cleanup needed.


if __name__ == "__main__":
    main()
