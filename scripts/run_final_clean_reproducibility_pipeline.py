"""
run_final_clean_reproducibility_pipeline.py - One-command final clean
reproducibility rerun of the complete CCIR experimental pipeline.

Retrains every model from frozen source data (never from committed
model/prediction files), regenerates every statistical test and the
human-reference evaluation, then compares every regenerated artifact
against the already-committed authoritative outputs, producing a
structured discrepancy report.

This script imports the existing, already-verified pipeline scripts as
modules and redirects their module-level path constants (MODELS, RESULTS,
DIAG, OUT_DIR, MODELS dict, ...) to point into an isolated clean-run
directory before calling each script's main()/run(). It never edits those
scripts and never reads their committed outputs as computational inputs.
See the module docstrings of the reused scripts (train_baselines.py,
predict_baselines.py, diagnostic_utils.py, diagnostic_ud_mapping.py,
evaluate_baselines.py, diagnostic_seen_unseen.py,
diagnostic_boundary_analysis.py, statistical_tests.py,
build_test_only_adjudicated_reference.py,
evaluate_models_on_test_only_human_reference.py) for what each one does.

Run from repo root:
  python scripts/run_final_clean_reproducibility_pipeline.py [--output-dir DIR]
      [--seed 42] [--verify-only] [--skip-training] [--reset]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import locale
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_clean_reproducibility_run"
SEED_DEFAULT = 42

# ---------------------------------------------------------------------------
# Frozen (read-only) input paths -- never written to by this pipeline.
# ---------------------------------------------------------------------------
FROZEN_INPUTS = {
    "panyo_train": ROOT / "data/processed/panyo_original_adp_docsplit_train.conllu",
    "panyo_dev": ROOT / "data/processed/panyo_original_adp_docsplit_dev.conllu",
    "panyo_test": ROOT / "data/processed/panyo_original_adp_docsplit_test.conllu",
    "panyo_jsonl_doc_sentence_map": ROOT / "data/processed/panyo_jsonl_doc_sentence_map.csv",
    "data_split_decisions": ROOT / "guidelines/data_split_decisions.md",
    "gum_train": ROOT / "data/en_gum-ud-train.conllu",
    "kaist_train": ROOT / "data/ko_kaist-ud-train.conllu",
    "p1_completed": ROOT / "outputs/human_validation/test_only/annotations/completed/Human_POS_Tagging_test_v1_P1_COMPLETED.xlsx",
    "p2_completed": ROOT / "outputs/human_validation/test_only/annotations/completed/Human_POS_Tagging_test_v1_P2_COMPLETED.xlsx",
    "adjudication_workbook": ROOT / "outputs/human_validation/test_only/annotations/Human_POS_Tagging_test_v1_Blind_Adjudication_FINAL_v1.2.xlsx",
    "canonical_key": ROOT / "outputs/human_validation/test_only/key/Human_POS_Tagging_test_v1_canonical_key.xlsx",
    "guideline_md": ROOT / "outputs/human_validation/guideline/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.md",
    "guideline_docx": ROOT / "outputs/human_validation/guideline/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.docx",
}

REUSED_SCRIPT_FILES = [
    "train_baselines.py", "predict_baselines.py", "diagnostic_utils.py",
    "diagnostic_ud_mapping.py", "evaluate_baselines.py", "diagnostic_seen_unseen.py",
    "diagnostic_boundary_analysis.py", "statistical_tests.py",
    "build_test_only_adjudicated_reference.py",
    "evaluate_models_on_test_only_human_reference.py",
    "perceptron_tagger.py", "read_conllu.py",
]

# ---------------------------------------------------------------------------
# Authoritative (already-committed) comparison targets -- Phase 8 ONLY.
# These constants are the single, explicitly-named place in this file where
# committed output paths may be read; they are never used as computational
# inputs anywhere else in this script (see
# scripts/tests/test_final_clean_reproducibility_pipeline.py for a static
# check enforcing this).
# ---------------------------------------------------------------------------
AUTH_BASELINES_OVERALL = ROOT / "results/baselines/overall_results.csv"
AUTH_CORRECTED_UD_OVERALL = ROOT / "outputs/corrected_ud_transfer/metrics/overall_results.csv"
AUTH_UD_MAPPING_RESULTS = ROOT / "outputs/corrected_ud_transfer/metrics/ud_transfer_mapping_results.csv"
AUTH_STATISTICAL_TESTS = ROOT / "outputs/statistical_tests/paired_comparisons.csv"
# outputs/statistical_tests/paired_comparisons.csv was generated BEFORE the
# UD-transfer lang_override fix (it reads whatever prediction file happened
# to sit in results/baselines/ at the time, which is the stale pre-fix UD
# model); outputs/corrected_ud_transfer/statistical_tests/paired_comparisons.csv
# is the already-committed, already-regenerated counterpart that reflects the
# corrected UD-transfer model. Any PREDEFINED_PAIRS row involving
# perceptron_pipeline_ud_transfer or a ud_transfer_* mapping variant is
# compared against THIS file instead; the other two pairs are unaffected by
# the fix and compared against AUTH_STATISTICAL_TESTS as normal.
AUTH_CORRECTED_STATISTICAL_TESTS = ROOT / "outputs/corrected_ud_transfer/statistical_tests/paired_comparisons.csv"
AUTH_HUMAN_REF_METRICS_JSON = ROOT / "outputs/human_validation/test_only/results/human_silver_metrics.json"
AUTH_HUMAN_REF_TOKENS_CSV = ROOT / "outputs/human_validation/test_only/results/final_human_reference_tokens.csv"
AUTH_HUMAN_MODEL_EVAL_REP_CSV = ROOT / "outputs/human_validation/test_only/model_evaluation/model_metrics_representative.csv"
AUTH_HUMAN_MODEL_EVAL_RANK_CSV = ROOT / "outputs/human_validation/test_only/model_evaluation/silver_vs_human_rank_comparison.csv"
AUTH_PROTECTED_ROOTS = [
    ROOT / "results/baselines",
    ROOT / "outputs/corrected_ud_transfer",
    ROOT / "outputs/statistical_tests",
    ROOT / "outputs/human_validation/test_only/results",
    ROOT / "outputs/human_validation/test_only/model_evaluation",
    ROOT / "models/baselines",
]

PATH_LEAK_PATTERNS = ["C:\\Users\\", "DESKTOP-", "/home/", "Python\\Programs\\CCIR", "Python/Programs/CCIR"]

AUDIT_TARGETS = {
    "majority_global": {"test": 0.5532},
    "majority_lang": {"test": 0.7106},
    "perceptron_joint": {"test": 0.9997},
    "perceptron_pipeline_panyo": {"test": 0.9997},
    "perceptron_pipeline_ud_transfer": {"test": 0.5800},
    "conservative": {"test": 0.5935},
    "panyo_coarse": {"test": 0.6316},
    "ko_oriented": {"test": 0.6313},
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    """Repo-relative POSIX-style path string, never absolute -- used for any
    path recorded into a public-facing output file (path-leak avoidance)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_status_porcelain() -> str:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout
    except Exception as e:
        return f"<git status failed: {e}>"


def log(msg: str) -> None:
    print(f"[clean-run] {msg}", flush=True)


class PhaseTimer:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def time(self, name: str):
        return _PhaseCtx(self, name)


class _PhaseCtx:
    def __init__(self, timer: PhaseTimer, name: str) -> None:
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        log(f"START  {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.t0
        status = "FAILED" if exc_type else "OK"
        log(f"{status}  {self.name} ({elapsed:.1f}s)")
        self.timer.rows.append({"phase": self.name, "seconds": round(elapsed, 3), "status": status})
        return False


# ---------------------------------------------------------------------------
# Phase 1: isolated clean-run directory
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir: Path, reset: bool, verify_only: bool, skip_training: bool) -> None:
    subdirs = [
        "environment", "baselines/models", "baselines/results", "baselines/diagnostics",
        "corrected_ud_transfer", "statistical_tests", "human_reference",
        "human_model_evaluation", "comparisons", "logs",
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        if reset:
            log(f"--reset given: clearing existing contents of {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
        elif verify_only or skip_training:
            log(f"Reusing non-empty existing output dir {output_dir} ({'--verify-only' if verify_only else '--skip-training'})")
        else:
            raise SystemExit(
                f"REFUSING to run: output directory {output_dir} already exists and is not "
                f"empty. The final authoritative clean run must start from an empty directory "
                f"(this prevents already-committed or stale prior clean-run artifacts from "
                f"silently leaking into new results). Pass --reset to explicitly wipe it, "
                f"--skip-training to reuse existing clean-run models/predictions for debugging "
                f"iteration only (not for the authoritative run), or point --output-dir at a "
                f"new empty path."
            )
    for sd in subdirs:
        (output_dir / sd).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 3: environment / determinism
# ---------------------------------------------------------------------------

def scan_for_path_leaks(root: Path, patterns: list[str] = PATH_LEAK_PATTERNS) -> dict:
    """Scan every text-ish file under root for absolute-path / hostname
    leaks. Returns {"files_checked": n, "hits": [{"file":..., "pattern":...}]}.
    Pure function (no side effects) so it is directly unit-testable."""
    exts = {".json", ".md", ".txt", ".csv"}
    files_checked = 0
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        files_checked += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pat in patterns:
            if pat in text:
                hits.append({"file": rel(p), "pattern": pat})
    return {"files_checked": files_checked, "hits": hits}


def fix_path_leak(path: Path) -> bool:
    """Best-effort fix: replace absolute repo-root paths with repo-relative
    POSIX paths. Returns True if the file was modified."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    original = text
    root_str_variants = [str(ROOT) + "\\", str(ROOT).replace("\\", "/") + "/", str(ROOT)]
    for variant in root_str_variants:
        text = text.replace(variant, "<repo_root>/")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def collect_environment(output_dir: Path, seed: int, git_status_before: str) -> dict:
    try:
        import importlib.metadata as im
        pkg_versions = {}
        for pkg in ("numpy", "openpyxl", "python-docx"):
            try:
                pkg_versions[pkg] = im.version(pkg)
            except im.PackageNotFoundError:
                pkg_versions[pkg] = None
    except Exception:
        pkg_versions = {}

    input_checksums = {name: sha256_of(p) for name, p in FROZEN_INPUTS.items()}
    script_checksums = {name: sha256_of(SCRIPTS_DIR / name) for name in REUSED_SCRIPT_FILES}

    env = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "package_versions": pkg_versions,
        "locale": list(locale.getlocale()),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "repo_commit_hash": git_commit_hash(),
        "git_status_before_run": git_status_before,
        "seed": seed,
        "numpy_seed_mechanism": "np.random.default_rng(seed) (no legacy global numpy seeding)",
        "model_training_seed": seed,
        "split_seed_note": "Panyo train/dev/test split is a frozen, pre-existing artifact (guidelines/data_split_decisions.md); no split is performed by this pipeline.",
        "bootstrap_seed": seed,
        "bootstrap_n_resamples": 10_000,
        "frozen_input_checksums_sha256": {k: v for k, v in input_checksums.items()},
        "frozen_input_checksum_status": "first run of this clean-run directory: recorded as baseline (no prior recorded checksum table existed to compare against)",
        "reused_script_checksums_sha256": script_checksums,
    }
    return env


# ---------------------------------------------------------------------------
# Phase 4: baseline models (train / predict / mapping / canonical metrics /
# seen-unseen / boundary)
# ---------------------------------------------------------------------------

def run_train_and_predict_baselines(output_dir: Path, seed: int, skip_training: bool) -> dict:
    clean_models = output_dir / "baselines" / "models"
    clean_results = output_dir / "baselines" / "results"

    import train_baselines
    import predict_baselines

    train_baselines.MODELS = clean_models
    train_baselines.RESULTS = clean_results
    predict_baselines.MODELS = clean_models
    predict_baselines.RESULTS = clean_results

    if seed != train_baselines.SEED:
        log(f"NOTE: --seed={seed} differs from repo convention SEED=42 used by "
            f"train_baselines.py/perceptron_tagger.py; per task spec, sticking with "
            f"the existing repo convention (42) for model training determinism.")

    n_expected_models = 8
    already_trained = clean_models.exists() and len(list(clean_models.glob("*.json"))) >= n_expected_models
    if skip_training and already_trained:
        log(f"--skip-training: reusing {len(list(clean_models.glob('*.json')))} existing model JSONs in {clean_models}")
    else:
        train_baselines._log_lines.clear()
        train_baselines.main()

    already_predicted = clean_results.exists() and len(list(clean_results.glob("*_predictions.conllu"))) >= 10
    if skip_training and already_predicted:
        log(f"--skip-training: reusing existing prediction files in {clean_results}")
    else:
        predict_baselines.main()

    return {"models_dir": clean_models, "results_dir": clean_results}


def run_ud_mapping_diagnostic(output_dir: Path) -> Path:
    clean_results = output_dir / "baselines" / "results"
    clean_corrected = output_dir / "corrected_ud_transfer"

    import diagnostic_utils
    import diagnostic_ud_mapping

    diagnostic_utils.RESULTS = clean_results
    diagnostic_ud_mapping.RESULTS = clean_results
    diagnostic_ud_mapping.DIAG = clean_corrected

    diagnostic_ud_mapping.run()
    return clean_corrected


def run_canonical_silver_evaluation(output_dir: Path) -> Path:
    clean_results = output_dir / "baselines" / "results"
    import evaluate_baselines
    evaluate_baselines.RESULTS = clean_results
    evaluate_baselines.main()
    return clean_results


def run_seen_unseen_and_boundary(output_dir: Path) -> Path:
    clean_results = output_dir / "baselines" / "results"
    clean_diag = output_dir / "baselines" / "diagnostics"

    import diagnostic_utils
    import diagnostic_seen_unseen
    import diagnostic_boundary_analysis

    diagnostic_utils.RESULTS = clean_results
    diagnostic_seen_unseen.DIAG = clean_diag
    diagnostic_boundary_analysis.DIAG = clean_diag

    diagnostic_seen_unseen.run()
    diagnostic_boundary_analysis.run()
    return clean_diag


# ---------------------------------------------------------------------------
# Phase 5: statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(output_dir: Path, seed: int) -> list[dict]:
    clean_results = output_dir / "baselines" / "results"
    clean_corrected = output_dir / "corrected_ud_transfer"
    clean_stats = output_dir / "statistical_tests"

    import statistical_tests
    statistical_tests.RESULTS_BASELINES = clean_results
    statistical_tests.RESULTS_DIAG = clean_corrected
    statistical_tests.load_model_tokens.cache_clear()

    rows = statistical_tests.run(
        splits=["dev", "test"], seed=seed, n_boot=10_000, alpha=0.05, out_dir=clean_stats,
    )
    return rows


# ---------------------------------------------------------------------------
# Phase 6: human reference reconstruction
# ---------------------------------------------------------------------------

def run_human_reference_reconstruction(output_dir: Path) -> dict:
    clean_human_ref = output_dir / "human_reference"
    import build_test_only_adjudicated_reference as ref
    result = ref.run(out_dir=clean_human_ref)

    # Task explicitly asks for a literally-named command_log.txt in addition
    # to the script's own reference_build_command_log.txt (kept unmodified).
    src = clean_human_ref / "reference_build_command_log.txt"
    dst = clean_human_ref / "command_log.txt"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Phase 7: model evaluation on human reference (clean-run predictions only)
# ---------------------------------------------------------------------------

def _build_clean_eval_models_dict(evalhuman, output_dir: Path) -> dict:
    clean_baselines_models = output_dir / "baselines" / "models"
    clean_baselines_results = output_dir / "baselines" / "results"
    clean_corrected = output_dir / "corrected_ud_transfer"
    clean_mapping_csv = clean_corrected / "ud_transfer_mapping_results.csv"
    clean_baselines_overall = clean_baselines_results / "overall_results.csv"

    new_models = copy.deepcopy(evalhuman.MODELS)
    redirect = {
        "global_majority": {
            "pred_path": clean_baselines_results / "majority_global_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "majority_global.json",
            "silver_csv": clean_baselines_overall, "silver_key_col": "model", "silver_key": "majority_global",
        },
        "lang_aware_majority": {
            "pred_path": clean_baselines_results / "majority_lang_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "majority_lang.json",
            "silver_csv": clean_baselines_overall, "silver_key_col": "model", "silver_key": "majority_lang",
        },
        "joint_perceptron": {
            "pred_path": clean_baselines_results / "perceptron_joint_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_joint.json",
            "silver_csv": clean_baselines_overall, "silver_key_col": "model", "silver_key": "perceptron_joint",
        },
        "pangyo_pipeline_perceptron": {
            "pred_path": clean_baselines_results / "perceptron_pipeline_panyo_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_pipeline_panyo_ko.json",
            "silver_csv": clean_baselines_overall, "silver_key_col": "model", "silver_key": "perceptron_pipeline_panyo",
        },
        "ud_transfer_no_mapping": {
            "pred_path": clean_baselines_results / "perceptron_pipeline_ud_transfer_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_pipeline_ud_ko_kaist.json",
            "silver_csv": clean_baselines_overall, "silver_key_col": "model", "silver_key": "perceptron_pipeline_ud_transfer",
        },
        "ud_transfer_conservative": {
            "pred_path": clean_corrected / "ud_transfer_conservative_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_pipeline_ud_ko_kaist.json",
            "silver_csv": clean_mapping_csv, "silver_key_col": "scheme", "silver_key": "conservative",
        },
        "ud_transfer_panyo_coarse": {
            "pred_path": clean_corrected / "ud_transfer_panyo_coarse_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_pipeline_ud_ko_kaist.json",
            "silver_csv": clean_mapping_csv, "silver_key_col": "scheme", "silver_key": "panyo_coarse",
        },
        "ud_transfer_ko_oriented": {
            "pred_path": clean_corrected / "ud_transfer_ko_oriented_test_predictions.conllu",
            "artifact_path": clean_baselines_models / "perceptron_pipeline_ud_ko_kaist.json",
            "silver_csv": clean_mapping_csv, "silver_key_col": "scheme", "silver_key": "ko_oriented",
        },
    }
    for key, overrides in redirect.items():
        new_models[key].update(overrides)
    return new_models


def run_human_model_evaluation(output_dir: Path) -> dict:
    clean_human_ref = output_dir / "human_reference"
    clean_human_eval = output_dir / "human_model_evaluation"

    import evaluate_models_on_test_only_human_reference as evalhuman

    evalhuman.MODELS = _build_clean_eval_models_dict(evalhuman, output_dir)
    evalhuman.MODEL_ORDER = list(evalhuman.MODELS.keys())
    evalhuman.HUMAN_REF_CSV = clean_human_ref / "final_human_reference_tokens.csv"

    result = evalhuman.run(out_dir=clean_human_eval)

    n_models_ok = len(result["aligned_by_model"])
    for k, aligned in result["aligned_by_model"].items():
        if len(aligned) != 1879:
            raise RuntimeError(f"Model {k!r}: expected 1879 aligned tokens, got {len(aligned)}")
    if result["excluded"]:
        raise RuntimeError(f"{len(result['excluded'])} model(s) excluded from human evaluation: {result['excluded']}")

    src = clean_human_eval / "model_evaluation_command_log.txt"
    dst = clean_human_eval / "command_log.txt"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    log(f"Human-reference model evaluation: {n_models_ok}/{len(evalhuman.MODELS)} models aligned exactly 1879/1879.")
    return result


# ---------------------------------------------------------------------------
# Phase 8: compare clean-run outputs against authoritative committed outputs
# ---------------------------------------------------------------------------

def _compare_overall_results(auth_path: Path, clean_path: Path, models: list[str], artifact: str) -> list[dict]:
    import clean_run_compare as cc
    auth_rows = cc.index_csv_rows(cc.read_csv_rows(auth_path), ["model", "split"])
    clean_rows = cc.index_csv_rows(cc.read_csv_rows(clean_path), ["model", "split"])
    fields = ["tokens", "correct", "accuracy", "non_punct_tokens", "non_punct_correct",
              "non_punct_accuracy", "macro_f1", "weighted_f1"]
    int_fields = {"tokens", "correct", "non_punct_tokens", "non_punct_correct"}
    rows = []
    for model in models:
        for split in ("dev", "test"):
            a = auth_rows.get((model, split))
            c = clean_rows.get((model, split))
            if a is None:
                continue  # not an authoritative row for this model/split combo (e.g. mapping schemes live elsewhere)
            for f in fields:
                tol = 0.0 if f in int_fields else 1e-10
                rows.append(cc.compare_row(
                    f"{artifact}:{model}:{split}", f,
                    a.get(f) if a else None, c.get(f) if c else None, tolerance=tol,
                ))
    return rows


def _compare_mapping_results(auth_path: Path, clean_path: Path) -> list[dict]:
    import clean_run_compare as cc
    auth_rows = cc.index_csv_rows(cc.read_csv_rows(auth_path), ["scheme", "split"])
    clean_rows = cc.index_csv_rows(cc.read_csv_rows(clean_path), ["scheme", "split"])
    fields = ["accuracy", "non_punct_accuracy", "macro_f1", "weighted_f1", "total", "correct"]
    int_fields = {"total", "correct"}
    rows = []
    for (scheme, split), a in auth_rows.items():
        c = clean_rows.get((scheme, split))
        for f in fields:
            tol = 0.0 if f in int_fields else 1e-10
            rows.append(cc.compare_row(
                f"ud_transfer_mapping:{scheme}:{split}", f, a.get(f), c.get(f) if c else None, tolerance=tol,
            ))
    return rows


UD_TRANSFER_MODEL_NAMES = {"perceptron_pipeline_ud_transfer", "ud_transfer_panyo_coarse", "ud_transfer_ko_oriented"}


def _compare_statistical_tests(stale_auth_path: Path, corrected_auth_path: Path, clean_path: Path) -> list[dict]:
    import clean_run_compare as cc
    stale_rows = cc.index_csv_rows(cc.read_csv_rows(stale_auth_path), ["split", "model_a", "model_b"])
    corrected_rows = cc.index_csv_rows(cc.read_csv_rows(corrected_auth_path), ["split", "model_a", "model_b"])
    clean_rows = cc.index_csv_rows(cc.read_csv_rows(clean_path), ["split", "model_a", "model_b"])
    exact_int_fields = ["total_tokens", "n_documents", "both_correct", "both_wrong",
                         "a_only_correct", "b_only_correct", "discordant_pairs"]
    float_fields = ["mcnemar_exact_p", "holm_adjusted_p", "observed_acc_diff_a_minus_b",
                     "bootstrap_mean_diff_a_minus_b", "ci95_low", "ci95_high"]
    bool_fields = ["significant_after_holm", "ci95_contains_zero"]
    rows = []
    for key, stale in stale_rows.items():
        involves_ud = key[1] in UD_TRANSFER_MODEL_NAMES or key[2] in UD_TRANSFER_MODEL_NAMES
        a = corrected_rows.get(key) if involves_ud else stale
        source_note = (
            "compared against outputs/corrected_ud_transfer/statistical_tests/paired_comparisons.csv "
            "(the already-committed, already-regenerated counterpart), not the stale "
            "outputs/statistical_tests/paired_comparisons.csv, because this pair involves the "
            "UD-transfer model whose lang_override bug fix predates that file's last regeneration."
            if involves_ud else ""
        )
        c = clean_rows.get(key)
        artifact = f"statistical_tests:{key[0]}:{key[1]}_vs_{key[2]}"
        for f in exact_int_fields:
            rows.append(cc.compare_row(artifact, f, a.get(f) if a else None, c.get(f) if c else None,
                                        tolerance=0.0, explanation=source_note))
        for f in float_fields:
            # Bootstrap results: exact equality expected since seed(42) + n_boot(10000)
            # + identical statistical_tests.py implementation are all reused as-is;
            # a nonzero diff here would indicate a real, investigatable discrepancy.
            rows.append(cc.compare_row(artifact, f, a.get(f) if a else None, c.get(f) if c else None,
                                        tolerance=0.0, explanation=source_note))
        for f in bool_fields:
            rows.append(cc.compare_row(artifact, f, a.get(f) if a else None, c.get(f) if c else None,
                                        tolerance=0.0, explanation=source_note))
    return rows


def _compare_human_reference(output_dir: Path) -> list[dict]:
    import clean_run_compare as cc
    clean_json = output_dir / "human_reference" / "human_silver_metrics.json"
    auth_json = AUTH_HUMAN_REF_METRICS_JSON
    auth = cc.read_json(auth_json)
    clean = cc.read_json(clean_json)
    rows = []
    if auth is None or clean is None:
        rows.append(cc.compare_row("human_reference", "human_silver_metrics.json",
                                    "present" if auth else None, "present" if clean else None))
        return rows

    for set_name in ("Representative", "Diagnostic", "Combined"):
        a = auth.get("human_vs_canonical", {}).get(set_name, {})
        c = clean.get("human_vs_canonical", {}).get(set_name, {})
        for f in ("n_sentences", "n_tokens", "correct", "non_punct_tokens", "non_punct_correct"):
            rows.append(cc.compare_row(f"human_reference:{set_name}", f, a.get(f), c.get(f), tolerance=0.0))
        for f in ("agreement", "non_punct_agreement", "macro_f1", "weighted_f1", "kappa",
                   "bootstrap_ci_low", "bootstrap_ci_high"):
            rows.append(cc.compare_row(f"human_reference:{set_name}", f, a.get(f), c.get(f), tolerance=1e-6,
                                        explanation="bootstrap-derived value: tolerance 1e-6 rather than exact "
                                                     "because it is itself a Monte Carlo estimate percentile read "
                                                     "off the same seeded 10000-resample distribution." if "bootstrap" in f else ""))

    a_adj = auth.get("adjudication_summary", {})
    c_adj = clean.get("adjudication_summary", {})
    for f in ("p1_selected", "p2_selected", "third_label", "post_audit_correction", "agreed",
              "final_tokenization_issue_yes_total"):
        rows.append(cc.compare_row("human_reference:adjudication_summary", f, a_adj.get(f), c_adj.get(f), tolerance=0.0))

    a_pre = auth.get("pre_adjudication", {})
    c_pre = clean.get("pre_adjudication", {})
    for set_name in ("Representative", "Diagnostic", "Combined"):
        a = a_pre.get(set_name, {})
        c = c_pre.get(set_name, {})
        rows.append(cc.compare_row(f"human_reference:pre_adjudication:{set_name}", "disagreements",
                                    a.get("disagreements"), c.get("disagreements"), tolerance=0.0))
        rows.append(cc.compare_row(f"human_reference:pre_adjudication:{set_name}", "agreement",
                                    a.get("agreement"), c.get("agreement"), tolerance=1e-9))

    return rows


def _compare_model_evaluation(output_dir: Path) -> list[dict]:
    import clean_run_compare as cc
    auth_rows = cc.index_csv_rows(cc.read_csv_rows(AUTH_HUMAN_MODEL_EVAL_REP_CSV), ["model"])
    clean_rows = cc.index_csv_rows(
        cc.read_csv_rows(output_dir / "human_model_evaluation" / "model_metrics_representative.csv"), ["model"]
    )
    fields = ["n_tokens", "correct", "accuracy", "non_punct_tokens", "non_punct_correct",
              "non_punct_accuracy", "macro_f1", "weighted_f1", "kappa"]
    int_fields = {"n_tokens", "correct", "non_punct_tokens", "non_punct_correct"}
    rows = []
    for (model,), a in auth_rows.items():
        c = clean_rows.get((model,))
        for f in fields:
            tol = 0.0 if f in int_fields else 1e-9
            rows.append(cc.compare_row(f"model_evaluation:{model}", f, a.get(f), c.get(f) if c else None, tolerance=tol))

    auth_rank = cc.index_csv_rows(cc.read_csv_rows(AUTH_HUMAN_MODEL_EVAL_RANK_CSV), ["model"])
    clean_rank = cc.index_csv_rows(
        cc.read_csv_rows(output_dir / "human_model_evaluation" / "silver_vs_human_rank_comparison.csv"), ["model"]
    )
    for (model,), a in auth_rank.items():
        c = clean_rank.get((model,))
        rows.append(cc.compare_row(f"model_rank:{model}", "human_rank", a.get("human_rank"),
                                    c.get("human_rank") if c else None, tolerance=0.0))
    return rows


def run_phase8_comparison(output_dir: Path) -> tuple[list[dict], dict]:
    import clean_run_compare as cc

    all_rows: list[dict] = []
    canonical_models = ["majority_global", "majority_lang", "perceptron_joint", "perceptron_pipeline_panyo"]
    all_rows += _compare_overall_results(
        AUTH_BASELINES_OVERALL, output_dir / "baselines" / "results" / "overall_results.csv",
        canonical_models, "canonical_silver",
    )
    all_rows += _compare_overall_results(
        AUTH_CORRECTED_UD_OVERALL, output_dir / "baselines" / "results" / "overall_results.csv",
        ["perceptron_pipeline_ud_transfer"], "corrected_ud_transfer_no_mapping",
    )
    all_rows += _compare_mapping_results(
        AUTH_UD_MAPPING_RESULTS, output_dir / "corrected_ud_transfer" / "ud_transfer_mapping_results.csv",
    )
    all_rows += _compare_statistical_tests(
        AUTH_STATISTICAL_TESTS, AUTH_CORRECTED_STATISTICAL_TESTS,
        output_dir / "statistical_tests" / "paired_comparisons.csv",
    )
    all_rows += _compare_human_reference(output_dir)
    all_rows += _compare_model_evaluation(output_dir)

    # Prediction equality: compare gold+pred label sequences token-for-token
    # for every canonical model's test predictions (exact equality expected).
    pred_eq_rows = _compare_prediction_equality(output_dir)
    all_rows += pred_eq_rows

    comparisons_dir = output_dir / "comparisons"
    cc.write_comparison_csv(
        [r for r in all_rows if r["artifact"].startswith("canonical_silver") or r["artifact"].startswith("corrected_ud_transfer_no_mapping") or r["artifact"].startswith("ud_transfer_mapping")],
        comparisons_dir / "metric_comparison.csv",
    )
    cc.write_comparison_csv([r for r in all_rows if r["artifact"].startswith("statistical_tests")],
                             comparisons_dir / "statistical_test_comparison.csv")
    cc.write_comparison_csv([r for r in all_rows if r["artifact"].startswith("human_reference")],
                             comparisons_dir / "human_reference_comparison.csv")
    cc.write_comparison_csv([r for r in all_rows if r["artifact"].startswith("model_evaluation")],
                             comparisons_dir / "model_evaluation_comparison.csv")
    cc.write_comparison_csv([r for r in all_rows if r["artifact"].startswith("model_rank")],
                             comparisons_dir / "model_rank_comparison.csv")
    cc.write_comparison_csv(pred_eq_rows, comparisons_dir / "prediction_equality.csv")

    summary = cc.summarize(all_rows)
    (comparisons_dir / "comparison_summary.json").write_text(
        json.dumps({"counts_by_status": summary, "n_rows": len(all_rows)}, indent=2), encoding="utf-8"
    )

    discrepancy_lines = ["Final Clean Reproducibility Run -- Discrepancy Report", "=" * 88, ""]
    discrepancy_lines.append(f"Total comparison rows: {len(all_rows)}")
    for status, n in summary.items():
        discrepancy_lines.append(f"  {status}: {n}")
    discrepancy_lines.append("")
    blocking = [r for r in all_rows if r["status"] in cc.BLOCKING_STATUSES]
    if blocking:
        discrepancy_lines.append(f"BLOCKING discrepancies ({len(blocking)}):")
        for r in blocking:
            discrepancy_lines.append(
                f"  [{r['status']}] {r['artifact']} / {r['field']}: "
                f"authoritative={r['authoritative_value']!r} clean_run={r['clean_run_value']!r} "
                f"diff={r['absolute_difference']!r} tol={r['tolerance']!r} -- {r['explanation']}"
            )
    else:
        discrepancy_lines.append("No unexplained_discrepancy or missing_artifact rows found.")
    non_exact = [r for r in all_rows if r["status"] not in (cc.STATUS_EXACT,)]
    if non_exact:
        discrepancy_lines.append("")
        discrepancy_lines.append(f"All non-exact-match rows ({len(non_exact)}), including benign ones:")
        for r in non_exact:
            discrepancy_lines.append(
                f"  [{r['status']}] {r['artifact']} / {r['field']}: "
                f"authoritative={r['authoritative_value']!r} clean_run={r['clean_run_value']!r} "
                f"diff={r['absolute_difference']!r} -- {r['explanation']}"
            )
    (comparisons_dir / "discrepancy_report.txt").write_text("\n".join(discrepancy_lines) + "\n", encoding="utf-8")

    return all_rows, summary


def _compare_prediction_equality(output_dir: Path) -> list[dict]:
    """Exact equality of (gold, pred) label sequences for the four canonical
    models' test predictions, aligned by (sent_id, token_index)."""
    import clean_run_compare as cc
    sys.path.insert(0, str(SCRIPTS_DIR))
    from read_conllu import F_UPOS, F_MISC, iter_sentences

    def get_pred(misc: str) -> str:
        for part in misc.split("|"):
            if part.startswith("PredUPOS="):
                return part[9:]
        return ""

    def load(path: Path) -> dict:
        if not path.exists():
            return {}
        out = {}
        for sent in iter_sentences(path):
            sid = next((c[12:] for c in sent["comments"] if c.startswith("# sent_id = ")), "")
            real = [t for t in sent["tokens"] if t["is_real"]]
            for i, tok in enumerate(real):
                f = tok["fields"]
                out[(sid, i)] = (f[F_UPOS], get_pred(f[F_MISC]))
        return out

    rows = []
    for model in ["majority_global", "majority_lang", "perceptron_joint", "perceptron_pipeline_panyo",
                  "perceptron_pipeline_ud_transfer"]:
        auth_path = ROOT / "results" / "baselines" / f"{model}_test_predictions.conllu"
        clean_path = output_dir / "baselines" / "results" / f"{model}_test_predictions.conllu"
        auth = load(auth_path)
        clean = load(clean_path)
        if not auth or not clean:
            rows.append(cc.compare_row(f"prediction_equality:{model}", "n_tokens",
                                        len(auth) if auth else None, len(clean) if clean else None))
            continue
        n_match = sum(1 for k in auth if auth.get(k) == clean.get(k))
        n_total = len(auth)
        if model == "perceptron_pipeline_ud_transfer":
            # This model's committed results/baselines/ prediction file predates
            # the UD lang_override fix (see train_baselines.py docstring); the
            # clean run reflects the corrected pipeline, so a difference here is
            # EXPECTED, not an unexplained discrepancy -- explained explicitly.
            rows.append(cc.compare_row(
                f"prediction_equality:{model}", "matching_tokens", n_total, n_match,
                force_status=(cc.STATUS_EXACT if n_match == n_total else cc.STATUS_SERIALIZATION),
                explanation=(
                    "" if n_match == n_total else
                    "results/baselines/*_ud_transfer_* predates the UD-transfer lang_override "
                    "fix already baked into the current train_baselines.py; the clean run "
                    "correctly differs here by design (compared instead against "
                    "outputs/corrected_ud_transfer/ elsewhere in this report)."
                ),
            ))
        else:
            rows.append(cc.compare_row(f"prediction_equality:{model}", "matching_tokens",
                                        n_total, n_match, tolerance=0))
    return rows


# ---------------------------------------------------------------------------
# Manifest / README
# ---------------------------------------------------------------------------

def write_manifest(output_dir: Path, args: argparse.Namespace, env: dict, timer: PhaseTimer,
                    leak_scan: dict, comparison_summary: dict) -> None:
    manifest = {
        "command": "python scripts/run_final_clean_reproducibility_pipeline.py"
                   + (f" --output-dir {args.output_dir}" if str(args.output_dir) != str(DEFAULT_OUTPUT_DIR) else "")
                   + (f" --seed {args.seed}" if args.seed != SEED_DEFAULT else "")
                   + (" --verify-only" if args.verify_only else "")
                   + (" --skip-training" if args.skip_training else "")
                   + (" --reset" if args.reset else ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": rel(output_dir),
        "environment_summary": {
            "python_version": env["python_version"], "repo_commit_hash": env["repo_commit_hash"],
            "seed": env["seed"],
        },
        "phase_timings_seconds": timer.rows,
        "path_leak_scan": {"files_checked": leak_scan["files_checked"], "n_hits_found_and_fixed": leak_scan["n_hits_found_and_fixed"]},
        "comparison_summary": comparison_summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# Final Clean Reproducibility Run

Generated by `scripts/run_final_clean_reproducibility_pipeline.py`. Every
number under this directory was computed fresh in this run from frozen
source corpora and frozen human-annotation evidence -- never from any
already-committed model/prediction/metrics file. See `manifest.json` for
exact timings, checksums, and the comparison-against-authoritative-outputs
summary, and `comparisons/discrepancy_report.txt` for the full discrepancy
list.

## Directory layout
- `environment/` -- Python/OS/package versions, checksums, seeds
- `baselines/` -- 8 trained model JSONs, dev+test predictions, canonical
  silver metrics, seen/unseen + boundary diagnostics
- `corrected_ud_transfer/` -- 3 UD-transfer label-mapping variants
- `statistical_tests/` -- exact McNemar + Holm-Bonferroni + paired bootstrap
  over the same 6-pair family as the committed paper results
- `human_reference/` -- reconstructed test-only adjudicated human reference
- `human_model_evaluation/` -- 8-model evaluation against the human reference
- `comparisons/` -- diff against already-committed authoritative outputs
- `logs/` -- pipeline stdout/stderr and phase timing
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--verify-only", action="store_true",
                         help="Skip computation; only rerun the Phase 8 comparison against whatever is already in the output dir.")
    parser.add_argument("--skip-training", action="store_true",
                         help="Reuse already-trained clean-run models/predictions if present, for debugging iteration only. NOT for the final authoritative run.")
    parser.add_argument("--reset", action="store_true",
                         help="Wipe and recreate the output directory if it already exists and is non-empty.")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    git_status_before = git_status_porcelain()
    random.seed(args.seed)

    timer = PhaseTimer()

    with timer.time("phase1_ensure_output_dir"):
        ensure_output_dir(output_dir, args.reset, args.verify_only, args.skip_training)

    if not args.verify_only:
        with timer.time("phase3_environment_precheck"):
            env = collect_environment(output_dir, args.seed, git_status_before)
            (output_dir / "environment" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

        with timer.time("phase4_train_predict_baselines"):
            run_train_and_predict_baselines(output_dir, args.seed, args.skip_training)

        with timer.time("phase4_ud_mapping_diagnostic"):
            run_ud_mapping_diagnostic(output_dir)

        with timer.time("phase4_canonical_silver_evaluation"):
            run_canonical_silver_evaluation(output_dir)

        with timer.time("phase4_seen_unseen_and_boundary"):
            run_seen_unseen_and_boundary(output_dir)

        with timer.time("phase5_statistical_tests"):
            run_statistical_tests(output_dir, args.seed)

        with timer.time("phase6_human_reference_reconstruction"):
            run_human_reference_reconstruction(output_dir)

        with timer.time("phase7_human_model_evaluation"):
            run_human_model_evaluation(output_dir)
    else:
        env = json.loads((output_dir / "environment" / "environment.json").read_text(encoding="utf-8"))

    with timer.time("phase8_comparison"):
        all_rows, summary = run_phase8_comparison(output_dir)

    with timer.time("phase3_path_leak_scan"):
        leak_scan = scan_for_path_leaks(output_dir)
        n_fixed = 0
        for hit in leak_scan["hits"]:
            if fix_path_leak(ROOT / hit["file"]):
                n_fixed += 1
        if n_fixed:
            leak_scan_after = scan_for_path_leaks(output_dir)
        else:
            leak_scan_after = leak_scan
        leak_scan["n_hits_found_and_fixed"] = n_fixed
        leak_scan["hits_remaining_after_fix"] = leak_scan_after["hits"]

    write_manifest(output_dir, args, env, timer, leak_scan, summary)

    log("")
    log("=== Phase 8 comparison summary ===")
    for status, n in summary.items():
        log(f"  {status}: {n}")

    import clean_run_compare as cc
    blocking = [r for r in all_rows if r["status"] in cc.BLOCKING_STATUSES]
    if leak_scan.get("hits_remaining_after_fix"):
        log(f"UNRESOLVED path leaks: {leak_scan['hits_remaining_after_fix']}")
        return 1
    if blocking:
        log(f"FAILING: {len(blocking)} unexplained_discrepancy/missing_artifact row(s) remain after comparison. "
            f"See {output_dir / 'comparisons' / 'discrepancy_report.txt'}")
        return 1

    log("Clean reproducibility run completed with no unexplained discrepancies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
