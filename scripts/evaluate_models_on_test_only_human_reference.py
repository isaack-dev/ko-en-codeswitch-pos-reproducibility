"""
evaluate_models_on_test_only_human_reference.py - Evaluate existing frozen
model predictions against the test-only final adjudicated human reference
(scripts/build_test_only_adjudicated_reference.py output), for all 8
predefined models. Never retrains, tunes, or reruns any model.

Alignment requires Source_Sent_ID + Token_ID + Token form to all match
between the human reference and a model's prediction file; any mismatch
excludes that model with a documented reason rather than being silently
patched over.

Representative_60 is the primary held-out evaluation set (headline metrics +
statistical comparisons). Diagnostic_20 is a targeted, non-representative
analysis reported separately by diagnostic category. Combined is descriptive
only.

Run from repo root (after scripts/build_test_only_adjudicated_reference.py):
  python scripts/evaluate_models_on_test_only_human_reference.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import openpyxl
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).parent))
import build_test_only_adjudicated_reference as ref  # noqa: E402
from read_conllu import F_FORM, F_MISC, get_misc_lang, iter_sentences  # noqa: E402
from statistical_tests import exact_mcnemar_pvalue, holm_bonferroni, doc_level_bootstrap  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "outputs/human_validation/test_only/results"
HUMAN_REF_CSV = RESULTS_DIR / "final_human_reference_tokens.csv"
TEST_CONLLU = ref.TEST_CONLLU
CANONICAL_KEY_XLSX = ref.CANONICAL_KEY_XLSX

OUT_DIR = ROOT / "outputs/human_validation/test_only/model_evaluation"

SEED = 42
N_BOOTSTRAP = 10_000
ALPHA = 0.05
NEAR_CEILING_SILVER_THRESHOLD = 0.95
NEAR_CEILING_PRESERVED_RANK = 2  # "preserved" = still ranks in the top 2 under human evaluation

BASELINES_OVERALL_CSV = ROOT / "results/baselines/overall_results.csv"
CORRECTED_UD_OVERALL_CSV = ROOT / "outputs/corrected_ud_transfer/metrics/overall_results.csv"
UD_MAPPING_RESULTS_CSV = ROOT / "outputs/corrected_ud_transfer/metrics/ud_transfer_mapping_results.csv"

MODELS = {
    "global_majority": {
        "display": "Global Majority",
        "pred_path": ROOT / "results/baselines/majority_global_test_predictions.conllu",
        "artifact_path": ROOT / "models/baselines/majority_global.json",
        "silver_csv": BASELINES_OVERALL_CSV, "silver_key_col": "model", "silver_key": "majority_global",
    },
    "lang_aware_majority": {
        "display": "Language-Aware Majority",
        "pred_path": ROOT / "results/baselines/majority_lang_test_predictions.conllu",
        "artifact_path": ROOT / "models/baselines/majority_lang.json",
        "silver_csv": BASELINES_OVERALL_CSV, "silver_key_col": "model", "silver_key": "majority_lang",
    },
    "joint_perceptron": {
        "display": "Joint Perceptron",
        "pred_path": ROOT / "results/baselines/perceptron_joint_test_predictions.conllu",
        "artifact_path": ROOT / "models/baselines/perceptron_joint.json",
        "silver_csv": BASELINES_OVERALL_CSV, "silver_key_col": "model", "silver_key": "perceptron_joint",
    },
    "pangyo_pipeline_perceptron": {
        "display": "Pangyo-Internal Pipeline Perceptron",
        "pred_path": ROOT / "results/baselines/perceptron_pipeline_panyo_test_predictions.conllu",
        "artifact_path": ROOT / "models/baselines/perceptron_pipeline_panyo_ko.json",
        "silver_csv": BASELINES_OVERALL_CSV, "silver_key_col": "model", "silver_key": "perceptron_pipeline_panyo",
    },
    "ud_transfer_no_mapping": {
        "display": "Corrected UD-Transfer (no mapping)",
        "pred_path": ROOT / "outputs/corrected_ud_transfer/predictions/perceptron_pipeline_ud_transfer_test_predictions.conllu",
        "artifact_path": ROOT / "outputs/corrected_ud_transfer/models/perceptron_pipeline_ud_ko_kaist.json",
        "silver_csv": CORRECTED_UD_OVERALL_CSV, "silver_key_col": "model", "silver_key": "perceptron_pipeline_ud_transfer",
    },
    "ud_transfer_conservative": {
        "display": "Conservative Mapping",
        "pred_path": ROOT / "outputs/corrected_ud_transfer/predictions/ud_transfer_conservative_test_predictions.conllu",
        "artifact_path": ROOT / "outputs/corrected_ud_transfer/models/perceptron_pipeline_ud_ko_kaist.json",
        "silver_csv": UD_MAPPING_RESULTS_CSV, "silver_key_col": "scheme", "silver_key": "conservative",
    },
    "ud_transfer_panyo_coarse": {
        "display": "Pangyo-Coarse Mapping",
        "pred_path": ROOT / "outputs/corrected_ud_transfer/predictions/ud_transfer_panyo_coarse_test_predictions.conllu",
        "artifact_path": ROOT / "outputs/corrected_ud_transfer/models/perceptron_pipeline_ud_ko_kaist.json",
        "silver_csv": UD_MAPPING_RESULTS_CSV, "silver_key_col": "scheme", "silver_key": "panyo_coarse",
    },
    "ud_transfer_ko_oriented": {
        "display": "KO-Oriented Mapping",
        "pred_path": ROOT / "outputs/corrected_ud_transfer/predictions/ud_transfer_ko_oriented_test_predictions.conllu",
        "artifact_path": ROOT / "outputs/corrected_ud_transfer/models/perceptron_pipeline_ud_ko_kaist.json",
        "silver_csv": UD_MAPPING_RESULTS_CSV, "silver_key_col": "scheme", "silver_key": "ko_oriented",
    },
}
MODEL_ORDER = list(MODELS.keys())


def get_pred_upos(misc: str) -> Optional[str]:
    for part in misc.split("|"):
        if part.startswith("PredUPOS="):
            return part[len("PredUPOS="):]
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_human_reference() -> list[dict]:
    if not HUMAN_REF_CSV.exists():
        raise FileNotFoundError(
            f"{HUMAN_REF_CSV} not found -- run scripts/build_test_only_adjudicated_reference.py first."
        )
    with HUMAN_REF_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["Token_ID"] = int(r["Token_ID"])
    if len(rows) != 1879:
        raise ValueError(f"Expected 1879 human reference tokens, found {len(rows)}")
    return rows


def load_silver_test_accuracy(csv_path: Path, key_col: str, key_val: str) -> float:
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") == "test" and row.get(key_col) == key_val:
                return float(row["accuracy"])
    raise ValueError(f"No test-split row for {key_col}={key_val!r} in {csv_path}")


def load_model_predictions(path: Path) -> dict[tuple[str, int], tuple[str, str, str]]:
    """{(sent_id, token_id): (form, pred_upos, lang)}"""
    out: dict[tuple[str, int], tuple[str, str, str]] = {}
    for sent in iter_sentences(path):
        sid = None
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[len("# sent_id = "):]
        if sid is None:
            continue
        real = [t for t in sent["tokens"] if t["is_real"]]
        for i, t in enumerate(real):
            f = t["fields"]
            key = (sid, i + 1)
            if key in out:
                raise ValueError(f"{path}: duplicate token key {key}")
            out[key] = (f[F_FORM], get_pred_upos(f[F_MISC]) or "", get_misc_lang(f[F_MISC]))
    return out


def load_lang_and_boundary() -> dict[tuple[str, int], tuple[str, bool]]:
    """{(sent_id, token_id): (lang, is_boundary)} from the authoritative test
    CoNLL-U's own Lang= MISC field -- never derived from any model's output."""
    out: dict[tuple[str, int], tuple[str, bool]] = {}
    for sent in iter_sentences(TEST_CONLLU):
        sid = None
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[len("# sent_id = "):]
        if sid is None:
            continue
        real = [t for t in sent["tokens"] if t["is_real"]]
        langs = [get_misc_lang(t["fields"][F_MISC]) for t in real]
        for i in range(len(real)):
            prev_lang = langs[i - 1] if i > 0 else None
            next_lang = langs[i + 1] if i < len(langs) - 1 else None
            is_boundary = (prev_lang is not None and prev_lang != langs[i]) or \
                          (next_lang is not None and next_lang != langs[i])
            out[(sid, i + 1)] = (langs[i], is_boundary)
    return out


def load_diagnostic_categories() -> dict[str, str]:
    """{Sample_ID: Diagnostic_Category} for Diagnostic-set sentences."""
    wb = openpyxl.load_workbook(CANONICAL_KEY_XLSX, data_only=True)
    ws = wb["Canonical_Key_80"]
    header = [c.value for c in ws[1]]
    idx = {name: i for i, name in enumerate(header)}
    out: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[idx["Sample_ID"]] is None:
            continue
        cat = row[idx["Diagnostic_Category"]]
        if cat:
            out[row[idx["Sample_ID"]]] = cat
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_model(
    model_key: str, human_rows: list[dict], lang_boundary: dict[tuple[str, int], tuple[str, bool]]
) -> tuple[list[dict], list[str]]:
    cfg = MODELS[model_key]
    path = cfg["pred_path"]
    if not path.exists():
        return [], [f"prediction file not found: {path}"]
    pred_index = load_model_predictions(path)

    aligned: list[dict] = []
    errors: list[str] = []
    seen_keys = set()
    for r in human_rows:
        key = (r["Source_Sent_ID"], r["Token_ID"])
        if key in seen_keys:
            errors.append(f"duplicate key in human reference itself: {key}")
            continue
        seen_keys.add(key)
        rec = pred_index.get(key)
        if rec is None:
            errors.append(f"missing prediction for {key} ({r['Token']!r})")
            continue
        form, pred, _lang = rec
        if not pred:
            errors.append(f"empty PredUPOS at {key}")
            continue
        if form != r["Token"]:
            errors.append(f"token form mismatch at {key}: model={form!r} human={r['Token']!r}")
            continue
        if pred not in ref.UPOS17:
            errors.append(f"invalid PredUPOS {pred!r} at {key}")
            continue
        lang, is_boundary = lang_boundary.get(key, ("", False))
        aligned.append({**r, "Pred_UPOS": pred, "Lang": lang, "Is_Boundary": is_boundary})
    return aligned, errors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_model_metrics(aligned: list[dict]) -> dict:
    n = len(aligned)
    pairs = [(r["Final_Human_UPOS"], r["Pred_UPOS"]) for r in aligned]
    correct = sum(1 for g, p in pairs if g == p)
    non_punct = [r for r in aligned if r["Final_Human_UPOS"] != "PUNCT"]
    non_punct_correct = sum(1 for r in non_punct if r["Final_Human_UPOS"] == r["Pred_UPOS"])
    per_class = ref.compute_per_class(pairs)

    by_sentence: dict[str, list[bool]] = defaultdict(list)
    for r in aligned:
        by_sentence[r["Sample_ID"]].append(r["Final_Human_UPOS"] == r["Pred_UPOS"])
    sample_ids = sorted(by_sentence)
    correct_by_sent = [sum(by_sentence[s]) for s in sample_ids]
    total_by_sent = [len(by_sentence[s]) for s in sample_ids]
    lo, hi, boot_mean = ref.sentence_bootstrap_ci(correct_by_sent, total_by_sent, seed=SEED, n_boot=N_BOOTSTRAP)

    confusions = Counter((r["Final_Human_UPOS"], r["Pred_UPOS"]) for r in aligned if r["Final_Human_UPOS"] != r["Pred_UPOS"])

    def strat(subset: list[dict]) -> tuple[int, int, float]:
        c = sum(1 for r in subset if r["Final_Human_UPOS"] == r["Pred_UPOS"])
        n_sub = len(subset)
        return c, n_sub, (c / n_sub if n_sub else 0.0)

    tok_yes = [r for r in aligned if r["Final_Tokenization_Issue"] == "Yes"]
    tok_no = [r for r in aligned if r["Final_Tokenization_Issue"] == "No"]

    lang_acc = {}
    for lang in ("KO", "EN", "OTHER"):
        subset = [r for r in aligned if r["Lang"] == lang]
        lang_acc[lang] = strat(subset)

    boundary_acc = strat([r for r in aligned if r["Is_Boundary"]])
    non_boundary_acc = strat([r for r in aligned if not r["Is_Boundary"]])

    by_key = {(r["Sample_ID"], r["Token_ID"]): (r["Final_Human_UPOS"] == r["Pred_UPOS"]) for r in aligned}

    return {
        "n_sentences": len(sample_ids), "n_tokens": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "non_punct_tokens": len(non_punct), "non_punct_correct": non_punct_correct,
        "non_punct_accuracy": non_punct_correct / len(non_punct) if non_punct else 0.0,
        "macro_f1": ref.macro_f1(per_class), "weighted_f1": ref.weighted_f1(per_class),
        "kappa": ref.cohens_kappa(pairs),
        "per_class": per_class, "confusions": confusions,
        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "bootstrap_mean": boot_mean,
        "tokenization_issue_yes": strat(tok_yes), "tokenization_issue_no": strat(tok_no),
        "ko_accuracy": lang_acc["KO"], "en_accuracy": lang_acc["EN"], "other_accuracy": lang_acc["OTHER"],
        "boundary_accuracy": boundary_acc, "non_boundary_accuracy": non_boundary_acc,
        "correct_by_sentence": correct_by_sent, "total_by_sentence": total_by_sent, "sample_ids": sample_ids,
        "_by_key": by_key,
    }


def compute_diagnostic_category_metrics(aligned: list[dict], categories: dict[str, str]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in aligned:
        cat = categories.get(r["Sample_ID"], "UNKNOWN")
        by_cat[cat].append(r)
    out = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        correct = sum(1 for r in rows if r["Final_Human_UPOS"] == r["Pred_UPOS"])
        out[cat] = {"n_tokens": n, "correct": correct, "accuracy": correct / n if n else 0.0}
    return out


# ---------------------------------------------------------------------------
# Pairwise statistical comparison (Representative_60 only)
# ---------------------------------------------------------------------------

def pairwise_mcnemar_and_bootstrap(rep_metrics: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """rep_metrics: {model_key: metrics dict with _by_key, correct_by_sentence,
    total_by_sentence}. Only pairs where BOTH models are present are compared."""
    available = [k for k in MODEL_ORDER if k in rep_metrics]
    mcnemar_rows = []
    boot_rows = []
    pairs = list(combinations(available, 2))
    pvals = []
    for a, b in pairs:
        ma, mb = rep_metrics[a], rep_metrics[b]
        keys = ma["_by_key"].keys() & mb["_by_key"].keys()
        both_c = both_w = a_only = b_only = 0
        for k in keys:
            ca = ma["_by_key"][k]
            cb = mb["_by_key"][k]
            if ca and cb:
                both_c += 1
            elif not ca and not cb:
                both_w += 1
            elif ca and not cb:
                a_only += 1
            else:
                b_only += 1
        p = exact_mcnemar_pvalue(a_only, b_only)
        pvals.append(p)
        mcnemar_rows.append({
            "model_a": a, "model_b": b, "model_a_display": MODELS[a]["display"], "model_b_display": MODELS[b]["display"],
            "n_tokens": len(keys), "both_correct": both_c, "both_wrong": both_w,
            "a_only_correct": a_only, "b_only_correct": b_only, "discordant": a_only + b_only,
            "mcnemar_exact_p": p,
        })

        boot = doc_level_bootstrap(
            np.array(ma["correct_by_sentence"]), np.array(mb["correct_by_sentence"]),
            np.array(ma["total_by_sentence"]), n_boot=N_BOOTSTRAP, seed=SEED,
        )
        boot_rows.append({
            "model_a": a, "model_b": b, "model_a_display": MODELS[a]["display"], "model_b_display": MODELS[b]["display"],
            "observed_diff_a_minus_b": boot["observed_diff"], "bootstrap_mean_diff": boot["bootstrap_mean_diff"],
            "ci95_low": boot["ci_low"], "ci95_high": boot["ci_high"], "ci95_contains_zero": boot["ci_contains_zero"],
        })

    if pvals:
        adjusted, reject = holm_bonferroni(pvals, ALPHA)
        for row, adj_p, sig in zip(mcnemar_rows, adjusted, reject):
            row["holm_adjusted_p"] = adj_p
            row["significant_after_holm"] = sig
    return mcnemar_rows, boot_rows


# ---------------------------------------------------------------------------
# Silver-vs-human rank comparison
# ---------------------------------------------------------------------------

def compute_rank_comparison(silver_acc: dict[str, float], rep_metrics: dict[str, dict]) -> list[dict]:
    available = [k for k in MODEL_ORDER if k in rep_metrics]
    silver_ranked = sorted(available, key=lambda k: -silver_acc[k])
    human_ranked = sorted(available, key=lambda k: -rep_metrics[k]["accuracy"])
    silver_rank = {k: i + 1 for i, k in enumerate(silver_ranked)}
    human_rank = {k: i + 1 for i, k in enumerate(human_ranked)}

    rows = []
    for k in available:
        s_acc = silver_acc[k]
        h_acc = rep_metrics[k]["accuracy"]
        is_near_ceiling = s_acc >= NEAR_CEILING_SILVER_THRESHOLD
        preserved = is_near_ceiling and human_rank[k] <= NEAR_CEILING_PRESERVED_RANK
        rows.append({
            "model": k, "display": MODELS[k]["display"],
            "silver_test_accuracy": s_acc, "silver_rank": silver_rank[k],
            "human_representative_accuracy": h_acc, "human_rank": human_rank[k],
            "absolute_score_change": h_acc - s_acc,
            "rank_change": silver_rank[k] - human_rank[k],
            "is_near_ceiling_under_silver": is_near_ceiling,
            "near_ceiling_preserved_under_human_eval": preserved if is_near_ceiling else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

ALIGNED_CSV_FIELDS_BASE = [
    "Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Token_ID", "Token",
    "Final_Human_UPOS", "Canonical_UPOS", "Final_Tokenization_Issue",
]


def write_aligned_predictions_csv(aligned_by_model: dict[str, list[dict]], human_rows: list[dict], path: Path) -> None:
    by_key: dict[tuple[str, int], dict] = {}
    for r in human_rows:
        by_key[(r["Sample_ID"], r["Token_ID"])] = {
            "Set": r["Set"], "Sample_ID": r["Sample_ID"], "Source_Sent_ID": r["Source_Sent_ID"],
            "Document_ID": r["Document_ID"], "Token_ID": r["Token_ID"], "Token": r["Token"],
            "Final_Human_UPOS": r["Final_Human_UPOS"], "Canonical_UPOS": r["Canonical_UPOS"],
            "Final_Tokenization_Issue": r["Final_Tokenization_Issue"],
        }
    pred_by_key: dict[str, dict[tuple[str, int], str]] = {}
    for model_key, rows in aligned_by_model.items():
        pred_by_key[model_key] = {(r["Sample_ID"], r["Token_ID"]): r["Pred_UPOS"] for r in rows}

    fieldnames = ALIGNED_CSV_FIELDS_BASE + \
        [f"Pred_{k}" for k in aligned_by_model] + [f"Correct_{k}" for k in aligned_by_model]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(by_key):
            row = dict(by_key[key])
            for model_key in aligned_by_model:
                pred = pred_by_key[model_key].get(key)
                row[f"Pred_{model_key}"] = pred if pred is not None else ""
                row[f"Correct_{model_key}"] = ("" if pred is None else str(pred == row["Final_Human_UPOS"]))
            w.writerow(row)


def _flat_metrics_row(model_key: str, set_name: str, m: dict) -> dict:
    return {
        "model": model_key, "model_display": MODELS[model_key]["display"], "set": set_name,
        "n_sentences": m["n_sentences"], "n_tokens": m["n_tokens"], "correct": m["correct"],
        "accuracy": m["accuracy"], "non_punct_tokens": m["non_punct_tokens"],
        "non_punct_correct": m["non_punct_correct"], "non_punct_accuracy": m["non_punct_accuracy"],
        "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"], "kappa": m["kappa"],
        "bootstrap_ci_low": m["bootstrap_ci_low"], "bootstrap_ci_high": m["bootstrap_ci_high"],
        "bootstrap_mean": m["bootstrap_mean"],
        "tok_issue_yes_correct": m["tokenization_issue_yes"][0], "tok_issue_yes_n": m["tokenization_issue_yes"][1],
        "tok_issue_yes_accuracy": m["tokenization_issue_yes"][2],
        "tok_issue_no_correct": m["tokenization_issue_no"][0], "tok_issue_no_n": m["tokenization_issue_no"][1],
        "tok_issue_no_accuracy": m["tokenization_issue_no"][2],
    }


def write_metrics_csv(metrics_by_model: dict[str, dict], set_name: str, path: Path) -> None:
    rows = [_flat_metrics_row(k, set_name, m) for k, m in metrics_by_model.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_per_upos_csv(all_metrics: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "UPOS", "precision", "recall", "f1", "support"])
        for model_key, sets in all_metrics.items():
            for set_name, m in sets.items():
                for tag in sorted(m["per_class"]):
                    v = m["per_class"][tag]
                    w.writerow([model_key, set_name, tag, v["precision"], v["recall"], v["f1"], v["support"]])


def write_confusion_matrices_csv(all_metrics: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "Final_Human_UPOS", "Pred_UPOS", "count"])
        for model_key, sets in all_metrics.items():
            for set_name, m in sets.items():
                for (h, p), n in sorted(m["confusions"].items(), key=lambda kv: -kv[1]):
                    w.writerow([model_key, set_name, h, p, n])


def write_diagnostic_category_csv(cat_metrics_by_model: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "diagnostic_category", "n_tokens", "correct", "accuracy"])
        for model_key, cats in cat_metrics_by_model.items():
            for cat, v in sorted(cats.items()):
                w.writerow([model_key, cat, v["n_tokens"], v["correct"], v["accuracy"]])


def write_language_boundary_csv(all_metrics: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "KO_correct", "KO_n", "KO_accuracy", "EN_correct", "EN_n", "EN_accuracy",
                    "OTHER_correct", "OTHER_n", "OTHER_accuracy",
                    "boundary_correct", "boundary_n", "boundary_accuracy",
                    "non_boundary_correct", "non_boundary_n", "non_boundary_accuracy"])
        for model_key, sets in all_metrics.items():
            for set_name, m in sets.items():
                w.writerow([
                    model_key, set_name,
                    *m["ko_accuracy"], *m["en_accuracy"], *m["other_accuracy"],
                    *m["boundary_accuracy"], *m["non_boundary_accuracy"],
                ])


def write_tokenization_stratum_csv(all_metrics: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "tok_issue_yes_correct", "tok_issue_yes_n", "tok_issue_yes_accuracy",
                    "tok_issue_no_correct", "tok_issue_no_n", "tok_issue_no_accuracy"])
        for model_key, sets in all_metrics.items():
            for set_name, m in sets.items():
                w.writerow([model_key, set_name, *m["tokenization_issue_yes"], *m["tokenization_issue_no"]])


def write_bootstrap_ci_csv(all_metrics: dict[str, dict[str, dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "set", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_mean", "seed", "n_bootstrap"])
        for model_key, sets in all_metrics.items():
            m = sets.get("Representative")
            if m:
                w.writerow([model_key, "Representative", m["bootstrap_ci_low"], m["bootstrap_ci_high"], m["bootstrap_mean"], SEED, N_BOOTSTRAP])


def write_mcnemar_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model_a", "model_a_display", "model_b", "model_b_display", "n_tokens",
            "both_correct", "both_wrong", "a_only_correct", "b_only_correct", "discordant",
            "mcnemar_exact_p", "holm_adjusted_p", "significant_after_holm",
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})


def write_bootstrap_diff_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model_a", "model_a_display", "model_b", "model_b_display",
            "observed_diff_a_minus_b", "bootstrap_mean_diff", "ci95_low", "ci95_high", "ci95_contains_zero",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_rank_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_rows(ws, rows: list[list], bold_rows: set[int] = frozenset()) -> None:
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if r_idx in bold_rows:
                cell.font = Font(bold=True)


def write_evaluation_workbook(
    all_metrics: dict[str, dict[str, dict]], rank_rows: list[dict], mcnemar_rows: list[dict],
    boot_rows: list[dict], cat_metrics_by_model: dict[str, dict[str, dict]], excluded: dict[str, list[str]],
    path: Path,
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Executive_Summary")
    rows = [
        ["Test-only Model Evaluation Against the Final Human Reference"], [],
        ["Representative_60 is the PRIMARY held-out estimate. Diagnostic_20 is a TARGETED, "
         "non-representative analysis. Combined is DESCRIPTIVE ONLY."], [],
        ["Model", "Representative Accuracy", "Representative Kappa", "Diagnostic Accuracy", "Excluded?"],
    ]
    for k in MODEL_ORDER:
        if k in excluded:
            rows.append([MODELS[k]["display"], "", "", "", "; ".join(excluded[k][:2])])
            continue
        rep_m = all_metrics[k]["Representative"]
        diag_m = all_metrics[k]["Diagnostic"]
        rows.append([MODELS[k]["display"], rep_m["accuracy"], rep_m["kappa"], diag_m["accuracy"], ""])
    _write_rows(ws, rows, bold_rows={1, 5})

    for set_name, sheet_name in (("Representative", "Representative_Metrics"), ("Diagnostic", "Diagnostic_Metrics")):
        ws = wb.create_sheet(sheet_name)
        rows = [[f"{set_name} Metrics"], [],
                ["Model", "Tokens", "Correct", "Accuracy", "Non-PUNCT Accuracy", "Macro F1", "Weighted F1",
                 "Kappa", "Bootstrap 95% CI Low", "Bootstrap 95% CI High"]]
        for k in MODEL_ORDER:
            if k not in all_metrics:
                continue
            m = all_metrics[k][set_name]
            rows.append([MODELS[k]["display"], m["n_tokens"], m["correct"], m["accuracy"], m["non_punct_accuracy"],
                         m["macro_f1"], m["weighted_f1"], m["kappa"], m["bootstrap_ci_low"], m["bootstrap_ci_high"]])
        _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Silver_vs_Human_Ranks")
    rows = [["Silver-Reference vs. Human-Reference Rank Comparison"], [],
            ["Model", "Silver Test Accuracy", "Silver Rank", "Human Rep. Accuracy", "Human Rank",
             "Abs. Score Change", "Rank Change", "Near-Ceiling Under Silver", "Preserved Under Human Eval"]]
    for r in rank_rows:
        rows.append([r["display"], r["silver_test_accuracy"], r["silver_rank"], r["human_representative_accuracy"],
                     r["human_rank"], r["absolute_score_change"], r["rank_change"],
                     r["is_near_ceiling_under_silver"], r["near_ceiling_preserved_under_human_eval"]])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Pairwise_McNemar")
    rows = [["Pairwise Exact McNemar Tests (Representative_60, Holm-corrected)"], [],
            ["Model A", "Model B", "Discordant", "a_only", "b_only", "Exact p", "Holm-adjusted p", "Significant"]]
    for r in mcnemar_rows:
        rows.append([r["model_a_display"], r["model_b_display"], r["discordant"], r["a_only_correct"],
                     r["b_only_correct"], r["mcnemar_exact_p"], r.get("holm_adjusted_p"), r.get("significant_after_holm")])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Bootstrap_Differences")
    rows = [["Sentence/Document-level Paired Bootstrap of Accuracy Differences (Representative_60)"], [],
            ["Representative_60 uses one sentence per document, so sentence- and document-level "
             "resampling units coincide."], [],
            ["Model A", "Model B", "Observed Diff (A-B)", "Bootstrap Mean Diff", "95% CI Low", "95% CI High", "CI Contains Zero"]]
    for r in boot_rows:
        rows.append([r["model_a_display"], r["model_b_display"], r["observed_diff_a_minus_b"],
                     r["bootstrap_mean_diff"], r["ci95_low"], r["ci95_high"], r["ci95_contains_zero"]])
    _write_rows(ws, rows, bold_rows={1, 5})

    ws = wb.create_sheet("Per_UPOS")
    rows = [["Per-UPOS Precision / Recall / F1 (Representative_60)"], [], ["Model", "UPOS", "Precision", "Recall", "F1", "Support"]]
    for k in MODEL_ORDER:
        if k not in all_metrics:
            continue
        for tag in sorted(all_metrics[k]["Representative"]["per_class"]):
            v = all_metrics[k]["Representative"]["per_class"][tag]
            rows.append([MODELS[k]["display"], tag, v["precision"], v["recall"], v["f1"], v["support"]])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Diagnostic_Categories")
    rows = [["Diagnostic_20 Accuracy by Category (targeted analysis, not an unbiased estimate)"], [],
            ["Model", "Category", "Tokens", "Correct", "Accuracy"]]
    for k, cats in cat_metrics_by_model.items():
        for cat, v in sorted(cats.items()):
            rows.append([MODELS[k]["display"], cat, v["n_tokens"], v["correct"], v["accuracy"]])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Tokenization_Strata")
    rows = [["Accuracy by Final_Tokenization_Issue Stratum (Representative_60)"], [],
            ["Model", "Issue=Yes Correct", "Issue=Yes N", "Issue=Yes Acc", "Issue=No Correct", "Issue=No N", "Issue=No Acc"]]
    for k in MODEL_ORDER:
        if k not in all_metrics:
            continue
        m = all_metrics[k]["Representative"]
        rows.append([MODELS[k]["display"], *m["tokenization_issue_yes"], *m["tokenization_issue_no"]])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Confusions")
    rows = [["Confusion Counts (Representative_60)"], [], ["Model", "Final_Human_UPOS", "Pred_UPOS", "Count"]]
    for k in MODEL_ORDER:
        if k not in all_metrics:
            continue
        for (h, p), n in sorted(all_metrics[k]["Representative"]["confusions"].items(), key=lambda kv: -kv[1])[:30]:
            rows.append([MODELS[k]["display"], h, p, n])
    _write_rows(ws, rows, bold_rows={1, 3})

    ws = wb.create_sheet("Aligned_Predictions")
    ws.append(["See aligned_token_predictions.csv for the full 1,879-row aligned table (kept out of this "
               "workbook sheet to avoid an unwieldy >20-column sheet); this sheet documents its schema."])
    ws.append([])
    ws.append(ALIGNED_CSV_FIELDS_BASE + [f"Pred_{k}" for k in all_metrics] + [f"Correct_{k}" for k in all_metrics])

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_manifest(
    excluded: dict[str, list[str]], silver_acc: dict[str, float], all_metrics: dict[str, dict[str, dict]],
    rank_rows: list[dict], discrepancies: list[str], path: Path,
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/evaluate_models_on_test_only_human_reference.py",
        "git_commit": ref.git_commit_hash(),
        "python_version": platform.python_version(),
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP, "alpha": ALPHA,
        "models": {
            k: {
                "display": cfg["display"],
                "prediction_file": str(cfg["pred_path"].relative_to(ROOT)),
                "prediction_file_sha256": ref.sha256_of(cfg["pred_path"]) if cfg["pred_path"].exists() else None,
                "model_artifact": str(cfg["artifact_path"].relative_to(ROOT)) if cfg["artifact_path"].exists() else None,
                "model_artifact_sha256": ref.sha256_of(cfg["artifact_path"]) if cfg["artifact_path"].exists() else None,
                "silver_test_accuracy_source": str(cfg["silver_csv"].relative_to(ROOT)),
                "silver_test_accuracy": silver_acc.get(k),
                "excluded": k in excluded,
                "exclusion_reasons": excluded.get(k, []),
                "n_aligned_tokens": all_metrics[k]["Combined"]["n_tokens"] if k in all_metrics else 0,
            }
            for k, cfg in MODELS.items()
        },
        "rank_comparison": rank_rows,
        "discrepancies": discrepancies,
        "human_reference_source": str(HUMAN_REF_CSV.relative_to(ROOT)),
        "human_reference_sha256": ref.sha256_of(HUMAN_REF_CSV) if HUMAN_REF_CSV.exists() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_command_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} python scripts/evaluate_models_on_test_only_human_reference.py\n")


def write_summary_txt(
    all_metrics: dict[str, dict[str, dict]], rank_rows: list[dict], mcnemar_rows: list[dict],
    excluded: dict[str, list[str]], path: Path,
) -> None:
    lines = ["Test-only Model Evaluation -- Summary", "=" * 88, "",
             "Representative_60 = PRIMARY. Diagnostic_20 = TARGETED (not unbiased). Combined = DESCRIPTIVE.", ""]
    lines.append("## Representative_60 headline accuracy (sorted)")
    ranked = sorted((k for k in MODEL_ORDER if k in all_metrics), key=lambda k: -all_metrics[k]["Representative"]["accuracy"])
    for k in ranked:
        m = all_metrics[k]["Representative"]
        lines.append(f"  {MODELS[k]['display']:38s} acc={m['accuracy']:.4f}  kappa={m['kappa']:.4f}  "
                     f"95%CI=[{m['bootstrap_ci_low']:.4f}, {m['bootstrap_ci_high']:.4f}]")
    lines.append("")
    if excluded:
        lines.append("## Excluded models")
        for k, reasons in excluded.items():
            lines.append(f"  {MODELS[k]['display']}: {reasons[0] if reasons else 'unknown reason'} "
                         f"(+{len(reasons)-1} more)" if len(reasons) > 1 else f"  {MODELS[k]['display']}: {reasons[0] if reasons else 'unknown reason'}")
        lines.append("")
    lines.append("## Silver-vs-human rank changes")
    for r in sorted(rank_rows, key=lambda r: r["human_rank"]):
        lines.append(f"  {r['display']:38s} silver_rank={r['silver_rank']} -> human_rank={r['human_rank']} "
                     f"(Δrank={r['rank_change']:+d}, Δscore={r['absolute_score_change']:+.4f})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path) -> None:
    text = """# Test-only Model Evaluation Against the Final Human Reference

Generated by `scripts/evaluate_models_on_test_only_human_reference.py`. No
model was retrained, tuned, or rerun; existing frozen prediction files were
aligned to the final human reference by Source_Sent_ID + Token_ID + Token
form (all three required to match).

- **Representative_60** -- primary held-out evaluation; headline metrics and
  all pairwise statistical tests.
- **Diagnostic_20** -- targeted analysis by diagnostic category; NOT an
  unbiased corpus-wide estimate.
- **Combined** -- descriptive only.

See `model_evaluation_manifest.json` for exact prediction-file/model-artifact
checksums per model, `summary.txt` for a compact ranked overview, and
`Human_POS_Tagging_test_v1_Model_Evaluation.xlsx` for the full workbook.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> dict:
    human_rows = load_human_reference()
    lang_boundary = load_lang_and_boundary()
    diagnostic_categories = load_diagnostic_categories()

    aligned_by_model: dict[str, list[dict]] = {}
    excluded: dict[str, list[str]] = {}
    silver_acc: dict[str, float] = {}

    for k, cfg in MODELS.items():
        print(f"Aligning {cfg['display']} ...")
        aligned, errors = align_model(k, human_rows, lang_boundary)
        if errors or len(aligned) != 1879:
            reasons = list(errors)
            if len(aligned) != 1879:
                reasons.insert(0, f"only {len(aligned)}/1879 tokens aligned")
            excluded[k] = reasons
            print(f"  EXCLUDED: {reasons[0]}" + (f" (+{len(reasons)-1} more)" if len(reasons) > 1 else ""))
            continue
        aligned_by_model[k] = aligned
        try:
            silver_acc[k] = load_silver_test_accuracy(cfg["silver_csv"], cfg["silver_key_col"], cfg["silver_key"])
        except ValueError as e:
            excluded[k] = [f"silver accuracy lookup failed: {e}"]
            del aligned_by_model[k]
            print(f"  EXCLUDED: {excluded[k][0]}")
            continue
        print(f"  OK: {len(aligned)}/1879 aligned, silver_test_accuracy={silver_acc[k]:.4f}")

    all_metrics: dict[str, dict[str, dict]] = {}
    cat_metrics_by_model: dict[str, dict[str, dict]] = {}
    for k, aligned in aligned_by_model.items():
        rep = [r for r in aligned if r["Set"] == "Representative"]
        diag = [r for r in aligned if r["Set"] == "Diagnostic"]
        if len(rep) != 1063:
            raise ValueError(f"{k}: expected 1063 Representative aligned tokens, got {len(rep)}")
        if len(diag) != 816:
            raise ValueError(f"{k}: expected 816 Diagnostic aligned tokens, got {len(diag)}")
        all_metrics[k] = {
            "Representative": compute_model_metrics(rep),
            "Diagnostic": compute_model_metrics(diag),
            "Combined": compute_model_metrics(aligned),
        }
        cat_metrics_by_model[k] = compute_diagnostic_category_metrics(diag, diagnostic_categories)

    rep_metrics = {k: all_metrics[k]["Representative"] for k in all_metrics}
    mcnemar_rows, boot_rows = pairwise_mcnemar_and_bootstrap(rep_metrics)
    rank_rows = compute_rank_comparison(silver_acc, rep_metrics)

    discrepancies: list[str] = []
    for k, sets in all_metrics.items():
        if sets["Representative"]["n_tokens"] != 1063:
            discrepancies.append(f"{k}: Representative token count != 1063")
        if sets["Diagnostic"]["n_tokens"] != 816:
            discrepancies.append(f"{k}: Diagnostic token count != 816")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_aligned_predictions_csv(aligned_by_model, human_rows, out_dir / "aligned_token_predictions.csv")
    write_metrics_csv({k: v["Representative"] for k, v in all_metrics.items()}, "Representative", out_dir / "model_metrics_representative.csv")
    write_metrics_csv({k: v["Diagnostic"] for k, v in all_metrics.items()}, "Diagnostic", out_dir / "model_metrics_diagnostic.csv")
    write_metrics_csv({k: v["Combined"] for k, v in all_metrics.items()}, "Combined", out_dir / "model_metrics_combined_descriptive.csv")
    write_per_upos_csv(all_metrics, out_dir / "model_per_upos_metrics.csv")
    write_confusion_matrices_csv(all_metrics, out_dir / "model_confusion_matrices.csv")
    write_diagnostic_category_csv(cat_metrics_by_model, out_dir / "diagnostic_category_metrics.csv")
    write_language_boundary_csv(all_metrics, out_dir / "language_boundary_metrics.csv")
    write_tokenization_stratum_csv(all_metrics, out_dir / "tokenization_stratum_metrics.csv")
    write_bootstrap_ci_csv(all_metrics, out_dir / "representative_sentence_bootstrap_cis.csv")
    write_mcnemar_csv(mcnemar_rows, out_dir / "representative_pairwise_exact_mcnemar.csv")
    write_bootstrap_diff_csv(boot_rows, out_dir / "representative_pairwise_bootstrap_differences.csv")
    write_rank_comparison_csv(rank_rows, out_dir / "silver_vs_human_rank_comparison.csv")
    write_evaluation_workbook(
        all_metrics, rank_rows, mcnemar_rows, boot_rows, cat_metrics_by_model, excluded,
        out_dir / "Human_POS_Tagging_test_v1_Model_Evaluation.xlsx",
    )

    def _json_safe(all_m):
        out = {}
        for k, sets in all_m.items():
            out[k] = {}
            for set_name, m in sets.items():
                m2 = {kk: vv for kk, vv in m.items() if kk not in ("confusions", "_by_key")}
                m2["confusions"] = [{"human": h, "pred": p, "count": n} for (h, p), n in m["confusions"].items()]
                out[k][set_name] = m2
        return out

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP, "alpha": ALPHA,
        "excluded_models": excluded,
        "metrics": _json_safe(all_metrics),
        "diagnostic_category_metrics": cat_metrics_by_model,
        "pairwise_mcnemar": mcnemar_rows,
        "pairwise_bootstrap": boot_rows,
        "rank_comparison": rank_rows,
    }
    (out_dir / "model_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    write_manifest(excluded, silver_acc, all_metrics, rank_rows, discrepancies, out_dir / "model_evaluation_manifest.json")
    write_command_log(out_dir / "model_evaluation_command_log.txt")
    write_summary_txt(all_metrics, rank_rows, mcnemar_rows, excluded, out_dir / "summary.txt")
    write_readme(out_dir / "README.md")

    print(f"\n{len(aligned_by_model)}/{len(MODELS)} models evaluated; {len(excluded)} excluded.")
    print(f"Wrote outputs under {out_dir}")
    return {
        "aligned_by_model": aligned_by_model, "excluded": excluded, "silver_acc": silver_acc,
        "all_metrics": all_metrics, "mcnemar_rows": mcnemar_rows, "boot_rows": boot_rows,
        "rank_rows": rank_rows, "cat_metrics_by_model": cat_metrics_by_model, "discrepancies": discrepancies,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run()


if __name__ == "__main__":
    main()
