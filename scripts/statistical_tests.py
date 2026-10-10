"""
statistical_tests.py - Statistically correct paired model comparisons.

For each predefined model pair (the same pairs reported in
results/paper_ready/tables/paired_comparison_table.csv, mirrored from
scripts/diagnostic_significance.py), computes:

  1. An exact two-sided McNemar test on token-level correctness, using the
     exact binomial distribution over the discordant pairs (not the
     chi-square/continuity-correction approximation used previously).
  2. Holm-Bonferroni-adjusted p-values across the predefined comparisons,
     computed separately per split (dev, test) since each split is its own
     family of 6 comparisons.
  3. A document-level paired bootstrap (10,000 resamples, seed 42) of the
     accuracy difference (Model A - Model B), resampling whole test
     documents (doc_id) with replacement.

This script does not modify any training code, corpus files, prediction
files, or the LaTeX paper. All outputs are written under
outputs/statistical_tests/.

Run from repo root:
  python scripts/statistical_tests.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_UPOS, F_MISC, iter_sentences  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS_BASELINES = ROOT / "results" / "baselines"
RESULTS_DIAG = ROOT / "results" / "diagnostic_analysis"
OUT_DIR = ROOT / "outputs" / "statistical_tests"
BOOT_DIR = OUT_DIR / "bootstrap_distributions"

SEED = 42
N_BOOTSTRAP = 10_000
ALPHA = 0.05
SPLITS = ["dev", "test"]

# Display names matching results/paper_ready/tables/paired_comparison_table.csv
MODEL_DISPLAY_NAMES = {
    "majority_global": "Global Majority",
    "majority_lang": "Lang-Aware Majority",
    "perceptron_joint": "Joint Perceptron",
    "perceptron_pipeline_panyo": "Panyo-Pipeline Perceptron",
    "perceptron_pipeline_ud_transfer": "UD-Transfer Pipeline",
    "ud_transfer_panyo_coarse": "UD-Transfer + Panyo-Coarse",
    "ud_transfer_ko_oriented": "UD-Transfer + KO-Oriented",
}

# Model pairs exactly as reported in results/paper_ready/tables/paired_comparison_table.csv
# (== scripts/diagnostic_significance.py PAIRS + MAPPED_PAIRS_NAMES).
PREDEFINED_PAIRS = [
    ("perceptron_joint", "perceptron_pipeline_panyo"),
    ("majority_global", "majority_lang"),
    ("majority_lang", "perceptron_pipeline_ud_transfer"),
    ("perceptron_joint", "majority_lang"),
    ("perceptron_pipeline_ud_transfer", "ud_transfer_panyo_coarse"),
    ("perceptron_pipeline_ud_transfer", "ud_transfer_ko_oriented"),
]

TokenKey = tuple[str, str, int]  # (doc_id, sent_id, token_index_in_sentence)


# ---------------------------------------------------------------------------
# CoNLL-U loading / alignment
# ---------------------------------------------------------------------------

def get_pred_path(model: str, split: str) -> Path:
    candidates = [
        RESULTS_BASELINES / f"{model}_{split}_predictions.conllu",
        RESULTS_DIAG / f"{model}_{split}_predictions.conllu",
    ]
    for c in candidates:
        if c.exists():
            return c
    checked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"No prediction file found for model={model!r} split={split!r}. Checked:\n{checked}"
    )


def get_pred_upos(misc: str) -> str:
    for part in misc.split("|"):
        if part.startswith("PredUPOS="):
            return part[len("PredUPOS="):]
    return ""


def get_doc_id(comments: list[str]) -> str:
    for c in comments:
        if c.startswith("# doc_id = "):
            return c[len("# doc_id = "):]
    return ""


def get_sent_id(comments: list[str]) -> str:
    for c in comments:
        if c.startswith("# sent_id = "):
            return c[len("# sent_id = "):]
    return ""


@lru_cache(maxsize=None)
def load_model_tokens(model: str, split: str) -> dict[TokenKey, tuple[str, str, str]]:
    """Return {token_key: (doc_id, gold_upos, pred_upos)} for one model/split.

    Fails loudly (raises) on a sentence missing doc_id/sent_id, a token
    missing PredUPOS, or a duplicate token key within the file.
    The returned dict must be treated as read-only (cached across pairs).
    """
    path = get_pred_path(model, split)
    records: dict[TokenKey, tuple[str, str, str]] = {}
    for sent in iter_sentences(path):
        doc_id = get_doc_id(sent["comments"])
        sent_id = get_sent_id(sent["comments"])
        if not doc_id:
            raise ValueError(f"{path}: sentence missing '# doc_id = ' comment: {sent['comments']}")
        if not sent_id:
            raise ValueError(f"{path}: sentence missing '# sent_id = ' comment: {sent['comments']}")
        real_toks = [t for t in sent["tokens"] if t["is_real"]]
        for idx, tok in enumerate(real_toks):
            f = tok["fields"]
            key = (doc_id, sent_id, idx)
            if key in records:
                raise ValueError(f"{path}: duplicate token key {key}")
            gold = f[F_UPOS]
            pred = get_pred_upos(f[F_MISC])
            if not pred:
                raise ValueError(f"{path}: token {key} has no PredUPOS in MISC {f[F_MISC]!r}")
            records[key] = (doc_id, gold, pred)
    if not records:
        raise ValueError(f"{path}: no tokens loaded")
    return records


def validate_alignment(
    records_a: dict[TokenKey, tuple[str, str, str]],
    records_b: dict[TokenKey, tuple[str, str, str]],
    model_a: str,
    model_b: str,
    split: str,
) -> list[TokenKey]:
    """Verify both models cover identical token keys with matching gold labels.

    Returns the sorted list of shared keys. Raises ValueError with a
    detailed report if keys are missing on either side or gold labels
    disagree (which would indicate misaligned corpora, not just different
    predictions).
    """
    keys_a = set(records_a)
    keys_b = set(records_b)
    if keys_a != keys_b:
        missing_in_b = sorted(keys_a - keys_b)
        missing_in_a = sorted(keys_b - keys_a)
        msg = [f"Token key mismatch between {model_a!r} and {model_b!r} on split {split!r}:"]
        if missing_in_b:
            msg.append(
                f"  {len(missing_in_b)} key(s) present in {model_a!r} but missing from "
                f"{model_b!r}, e.g. {missing_in_b[:5]}"
            )
        if missing_in_a:
            msg.append(
                f"  {len(missing_in_a)} key(s) present in {model_b!r} but missing from "
                f"{model_a!r}, e.g. {missing_in_a[:5]}"
            )
        raise ValueError("\n".join(msg))

    keys = sorted(keys_a)
    gold_mismatches = [k for k in keys if records_a[k][1] != records_b[k][1]]
    if gold_mismatches:
        example = gold_mismatches[0]
        raise ValueError(
            f"Gold UPOS mismatch between {model_a!r} and {model_b!r} on split {split!r} "
            f"at {len(gold_mismatches)} token(s), e.g. key={example} "
            f"gold_a={records_a[example][1]!r} gold_b={records_b[example][1]!r}"
        )
    return keys


# ---------------------------------------------------------------------------
# Exact McNemar test
# ---------------------------------------------------------------------------

def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value via the exact binomial distribution.

    For the symmetric p=0.5 case this is equivalent to
    R's binom.test(min(a_only, b_only), a_only + b_only, 0.5,
    alternative="two.sided")$p.value, and to
    scipy.stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue:

        n = a_only + b_only
        k = min(a_only, b_only)
        p = 2 * P(X <= k),  X ~ Binomial(n, 0.5),  capped at 1.0

    Computed in log-space (via lgamma) so it does not overflow/underflow
    for large discordant-pair counts (tens of thousands of tokens).
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    log_half = math.log(0.5)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + n * log_half
        for i in range(k + 1)
    ]
    m = max(log_terms)
    log_one_tail = m + math.log(sum(math.exp(t - m) for t in log_terms))
    p = 2.0 * math.exp(log_one_tail)
    return min(p, 1.0)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm step-down adjustment. Returns (adjusted_pvalues, reject_null),
    both aligned to the input order (not sorted)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        val = min((m - rank) * pvalues[i], 1.0)
        running_max = max(running_max, val)
        adjusted[i] = running_max
    reject = [adjusted[i] < alpha for i in range(m)]
    return adjusted, reject


# ---------------------------------------------------------------------------
# Document-level paired bootstrap
# ---------------------------------------------------------------------------

def doc_level_bootstrap(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_tokens: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Document-level paired bootstrap of accuracy(A) - accuracy(B).

    correct_a, correct_b, n_tokens are parallel per-document arrays: the
    count of correctly-tagged tokens for model A / model B, and the total
    token count, for each test document. Resampling unit is the document
    (not token or sentence): each bootstrap replicate draws len(doc) doc
    indices with replacement and includes ALL tokens from each sampled
    document.
    """
    n_docs = len(correct_a)
    total_tokens = int(n_tokens.sum())
    observed_diff = float((correct_a.sum() - correct_b.sum()) / total_tokens)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_docs, size=(n_boot, n_docs))
    boot_a = correct_a[idx].sum(axis=1)
    boot_b = correct_b[idx].sum(axis=1)
    boot_n = n_tokens[idx].sum(axis=1)
    diffs = (boot_a - boot_b) / boot_n

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_documents": n_docs,
        "total_tokens": total_tokens,
        "observed_diff": observed_diff,
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_contains_zero": bool(ci_low <= 0.0 <= ci_high),
        "distribution": diffs,
    }


# ---------------------------------------------------------------------------
# Per-pair comparison
# ---------------------------------------------------------------------------

def compare_pair(
    model_a: str,
    model_b: str,
    split: str,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Compute McNemar counts + document-level bootstrap for one model pair
    on one split, using a single shared aligned token population for both."""
    records_a = load_model_tokens(model_a, split)
    records_b = load_model_tokens(model_b, split)
    keys = validate_alignment(records_a, records_b, model_a, model_b, split)

    both_c = both_w = a_only = b_only = 0
    doc_correct_a: dict[str, int] = defaultdict(int)
    doc_correct_b: dict[str, int] = defaultdict(int)
    doc_n: dict[str, int] = defaultdict(int)

    for key in keys:
        doc_id, gold, pred_a = records_a[key]
        _, _, pred_b = records_b[key]
        ca = gold == pred_a
        cb = gold == pred_b
        if ca and cb:
            both_c += 1
        elif not ca and not cb:
            both_w += 1
        elif ca and not cb:
            a_only += 1
        else:
            b_only += 1
        doc_correct_a[doc_id] += int(ca)
        doc_correct_b[doc_id] += int(cb)
        doc_n[doc_id] += 1

    total_tokens = len(keys)
    p_exact = exact_mcnemar_pvalue(a_only, b_only)

    doc_ids = sorted(doc_n)
    ca_arr = np.array([doc_correct_a[d] for d in doc_ids], dtype=np.int64)
    cb_arr = np.array([doc_correct_b[d] for d in doc_ids], dtype=np.int64)
    n_arr = np.array([doc_n[d] for d in doc_ids], dtype=np.int64)
    boot = doc_level_bootstrap(ca_arr, cb_arr, n_arr, n_boot=n_boot, seed=seed)

    return {
        "split": split,
        "model_a": model_a,
        "model_b": model_b,
        "total_tokens": total_tokens,
        "both_correct": both_c,
        "both_wrong": both_w,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant_pairs": a_only + b_only,
        "mcnemar_exact_p": p_exact,
        "n_documents": boot["n_documents"],
        "bootstrap": boot,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "split", "model_a", "model_a_display", "model_b", "model_b_display",
    "total_tokens", "n_documents",
    "both_correct", "both_wrong", "a_only_correct", "b_only_correct", "discordant_pairs",
    "mcnemar_exact_p", "holm_adjusted_p", "alpha", "significant_after_holm",
    "observed_acc_diff_a_minus_b", "bootstrap_mean_diff_a_minus_b",
    "ci95_low", "ci95_high", "ci95_contains_zero",
]


def row_to_flat(r: dict, alpha: float) -> dict:
    b = r["bootstrap"]
    return {
        "split": r["split"],
        "model_a": r["model_a"],
        "model_a_display": MODEL_DISPLAY_NAMES.get(r["model_a"], r["model_a"]),
        "model_b": r["model_b"],
        "model_b_display": MODEL_DISPLAY_NAMES.get(r["model_b"], r["model_b"]),
        "total_tokens": r["total_tokens"],
        "n_documents": r["n_documents"],
        "both_correct": r["both_correct"],
        "both_wrong": r["both_wrong"],
        "a_only_correct": r["a_only_correct"],
        "b_only_correct": r["b_only_correct"],
        "discordant_pairs": r["discordant_pairs"],
        "mcnemar_exact_p": r["mcnemar_exact_p"],
        "holm_adjusted_p": r["holm_adjusted_p"],
        "alpha": alpha,
        "significant_after_holm": r["significant_after_holm"],
        "observed_acc_diff_a_minus_b": b["observed_diff"],
        "bootstrap_mean_diff_a_minus_b": b["bootstrap_mean_diff"],
        "ci95_low": b["ci_low"],
        "ci95_high": b["ci_high"],
        "ci95_contains_zero": b["ci_contains_zero"],
    }


def write_csv(rows: list[dict], alpha: float, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(row_to_flat(r, alpha))


def write_json(rows: list[dict], alpha: float, meta: dict, path: Path) -> None:
    payload = {
        "metadata": meta,
        "results": [row_to_flat(r, alpha) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_txt(rows: list[dict], alpha: float, meta: dict, path: Path) -> None:
    lines = []
    lines.append("Paired Model Comparison")
    lines.append("Exact McNemar test + Holm-Bonferroni correction + document-level paired bootstrap")
    lines.append("=" * 88)
    lines.append(f"seed = {meta['seed']}, n_bootstrap = {meta['n_bootstrap']}, alpha = {alpha}")
    lines.append(f"generated_at (UTC) = {meta['generated_at_utc']}")
    lines.append("Holm-Bonferroni family = the 6 predefined pairs, corrected separately per split.")
    lines.append("")

    for split in meta["splits"]:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        lines.append(f"## Split: {split}")
        lines.append("-" * 88)
        for r in split_rows:
            f = row_to_flat(r, alpha)
            lines.append(
                f"{f['model_a_display']}  vs  {f['model_b_display']}"
            )
            lines.append(
                f"  total_tokens={f['total_tokens']:,}  documents={f['n_documents']:,}"
            )
            lines.append(
                f"  both_correct={f['both_correct']:,}  both_wrong={f['both_wrong']:,}  "
                f"a_only={f['a_only_correct']:,}  b_only={f['b_only_correct']:,}  "
                f"discordant={f['discordant_pairs']:,}"
            )
            lines.append(
                f"  McNemar exact p = {f['mcnemar_exact_p']:.6g}   "
                f"Holm-adjusted p = {f['holm_adjusted_p']:.6g}   "
                f"significant (alpha={alpha}) = {f['significant_after_holm']}"
            )
            lines.append(
                f"  Bootstrap (doc-level, {meta['n_bootstrap']} resamples): "
                f"observed diff (A-B) = {f['observed_acc_diff_a_minus_b']:+.6f}   "
                f"bootstrap mean diff = {f['bootstrap_mean_diff_a_minus_b']:+.6f}   "
                f"95% CI = [{f['ci95_low']:+.6f}, {f['ci95_high']:+.6f}]   "
                f"CI contains zero = {f['ci95_contains_zero']}"
            )
            lines.append("")
        lines.append("")

    # Interpretation, focused on Joint Perceptron vs Panyo-Pipeline Perceptron.
    lines.append("## Interpretation: Joint Perceptron vs Panyo-Pipeline Perceptron")
    lines.append("-" * 88)
    for split in meta["splits"]:
        r = next(
            (r for r in rows if r["split"] == split
             and r["model_a"] == "perceptron_joint"
             and r["model_b"] == "perceptron_pipeline_panyo"),
            None,
        )
        if r is None:
            continue
        f = row_to_flat(r, alpha)
        lines.append(
            f"[{split}] discordant={f['discordant_pairs']}, exact p={f['mcnemar_exact_p']:.6g}, "
            f"Holm-adjusted p={f['holm_adjusted_p']:.6g}, "
            f"significant={f['significant_after_holm']}; "
            f"bootstrap 95% CI for acc diff = "
            f"[{f['ci95_low']:+.6f}, {f['ci95_high']:+.6f}] "
            f"(contains zero: {f['ci95_contains_zero']})."
        )
    lines.append("")
    lines.append(
        "The paper's 'no significant paired difference' claim for Joint vs "
        "Panyo-Pipeline (previously based on an approximate, continuity-corrected "
        "chi2 = 1.33 with no exact p-value) is evaluated above using an exact "
        "binomial McNemar p-value, Holm-Bonferroni correction across all 6 "
        "predefined comparisons, and a document-level bootstrap CI. See both "
        "checks above for the current, statistically rigorous verdict."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bootstrap_distributions(rows: list[dict], boot_dir: Path) -> None:
    boot_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        fname = f"{r['model_a']}__vs__{r['model_b']}__{r['split']}.csv"
        path = boot_dir / fname
        diffs = r["bootstrap"]["distribution"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["replicate", "acc_diff_a_minus_b"])
            for i, d in enumerate(diffs):
                w.writerow([i, f"{d:.10f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    splits: list[str] = SPLITS,
    seed: int = SEED,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    out_dir: Path = OUT_DIR,
) -> list[dict]:
    boot_dir = out_dir / "bootstrap_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for split in splits:
        print(f"=== split: {split} ===")
        split_results = []
        for model_a, model_b in PREDEFINED_PAIRS:
            print(f"  comparing {model_a} vs {model_b} ...")
            r = compare_pair(model_a, model_b, split, n_boot=n_boot, seed=seed)
            split_results.append(r)
            print(
                f"    discordant={r['discordant_pairs']}  "
                f"exact_p={r['mcnemar_exact_p']:.6g}  "
                f"boot_diff={r['bootstrap']['observed_diff']:+.6f}  "
                f"95%CI=[{r['bootstrap']['ci_low']:+.6f}, {r['bootstrap']['ci_high']:+.6f}]"
            )

        pvals = [r["mcnemar_exact_p"] for r in split_results]
        adjusted, reject = holm_bonferroni(pvals, alpha)
        for r, adj_p, sig in zip(split_results, adjusted, reject):
            r["holm_adjusted_p"] = adj_p
            r["significant_after_holm"] = sig
            all_rows.append(r)

    meta = {
        "seed": seed,
        "n_bootstrap": n_boot,
        "alpha": alpha,
        "splits": splits,
        "predefined_pairs": PREDEFINED_PAIRS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    write_csv(all_rows, alpha, out_dir / "paired_comparisons.csv")
    write_json(all_rows, alpha, meta, out_dir / "paired_comparisons.json")
    write_txt(all_rows, alpha, meta, out_dir / "paired_comparisons.txt")
    write_bootstrap_distributions(all_rows, boot_dir)

    print(f"\nWrote {out_dir / 'paired_comparisons.csv'}")
    print(f"Wrote {out_dir / 'paired_comparisons.json'}")
    print(f"Wrote {out_dir / 'paired_comparisons.txt'}")
    print(f"Wrote {len(all_rows)} bootstrap distribution files under {boot_dir}")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    run(
        splits=args.splits,
        seed=args.seed,
        n_boot=args.n_boot,
        alpha=args.alpha,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
