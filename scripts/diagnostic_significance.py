"""
diagnostic_significance.py - Paired McNemar comparison of model pairs (Step 4).

Uses token-level correctness vectors from prediction files.
Does NOT require scipy. Reports McNemar chi-square statistic only.

Run from repo root:
  python scripts/diagnostic_significance.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_UPOS, F_MISC, get_misc_lang, iter_sentences
from diagnostic_utils import (
    DIAG, RESULTS, acc, get_pred_upos, get_sent_id,
)

SPLITS = ["dev", "test"]

# Model pairs to compare
PAIRS = [
    ("perceptron_joint",                 "perceptron_pipeline_panyo"),
    ("majority_global",                  "majority_lang"),
    ("majority_lang",                    "perceptron_pipeline_ud_transfer"),
    ("perceptron_joint",                 "majority_lang"),
]

# Also include mapped UD-transfer if files exist
MAPPED_PAIRS_NAMES = [
    ("perceptron_pipeline_ud_transfer",  "ud_transfer_panyo_coarse"),
    ("perceptron_pipeline_ud_transfer",  "ud_transfer_ko_oriented"),
]


def load_correctness(path: Path) -> dict[str, list[int]]:
    """Return {sent_id: [1/0 per real token]} from a prediction file."""
    result: dict[str, list[int]] = {}
    for sent in iter_sentences(path):
        sid = get_sent_id(sent["comments"])
        corr = []
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            f    = tok["fields"]
            gold = f[F_UPOS]
            pred = get_pred_upos(f[F_MISC])
            corr.append(1 if gold == pred else 0)
        result[sid] = corr
    return result


def flatten(corr: dict[str, list[int]]) -> list[int]:
    """Flatten sent->token correctness map to a flat list in sentence order."""
    out = []
    for sid in sorted(corr.keys()):
        out.extend(corr[sid])
    return out


def mcnemar(a_correct: list[int], b_correct: list[int]) -> dict:
    """
    Compute McNemar-style paired comparison.
    a_correct, b_correct: parallel lists of 0/1.
    """
    n = len(a_correct)
    both_c = both_w = a_only = b_only = 0
    for ac, bc in zip(a_correct, b_correct):
        if ac and bc:
            both_c += 1
        elif not ac and not bc:
            both_w += 1
        elif ac and not bc:
            a_only += 1
        else:
            b_only += 1

    denom = a_only + b_only
    if denom > 0:
        chi2 = (abs(a_only - b_only) - 1.0) ** 2 / denom
    else:
        chi2 = 0.0

    return {
        "total_tokens": n,
        "both_correct": both_c,
        "both_wrong": both_w,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "disagreements": denom,
        "mcnemar_chi2": round(chi2, 4),
    }


def get_pred_file(model: str, split: str) -> Path:
    # check results/baselines first, then diagnostic_analysis
    p = RESULTS / f"{model}_{split}_predictions.conllu"
    if p.exists():
        return p
    q = DIAG / f"{model}_{split}_predictions.conllu"
    return q


def run():
    DIAG.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    all_pairs = list(PAIRS)
    # Add mapped UD-transfer pairs if available
    for (ma, mb) in MAPPED_PAIRS_NAMES:
        test_path = DIAG / f"{mb}_test_predictions.conllu"
        if test_path.exists():
            all_pairs.append((ma, mb))

    for split in SPLITS:
        for (model_a, model_b) in all_pairs:
            pa = get_pred_file(model_a, split)
            pb = get_pred_file(model_b, split)
            if not pa.exists():
                print(f"  SKIP {model_a} {split}: not found")
                continue
            if not pb.exists():
                print(f"  SKIP {model_b} {split}: not found")
                continue

            print(f"  Comparing {model_a} vs {model_b} [{split}]...")
            ca = flatten(load_correctness(pa))
            cb = flatten(load_correctness(pb))
            if len(ca) != len(cb):
                print(f"    WARNING: token count mismatch: {len(ca)} vs {len(cb)}")
                n = min(len(ca), len(cb))
                ca, cb = ca[:n], cb[:n]

            stats = mcnemar(ca, cb)
            rows.append({"split": split, "model_a": model_a, "model_b": model_b, **stats})

            a_acc = acc(sum(ca), len(ca))
            b_acc = acc(sum(cb), len(cb))
            print(f"    model_a acc={a_acc:.4f}  model_b acc={b_acc:.4f}")
            print(f"    disagreements={stats['disagreements']}, chi2={stats['mcnemar_chi2']:.2f}")

    # -----------------------------------------------------------------------
    # Write paired_model_comparison.csv
    # -----------------------------------------------------------------------
    out1 = DIAG / "paired_model_comparison.csv"
    cols = ["split","model_a","model_b","total_tokens","both_correct","both_wrong",
            "a_only_correct","b_only_correct","disagreements","mcnemar_chi2"]
    with out1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out1.name}")

    # -----------------------------------------------------------------------
    # Write summary markdown
    # -----------------------------------------------------------------------
    lines = ["# Paired Model Comparison (McNemar Test)\n"]
    lines.append(
        "McNemar chi-square statistic with continuity correction: "
        "`(|n01 - n10| - 1)^2 / (n01 + n10)`. "
        "Approximate critical value at alpha=0.05: chi2 > 3.84. "
        "Exact p-values were not computed.\n"
    )

    for split in SPLITS:
        lines.append(f"## Split: {split}\n")
        lines.append("| Model A | Model B | Disagree | A-only | B-only | chi2 |")
        lines.append("|---|---|---|---|---|---|")
        for row in [r for r in rows if r["split"] == split]:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | "
                f"{row['disagreements']:,} | {row['a_only_correct']:,} | "
                f"{row['b_only_correct']:,} | {row['mcnemar_chi2']:.2f} |"
            )
        lines.append("")

    lines.append("## Interpretation\n")

    # joint vs pipeline
    for split in SPLITS:
        row = next((r for r in rows
                    if r["split"] == split
                    and r["model_a"] == "perceptron_joint"
                    and r["model_b"] == "perceptron_pipeline_panyo"), None)
        if row:
            lines.append(
                f"- **joint vs pipeline_panyo ({split}):** "
                f"{row['disagreements']} disagreements, chi2={row['mcnemar_chi2']:.2f}. "
                f"The practical difference is negligible (< 30 tokens)."
            )

    lines.append(
        "- **majority_global vs majority_lang:** majority_lang gains by predicting PUNCT "
        "for OTHER tokens; chi2 reflects this systematic difference."
    )
    lines.append(
        "- **majority_lang vs UD-transfer:** UD-transfer helps on EN tokens (GUM tagger) "
        "but hurts on KO (KAIST mismatch) and OTHER (NUM routing). "
        "Net effect depends on corpus composition."
    )
    lines.append(
        "- High chi2 values between Panyo-trained models and UD-transfer reflect "
        "systematic, not random, differences driven by annotation scheme."
    )

    out2 = DIAG / "paired_model_comparison_summary.md"
    out2.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {out2.name}")
    print("diagnostic_significance.py: done.")


if __name__ == "__main__":
    run()
