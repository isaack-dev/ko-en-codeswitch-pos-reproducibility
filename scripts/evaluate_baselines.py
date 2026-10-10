"""
evaluate_baselines.py - Evaluate Step 3 baseline prediction files.

Reads gold UPOS from column 4 and predicted UPOS from PredUPOS=... in MISC.

Metrics computed:
  - Overall token-level accuracy
  - Non-punctuation accuracy (excluding gold PUNCT)
  - Per-class precision / recall / F1 / support
  - Macro F1 (unweighted mean over classes with support > 0)
  - Weighted F1 (support-weighted mean)
  - Language-wise accuracy (KO / EN / OTHER / MIXED)
  - Confusion matrix

Output files (results/baselines/):
  overall_results.csv
  pos_wise_f1.csv
  language_wise_accuracy.csv
  confusion_matrix_<model>.csv

Run from repo root:
  python scripts/evaluate_baselines.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, get_misc_lang, iter_sentences

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "baselines"

MODELS = [
    "majority_global",
    "majority_lang",
    "perceptron_joint",
    "perceptron_pipeline_panyo",
    "perceptron_pipeline_ud_transfer",
]
SPLITS = ["dev", "test"]


# ---------------------------------------------------------------------------
# Read predictions from a file
# ---------------------------------------------------------------------------

def get_pred_upos(misc: str) -> str | None:
    """Extract PredUPOS=<tag> from MISC; return None if absent."""
    for part in misc.split("|"):
        if part.startswith("PredUPOS="):
            return part[9:]
    return None


def read_predictions(path: Path) -> list[dict]:
    """
    Parse a prediction file into token records.
    Each record: {gold, pred, lang, sent_id}
    Only real tokens with both gold UPOS and PredUPOS are included.
    """
    records: list[dict] = []
    for sent in iter_sentences(path):
        sid = ""
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[12:]
                break
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            fields = tok["fields"]
            gold   = fields[F_UPOS]
            misc   = fields[F_MISC]
            pred   = get_pred_upos(misc)
            lang   = get_misc_lang(misc)
            if gold and pred:
                records.append({"gold": gold, "pred": pred, "lang": lang, "sent_id": sid})
    return records


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def compute_accuracy(records: list[dict], exclude_gold: str | None = None) -> tuple[int, int]:
    """Returns (correct, total) optionally excluding a gold label."""
    corr = tot = 0
    for r in records:
        if exclude_gold and r["gold"] == exclude_gold:
            continue
        tot += 1
        if r["gold"] == r["pred"]:
            corr += 1
    return corr, tot


def compute_per_class(records: list[dict]) -> dict[str, dict]:
    """
    Return per-class TP, FP, FN, support (= gold count).
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    sup: Counter = Counter()

    for r in records:
        g, p = r["gold"], r["pred"]
        sup[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1

    all_tags = sorted(set(list(sup.keys()) + list(fp.keys())))
    result: dict[str, dict] = {}
    for tag in all_tags:
        prec = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0.0
        rec  = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec)  if (prec + rec) > 0        else 0.0
        result[tag] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   sup[tag],
            "tp":        tp[tag],
            "fp":        fp[tag],
            "fn":        fn[tag],
        }
    return result


def compute_macro_f1(per_class: dict[str, dict]) -> float:
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def compute_weighted_f1(per_class: dict[str, dict]) -> float:
    total = sum(v["support"] for v in per_class.values())
    if total == 0:
        return 0.0
    return round(
        sum(v["f1"] * v["support"] for v in per_class.values()) / total, 4
    )


def compute_lang_accuracy(records: list[dict]) -> dict[str, tuple[int, int]]:
    """Returns {lang: (correct, total)}."""
    lang_corr: Counter = Counter()
    lang_tot:  Counter = Counter()
    for r in records:
        lang = r["lang"] or "UNKNOWN"
        lang_tot[lang] += 1
        if r["gold"] == r["pred"]:
            lang_corr[lang] += 1
    all_langs = sorted(lang_tot.keys())
    return {l: (lang_corr[l], lang_tot[l]) for l in all_langs}


def compute_confusion(records: list[dict]) -> tuple[list[str], dict]:
    """
    Returns (sorted_tags, confusion_dict).
    confusion_dict[(gold, pred)] = count
    """
    gold_tags = sorted({r["gold"] for r in records})
    pred_tags = sorted({r["pred"] for r in records})
    all_tags  = sorted(set(gold_tags) | set(pred_tags))
    conf: dict = defaultdict(int)
    for r in records:
        conf[(r["gold"], r["pred"])] += 1
    return all_tags, conf


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    overall_rows:  list[dict] = []
    pos_rows:      list[dict] = []
    lang_rows:     list[dict] = []

    for model in MODELS:
        for split in SPLITS:
            pred_path = RESULTS / f"{model}_{split}_predictions.conllu"
            if not pred_path.exists():
                print(f"  SKIP: {pred_path.name} not found")
                continue

            print(f"\nEvaluating {model} / {split} ...")
            records = read_predictions(pred_path)
            if not records:
                print(f"  WARNING: no records found in {pred_path.name}")
                continue

            # --- overall accuracy ---
            corr, tot       = compute_accuracy(records)
            corr_np, tot_np = compute_accuracy(records, exclude_gold="PUNCT")
            acc    = round(corr / tot,       4) if tot    else 0.0
            acc_np = round(corr_np / tot_np, 4) if tot_np else 0.0

            # --- per-class metrics ---
            pc = compute_per_class(records)
            mf1 = compute_macro_f1(pc)
            wf1 = compute_weighted_f1(pc)

            overall_rows.append({
                "model":                model,
                "split":               split,
                "tokens":              tot,
                "correct":             corr,
                "accuracy":            acc,
                "non_punct_tokens":    tot_np,
                "non_punct_correct":   corr_np,
                "non_punct_accuracy":  acc_np,
                "macro_f1":            mf1,
                "weighted_f1":         wf1,
            })
            print(f"  acc={acc:.4f}  acc_np={acc_np:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}")

            # --- per-class rows ---
            for tag, m in pc.items():
                pos_rows.append({
                    "model":     model,
                    "split":     split,
                    "tag":       tag,
                    "precision": m["precision"],
                    "recall":    m["recall"],
                    "f1":        m["f1"],
                    "support":   m["support"],
                    "tp":        m["tp"],
                    "fp":        m["fp"],
                    "fn":        m["fn"],
                })

            # --- language-wise accuracy ---
            for lang, (lc, lt) in compute_lang_accuracy(records).items():
                lang_rows.append({
                    "model":    model,
                    "split":    split,
                    "lang":     lang,
                    "correct":  lc,
                    "total":    lt,
                    "accuracy": round(lc / lt, 4) if lt else 0.0,
                })
                print(f"  {lang}: {lc}/{lt} = {lc/lt:.4f}" if lt else f"  {lang}: 0/0")

            # --- confusion matrix ---
            all_tags, conf = compute_confusion(records)
            conf_path = RESULTS / f"confusion_matrix_{model}_{split}.csv"
            with conf_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["gold\\pred"] + all_tags)
                for g in all_tags:
                    w.writerow([g] + [conf.get((g, p), 0) for p in all_tags])
            print(f"  Confusion matrix -> {conf_path.name}")

    # -----------------------------------------------------------------------
    # Write aggregate CSVs
    # -----------------------------------------------------------------------
    _write_csv(
        RESULTS / "overall_results.csv",
        ["model", "split", "tokens", "correct", "accuracy",
         "non_punct_tokens", "non_punct_correct", "non_punct_accuracy",
         "macro_f1", "weighted_f1"],
        overall_rows,
    )
    print(f"\nWrote {RESULTS / 'overall_results.csv'}")

    _write_csv(
        RESULTS / "pos_wise_f1.csv",
        ["model", "split", "tag", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
        pos_rows,
    )
    print(f"Wrote {RESULTS / 'pos_wise_f1.csv'}")

    _write_csv(
        RESULTS / "language_wise_accuracy.csv",
        ["model", "split", "lang", "correct", "total", "accuracy"],
        lang_rows,
    )
    print(f"Wrote {RESULTS / 'language_wise_accuracy.csv'}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n=== Overall Results Summary ===")
    print(f"{'Model':<40} {'Split':<5} {'Acc':>7} {'NP-Acc':>7} {'MacroF1':>8} {'WtF1':>7}")
    print("-" * 80)
    for r in overall_rows:
        print(f"{r['model']:<40} {r['split']:<5} "
              f"{r['accuracy']:>7.4f} {r['non_punct_accuracy']:>7.4f} "
              f"{r['macro_f1']:>8.4f} {r['weighted_f1']:>7.4f}")


if __name__ == "__main__":
    main()
