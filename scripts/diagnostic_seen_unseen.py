"""
diagnostic_seen_unseen.py - Seen vs unseen token accuracy analysis (Step 4).

Definitions:
  seen_form      : lowercase(FORM) appears in training set
  unseen_form    : lowercase(FORM) does NOT appear in training set
  seen_form_lang : (lowercase(FORM), Lang) pair appears in training set
  seen_form_lang_gold: (lowercase(FORM), Lang, gold_UPOS) triple appears in training set

Run from repo root:
  python scripts/diagnostic_seen_unseen.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_UPOS, F_MISC, get_misc_lang, iter_sentences
from diagnostic_utils import (
    DIAG, TRAIN_FILE, ALL_MODELS, acc,
    iter_token_records, pred_path, get_pred_upos,
)

SPLITS  = ["dev", "test"]
MODELS  = ALL_MODELS


# ---------------------------------------------------------------------------
# Build training vocabulary sets
# ---------------------------------------------------------------------------

def build_train_vocab(train_path: Path) -> tuple[set, set, set]:
    """
    Returns:
      seen_lower      : set of lowercase forms in train
      seen_form_lang  : set of (lower_form, lang) pairs in train
      seen_flt        : set of (lower_form, lang, gold_upos) triples in train
    """
    seen_lower:     set[str]   = set()
    seen_form_lang: set[tuple] = set()
    seen_flt:       set[tuple] = set()

    for sent in iter_sentences(train_path):
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            f     = tok["fields"]
            form  = f[F_FORM].lower()
            lang  = get_misc_lang(f[F_MISC])
            upos  = f[F_UPOS]
            seen_lower.add(form)
            seen_form_lang.add((form, lang))
            seen_flt.add((form, lang, upos))

    return seen_lower, seen_form_lang, seen_flt


# ---------------------------------------------------------------------------
# Accumulate stats
# ---------------------------------------------------------------------------

def run():
    DIAG.mkdir(parents=True, exist_ok=True)
    print("Building training vocabulary...")
    seen_lower, seen_form_lang, seen_flt = build_train_vocab(TRAIN_FILE)
    print(f"  Unique lowercase forms in train: {len(seen_lower):,}")
    print(f"  Unique (form, lang) pairs in train: {len(seen_form_lang):,}")

    # -----------------------------------------------------------------------
    # Row accumulators
    # seen_unseen_accuracy.csv     : model x split x condition
    # seen_unseen_by_lang.csv      : model x split x condition x lang
    # seen_unseen_by_pos.csv       : model x split x condition x gold_upos
    # -----------------------------------------------------------------------
    # {(model, split, cond): [total, correct]}
    acc_rows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    lang_rows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    pos_rows:  dict[tuple, list[int]] = defaultdict(lambda: [0, 0])

    # errors on unseen tokens (up to 500 per model+split)
    unseen_errors: list[dict] = []

    for model in MODELS:
        for split in SPLITS:
            p = pred_path(model, split)
            if not p.exists():
                print(f"  SKIP {p.name}")
                continue
            print(f"  Processing {model} {split}...")

            for tr in iter_token_records(p):
                lo   = tr.form.lower()
                cond = "seen" if lo in seen_lower else "unseen"
                ok   = 1 if tr.gold_upos == tr.pred_upos else 0

                acc_rows[(model, split, cond)][0] += 1
                acc_rows[(model, split, cond)][1] += ok

                lang_rows[(model, split, cond, tr.lang)][0] += 1
                lang_rows[(model, split, cond, tr.lang)][1] += ok

                pos_rows[(model, split, cond, tr.gold_upos)][0] += 1
                pos_rows[(model, split, cond, tr.gold_upos)][1] += ok

                if cond == "unseen" and ok == 0 and len(unseen_errors) < 500:
                    unseen_errors.append({
                        "model": model, "split": split,
                        "sent_id": tr.sent_id, "doc_id": tr.doc_id,
                        "token_index": tr.tok_idx,
                        "form": tr.form, "lemma": tr.lemma, "lang": tr.lang,
                        "gold_upos": tr.gold_upos, "pred_upos": tr.pred_upos,
                        "prev_form": tr.prev_form, "next_form": tr.next_form,
                        "sentence_context": tr.sentence_context[:120],
                    })

    # -----------------------------------------------------------------------
    # Write seen_unseen_accuracy.csv
    # -----------------------------------------------------------------------
    out1 = DIAG / "seen_unseen_accuracy.csv"
    with out1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "total", "correct", "accuracy", "error_count"])
        for (model, split, cond), (tot, cor) in sorted(acc_rows.items()):
            w.writerow([model, split, cond, tot, cor, acc(cor, tot), tot - cor])
    print(f"  -> {out1.name}")

    # -----------------------------------------------------------------------
    # Write seen_unseen_by_lang.csv
    # -----------------------------------------------------------------------
    out2 = DIAG / "seen_unseen_by_lang.csv"
    with out2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "lang", "total", "correct", "accuracy", "error_count"])
        for (model, split, cond, lang), (tot, cor) in sorted(lang_rows.items()):
            w.writerow([model, split, cond, lang, tot, cor, acc(cor, tot), tot - cor])
    print(f"  -> {out2.name}")

    # -----------------------------------------------------------------------
    # Write seen_unseen_by_pos.csv
    # -----------------------------------------------------------------------
    out3 = DIAG / "seen_unseen_by_pos.csv"
    with out3.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "gold_upos", "total", "correct", "accuracy", "error_count"])
        for (model, split, cond, upos), (tot, cor) in sorted(pos_rows.items()):
            w.writerow([model, split, cond, upos, tot, cor, acc(cor, tot), tot - cor])
    print(f"  -> {out3.name}")

    # -----------------------------------------------------------------------
    # Write unseen_token_errors.csv
    # -----------------------------------------------------------------------
    out4 = DIAG / "unseen_token_errors.csv"
    cols4 = ["model","split","sent_id","doc_id","token_index","form","lemma","lang",
             "gold_upos","pred_upos","prev_form","next_form","sentence_context"]
    with out4.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols4)
        w.writeheader()
        w.writerows(unseen_errors)
    print(f"  -> {out4.name} ({len(unseen_errors)} rows)")

    # -----------------------------------------------------------------------
    # Write unseen_token_summary.md
    # -----------------------------------------------------------------------
    # Gather key stats for the summary
    summary_lines = ["# Seen vs Unseen Token Analysis\n"]

    for split in SPLITS:
        summary_lines.append(f"## Split: {split}\n")
        summary_lines.append("| Model | Seen total | Seen acc | Unseen total | Unseen acc |")
        summary_lines.append("|---|---|---|---|---|")
        for model in MODELS:
            s_tot, s_cor = acc_rows.get((model, split, "seen"), [0, 0])
            u_tot, u_cor = acc_rows.get((model, split, "unseen"), [0, 0])
            pct_unseen = 100.0 * u_tot / (s_tot + u_tot) if (s_tot + u_tot) else 0.0
            summary_lines.append(
                f"| {model} | {s_tot:,} ({100-pct_unseen:.1f}%) | "
                f"{acc(s_cor,s_tot):.4f} | {u_tot:,} ({pct_unseen:.1f}%) | "
                f"{acc(u_cor,u_tot):.4f} |"
            )
        summary_lines.append("")

    summary_lines.append("## Interpretation\n")
    # Check joint test seen vs unseen
    s_tot, s_cor = acc_rows.get(("perceptron_joint", "test", "seen"), [0, 0])
    u_tot, u_cor = acc_rows.get(("perceptron_joint", "test", "unseen"), [0, 0])
    pct_u = 100.0 * u_tot / (s_tot + u_tot) if (s_tot + u_tot) else 0.0
    summary_lines.append(
        f"- On test, {pct_u:.1f}% of tokens ({u_tot:,}) are unseen by lowercase form."
    )
    if u_tot > 0:
        summary_lines.append(
            f"- perceptron_joint accuracy on unseen tokens (test): {acc(u_cor, u_tot):.4f}"
            f" ({u_cor:,}/{u_tot:,} correct)."
        )
    summary_lines.append(
        "- Near-perfect accuracy on seen tokens is consistent with lexical memorization."
    )
    if u_tot < 200:
        summary_lines.append(
            "- Unseen token count is very small; unseen-accuracy estimates have wide uncertainty."
        )
    summary_lines.append(
        "- These results do not by themselves prove or disprove generalization; "
        "the small unseen-token count limits conclusions."
    )

    out5 = DIAG / "unseen_token_summary.md"
    out5.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"  -> {out5.name}")
    print("diagnostic_seen_unseen.py: done.")


if __name__ == "__main__":
    run()
