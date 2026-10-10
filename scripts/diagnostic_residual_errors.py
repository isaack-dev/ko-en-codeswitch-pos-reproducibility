"""
diagnostic_residual_errors.py - Full residual error catalog for Panyo-trained models.

Models: perceptron_joint, perceptron_pipeline_panyo

All errors are preserved (not sampled). Adds boundary and seen/unseen annotations.

Run from repo root:
  python scripts/diagnostic_residual_errors.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_UPOS, F_MISC, get_misc_lang, iter_sentences
from diagnostic_utils import (
    DIAG, TRAIN_FILE, PANYO_MODELS, acc, pred_path,
    get_pred_upos, get_sent_id, get_doc_id,
)

SPLITS = ["dev", "test"]


def _is_ko_en_boundary(a: str, b: str) -> bool:
    return (a == "KO" and b == "EN") or (a == "EN" and b == "KO")


def build_seen_lower(train_path: Path) -> set:
    from read_conllu import iter_sentences as _iter
    seen = set()
    for sent in _iter(train_path):
        for tok in sent["tokens"]:
            if tok["is_real"]:
                seen.add(tok["fields"][F_FORM].lower())
    return seen


def region_for_token(i: int, langs: list[str]) -> tuple[bool, bool, bool]:
    """Returns (is_boundary, is_near1, is_near2)."""
    n = len(langs)
    boundary_set: set[int] = set()
    for j in range(n - 1):
        if _is_ko_en_boundary(langs[j], langs[j+1]):
            boundary_set.add(j)
            boundary_set.add(j+1)
    if i in boundary_set:
        return True, False, False
    d = min((abs(i - b) for b in boundary_set), default=999)
    return False, (d == 1), (d == 2)


FULL_COLS = [
    "model","split","sent_id","doc_id","token_index","form","lemma","lang",
    "gold_upos","pred_upos","prev_form","prev_lang","prev_gold",
    "next_form","next_lang","next_gold",
    "is_boundary_token","is_near_boundary_1","is_near_boundary_2",
    "seen_form","seen_lower","sentence_context",
]


def run():
    DIAG.mkdir(parents=True, exist_ok=True)
    print("Building training vocabulary for seen/unseen...")
    seen_lower = build_seen_lower(TRAIN_FILE)
    print(f"  {len(seen_lower):,} unique lowercase forms in train")

    # Per-model error list (all errors, both splits)
    model_errors: dict[str, list[dict]] = {m: [] for m in PANYO_MODELS}
    # Summary stats: {(model, split, category, value): count}
    summary_counts: dict[tuple, int] = defaultdict(int)

    for model in PANYO_MODELS:
        for split in SPLITS:
            p = pred_path(model, split)
            if not p.exists():
                continue
            print(f"  Processing {model} {split}...")

            for sent in iter_sentences(p):
                sid  = get_sent_id(sent["comments"])
                did  = get_doc_id(sent["comments"])
                real = [t for t in sent["tokens"] if t["is_real"]]
                n    = len(real)
                if not n:
                    continue

                langs = [get_misc_lang(t["fields"][F_MISC]) for t in real]
                golds = [t["fields"][F_UPOS] for t in real]
                preds = [get_pred_upos(t["fields"][F_MISC]) for t in real]
                forms = [t["fields"][F_FORM] for t in real]
                text  = " ".join(forms)[:120]

                for i in range(n):
                    if golds[i] == preds[i]:
                        continue
                    lo = forms[i].lower()
                    is_b, is_n1, is_n2 = region_for_token(i, langs)
                    seen = lo in seen_lower

                    p_f = real[i-1]["fields"] if i > 0 else None
                    n_f = real[i+1]["fields"] if i < n-1 else None

                    row = {
                        "model": model, "split": split,
                        "sent_id": sid, "doc_id": did,
                        "token_index": i,
                        "form": forms[i], "lemma": real[i]["fields"][2],
                        "lang": langs[i],
                        "gold_upos": golds[i], "pred_upos": preds[i],
                        "prev_form": p_f[F_FORM] if p_f else "",
                        "prev_lang": get_misc_lang(p_f[F_MISC]) if p_f else "",
                        "prev_gold": p_f[F_UPOS] if p_f else "",
                        "next_form": n_f[F_FORM] if n_f else "",
                        "next_lang": get_misc_lang(n_f[F_MISC]) if n_f else "",
                        "next_gold": n_f[F_UPOS] if n_f else "",
                        "is_boundary_token": is_b,
                        "is_near_boundary_1": is_n1,
                        "is_near_boundary_2": is_n2,
                        "seen_form": seen,
                        "seen_lower": seen,
                        "sentence_context": text,
                    }
                    model_errors[model].append(row)

                    # summary counts
                    summary_counts[(model, split, "total", "")] += 1
                    summary_counts[(model, split, "by_gold", golds[i])] += 1
                    summary_counts[(model, split, "by_pred", preds[i])] += 1
                    summary_counts[(model, split, "confusion", f"{golds[i]}->{preds[i]}")] += 1
                    summary_counts[(model, split, "by_lang", langs[i])] += 1
                    summary_counts[(model, split, "by_seen", "seen" if seen else "unseen")] += 1
                    region = "boundary" if is_b else ("near1" if is_n1 else ("near2" if is_n2 else "non"))
                    summary_counts[(model, split, "by_region", region)] += 1
                    summary_counts[(model, split, "by_form", forms[i])] += 1

    # -----------------------------------------------------------------------
    # Write per-model error CSVs
    # -----------------------------------------------------------------------
    for model in PANYO_MODELS:
        fname = f"residual_errors_{model.replace('perceptron_','')}.csv"
        out = DIAG / fname
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FULL_COLS)
            w.writeheader()
            w.writerows(model_errors[model])
        print(f"  -> {out.name} ({len(model_errors[model])} errors)")

    # -----------------------------------------------------------------------
    # Write residual_error_summary.csv
    # -----------------------------------------------------------------------
    out_s = DIAG / "residual_error_summary.csv"
    with out_s.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model","split","category","value","count"])
        for (model, split, cat, val), cnt in sorted(
            summary_counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], -x[1])
        ):
            w.writerow([model, split, cat, val, cnt])
    print(f"  -> {out_s.name}")

    # -----------------------------------------------------------------------
    # Overlap analysis: errors made by joint only / pipeline only / both
    # -----------------------------------------------------------------------
    joint_errs_set:    dict[str, set] = {}
    pipeline_errs_set: dict[str, set] = {}
    for split in SPLITS:
        def err_key(row: dict) -> tuple:
            return (row["sent_id"], row["token_index"], row["gold_upos"], row["pred_upos"])

        joint_errs_set[split]    = {err_key(r) for r in model_errors["perceptron_joint"]
                                     if r["split"] == split}
        pipeline_errs_set[split] = {err_key(r) for r in model_errors["perceptron_pipeline_panyo"]
                                     if r["split"] == split}

    # -----------------------------------------------------------------------
    # Write residual_error_summary.md
    # -----------------------------------------------------------------------
    lines = ["# Residual Error Analysis: Panyo-Trained Models\n"]

    for split in SPLITS:
        lines.append(f"## Split: {split}\n")
        for model in PANYO_MODELS:
            total = summary_counts.get((model, split, "total", ""), 0)
            lines.append(f"### {model}: {total} errors\n")

            # By gold UPOS
            lines.append("**By gold UPOS:**\n")
            by_gold = [(v, cnt) for (m, s, cat, v), cnt in summary_counts.items()
                       if m == model and s == split and cat == "by_gold"]
            by_gold.sort(key=lambda x: -x[1])
            lines.append("| Gold UPOS | Errors |")
            lines.append("|---|---|")
            for tag, cnt in by_gold:
                lines.append(f"| {tag} | {cnt} |")
            lines.append("")

            # Top confusions
            lines.append("**Top confusion pairs:**\n")
            by_conf = [(v, cnt) for (m, s, cat, v), cnt in summary_counts.items()
                       if m == model and s == split and cat == "confusion"]
            by_conf.sort(key=lambda x: -x[1])
            lines.append("| Gold->Pred | Errors |")
            lines.append("|---|---|")
            for pair, cnt in by_conf[:10]:
                lines.append(f"| {pair} | {cnt} |")
            lines.append("")

            # By lang
            lines.append("**By Lang:**\n")
            by_lang = [(v, cnt) for (m, s, cat, v), cnt in summary_counts.items()
                       if m == model and s == split and cat == "by_lang"]
            by_lang.sort(key=lambda x: -x[1])
            for lg, cnt in by_lang:
                lines.append(f"- {lg}: {cnt}")
            lines.append("")

            # By seen/unseen
            lines.append("**By seen/unseen:**\n")
            by_seen = [(v, cnt) for (m, s, cat, v), cnt in summary_counts.items()
                       if m == model and s == split and cat == "by_seen"]
            for sv, cnt in sorted(by_seen):
                lines.append(f"- {sv}: {cnt}")
            lines.append("")

        # Overlap
        j_set = joint_errs_set[split]
        p_set = pipeline_errs_set[split]
        j_only = len(j_set - p_set)
        p_only = len(p_set - j_set)
        both   = len(j_set & p_set)
        lines.append(f"### Joint vs Pipeline Panyo error overlap ({split})\n")
        lines.append(f"- Errors in joint only: {j_only}")
        lines.append(f"- Errors in pipeline only: {p_only}")
        lines.append(f"- Errors in both: {both}")
        lines.append("")

    lines.append("## Interpretation\n")
    j_test = summary_counts.get(("perceptron_joint", "test", "total", ""), 0)
    p_test = summary_counts.get(("perceptron_pipeline_panyo", "test", "total", ""), 0)
    lines.append(f"- Total residual errors on test: joint={j_test}, pipeline_panyo={p_test}.")
    lines.append(
        "- With ~20-30 errors total, all errors are preserved for manual inspection."
    )
    lines.append(
        "- PROPN->NOUN errors reflect genuine PROPN/NOUN ambiguity for proper nouns "
        "that also appear as common nouns."
    )
    lines.append(
        "- AUX errors may reflect a specific auxiliary form that appears rarely in training "
        "or whose features conflict with common NOUN features."
    )

    out_md = DIAG / "residual_error_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {out_md.name}")
    print("diagnostic_residual_errors.py: done.")


if __name__ == "__main__":
    run()
