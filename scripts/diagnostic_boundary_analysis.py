"""
diagnostic_boundary_analysis.py - Code-switching boundary error analysis (Step 4).

A direct KO<->EN boundary is between adjacent real tokens where Lang switches
between KO and EN (ignoring direction).

Token region labels:
  boundary_token  : participates directly in a KO<->EN adjacent pair
  near_boundary_1 : within distance 1 of any boundary token (but not a boundary token)
  near_boundary_2 : within distance 2 but not 1 or 0
  non_boundary    : more than 2 tokens from any boundary

Run from repo root:
  python scripts/diagnostic_boundary_analysis.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_UPOS, F_MISC, get_misc_lang, iter_sentences
from diagnostic_utils import (
    DIAG, ALL_MODELS, acc, pred_path, get_pred_upos,
    get_sent_id, get_doc_id,
)

SPLITS = ["dev", "test"]
MODELS = ALL_MODELS


def _is_ko_en_boundary(lang_a: str, lang_b: str) -> bool:
    return (lang_a == "KO" and lang_b == "EN") or (lang_a == "EN" and lang_b == "KO")


def boundary_direction(lang_a: str, lang_b: str) -> str:
    if lang_a == "KO" and lang_b == "EN":
        return "KO_to_EN"
    if lang_a == "EN" and lang_b == "KO":
        return "EN_to_KO"
    return ""


def region_labels(langs: list[str]) -> list[str]:
    """
    For a list of Lang values (real tokens only), return region label for each token.
    """
    n = len(langs)
    boundary_set: set[int] = set()
    for i in range(n - 1):
        if _is_ko_en_boundary(langs[i], langs[i+1]):
            boundary_set.add(i)
            boundary_set.add(i + 1)

    labels = []
    for i in range(n):
        if i in boundary_set:
            labels.append("boundary_token")
        elif any(abs(i - b) == 1 for b in boundary_set):
            labels.append("near_boundary_1")
        elif any(abs(i - b) == 2 for b in boundary_set):
            labels.append("near_boundary_2")
        else:
            labels.append("non_boundary")
    return labels


def run():
    DIAG.mkdir(parents=True, exist_ok=True)

    # region accuracy: {(model, split, region): [total, correct]}
    region_acc:  dict[tuple, list] = defaultdict(lambda: [0, 0])
    # direction accuracy: {(model, split, direction): [n_pairs, both_correct, left_ok, right_ok]}
    dir_acc:     dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0])
    # pattern: {(model, split, prev_lang, curr_lang, prev_upos, curr_upos): [total, both_correct]}
    pat_acc:     dict[tuple, list] = defaultdict(lambda: [0, 0])

    boundary_errors: list[dict] = []

    for model in MODELS:
        for split in SPLITS:
            p = pred_path(model, split)
            if not p.exists():
                continue
            print(f"  Processing {model} {split}...")

            for sent in iter_sentences(p):
                sid = get_sent_id(sent["comments"])
                did = get_doc_id(sent["comments"])
                real = [t for t in sent["tokens"] if t["is_real"]]
                if not real:
                    continue

                n = len(real)
                langs     = [get_misc_lang(t["fields"][F_MISC]) for t in real]
                golds     = [t["fields"][F_UPOS] for t in real]
                preds     = [get_pred_upos(t["fields"][F_MISC]) for t in real]
                forms     = [t["fields"][F_FORM] for t in real]
                regions   = region_labels(langs)
                text_ctx  = " ".join(forms)[:120]

                # region accuracy per token
                for i in range(n):
                    ok = 1 if golds[i] == preds[i] else 0
                    region_acc[(model, split, regions[i])][0] += 1
                    region_acc[(model, split, regions[i])][1] += ok

                # boundary pairs
                for i in range(n - 1):
                    if not _is_ko_en_boundary(langs[i], langs[i+1]):
                        continue
                    direction = boundary_direction(langs[i], langs[i+1])
                    l_ok = 1 if golds[i]   == preds[i]   else 0
                    r_ok = 1 if golds[i+1] == preds[i+1] else 0
                    both = 1 if l_ok and r_ok else 0

                    dir_acc[(model, split, direction)][0] += 1
                    dir_acc[(model, split, direction)][1] += both
                    dir_acc[(model, split, direction)][2] += l_ok
                    dir_acc[(model, split, direction)][3] += r_ok

                    pat_key = (model, split, langs[i], langs[i+1], golds[i], golds[i+1])
                    pat_acc[pat_key][0] += 1
                    pat_acc[pat_key][1] += both

                    # Collect error examples
                    if (l_ok == 0 or r_ok == 0) and len(boundary_errors) < 500:
                        boundary_errors.append({
                            "model": model, "split": split,
                            "sent_id": sid, "doc_id": did,
                            "boundary_direction": direction,
                            "prev_form": forms[i],   "prev_lang": langs[i],
                            "prev_gold": golds[i],   "prev_pred": preds[i],
                            "curr_form": forms[i+1], "curr_lang": langs[i+1],
                            "curr_gold": golds[i+1], "curr_pred": preds[i+1],
                            "sentence_context": text_ctx,
                        })

    # -----------------------------------------------------------------------
    # Write boundary_region_accuracy.csv
    # -----------------------------------------------------------------------
    out1 = DIAG / "boundary_region_accuracy.csv"
    with out1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model","split","region","total","correct","accuracy","error_count"])
        for (model, split, region), (tot, cor) in sorted(region_acc.items()):
            w.writerow([model, split, region, tot, cor, acc(cor, tot), tot-cor])
    print(f"  -> {out1.name}")

    # -----------------------------------------------------------------------
    # Write boundary_direction_accuracy.csv
    # -----------------------------------------------------------------------
    out2 = DIAG / "boundary_direction_accuracy.csv"
    with out2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model","split","direction","total_boundary_pairs",
                    "pair_both_correct","pair_accuracy","left_token_acc","right_token_acc"])
        for (model, split, direction), (n_pairs, both, l_sum, r_sum) in sorted(dir_acc.items()):
            w.writerow([
                model, split, direction, n_pairs, both,
                acc(both, n_pairs), acc(l_sum, n_pairs), acc(r_sum, n_pairs),
            ])
    print(f"  -> {out2.name}")

    # -----------------------------------------------------------------------
    # Write boundary_pattern_accuracy.csv (top patterns by total count)
    # -----------------------------------------------------------------------
    out3 = DIAG / "boundary_pattern_accuracy.csv"
    with out3.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model","split","prev_lang","curr_lang","prev_upos","curr_upos",
                    "total_pairs","both_correct","pair_accuracy"])
        for (model, split, pl, cl, pu, cu), (tot, both) in sorted(
            pat_acc.items(), key=lambda x: -x[1][0]
        ):
            w.writerow([model, split, pl, cl, pu, cu, tot, both, acc(both, tot)])
    print(f"  -> {out3.name}")

    # -----------------------------------------------------------------------
    # Write boundary_error_examples.csv
    # -----------------------------------------------------------------------
    out4 = DIAG / "boundary_error_examples.csv"
    cols4 = ["model","split","sent_id","doc_id","boundary_direction",
             "prev_form","prev_lang","prev_gold","prev_pred",
             "curr_form","curr_lang","curr_gold","curr_pred","sentence_context"]
    with out4.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols4)
        w.writeheader()
        w.writerows(boundary_errors)
    print(f"  -> {out4.name} ({len(boundary_errors)} rows)")

    # -----------------------------------------------------------------------
    # Write boundary_summary.md
    # -----------------------------------------------------------------------
    lines = ["# Code-Switching Boundary Analysis\n"]
    for split in SPLITS:
        lines.append(f"## Split: {split}\n")
        lines.append("### Region Accuracy\n")
        lines.append("| Model | boundary_token | near_boundary_1 | near_boundary_2 | non_boundary |")
        lines.append("|---|---|---|---|---|")
        regions = ["boundary_token", "near_boundary_1", "near_boundary_2", "non_boundary"]
        for model in MODELS:
            vals = []
            for r in regions:
                tot, cor = region_acc.get((model, split, r), [0, 0])
                vals.append(f"{acc(cor,tot):.4f} ({tot:,})")
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        lines.append("### Boundary Direction Accuracy (pair-level)\n")
        lines.append("| Model | Direction | Pairs | Both correct | Pair acc | Left acc | Right acc |")
        lines.append("|---|---|---|---|---|---|---|")
        for model in MODELS:
            for direction in ["KO_to_EN", "EN_to_KO"]:
                v = dir_acc.get((model, split, direction), [0, 0, 0, 0])
                n_p, both, ls, rs = v
                lines.append(
                    f"| {model} | {direction} | {n_p:,} | {both:,} | "
                    f"{acc(both,n_p):.4f} | {acc(ls,n_p):.4f} | {acc(rs,n_p):.4f} |"
                )
        lines.append("")

    lines.append("## Interpretation\n")
    # Compute whether boundary tokens are harder for joint on test
    b_tot, b_cor = region_acc.get(("perceptron_joint", "test", "boundary_token"), [0, 0])
    nb_tot, nb_cor = region_acc.get(("perceptron_joint", "test", "non_boundary"), [0, 0])
    lines.append(
        f"- perceptron_joint test: boundary_token acc={acc(b_cor,b_tot):.4f} ({b_tot:,} tokens); "
        f"non_boundary acc={acc(nb_cor,nb_tot):.4f} ({nb_tot:,} tokens)."
    )
    err_count = len(boundary_errors)
    panyo_errs = sum(1 for e in boundary_errors if e["model"] in ("perceptron_joint","perceptron_pipeline_panyo"))
    ud_errs    = sum(1 for e in boundary_errors if e["model"] == "perceptron_pipeline_ud_transfer")
    lines.append(
        f"- Total boundary error examples collected: {err_count} "
        f"(Panyo models: {panyo_errs}, UD-transfer: {ud_errs})."
    )
    if panyo_errs < 20:
        lines.append(
            "- Very few boundary errors for Panyo-trained models; "
            "boundary difficulty cannot be assessed reliably for those models."
        )
    lines.append(
        "- UD-transfer boundary errors are expected to be more numerous due to annotation mismatch."
    )

    out5 = DIAG / "boundary_summary.md"
    out5.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {out5.name}")
    print("diagnostic_boundary_analysis.py: done.")


if __name__ == "__main__":
    run()
