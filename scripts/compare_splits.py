"""
compare_splits.py - Compare Step 1 sentence-level split vs Step 2 document-level split.

Inputs:
    Step 1 (sentence-level):
        data/processed/panyo_original_adp_train.conllu
        data/processed/panyo_original_adp_dev.conllu
        data/processed/panyo_original_adp_test.conllu
    Step 2 (document-level):
        data/processed/panyo_original_adp_docsplit_train.conllu
        data/processed/panyo_original_adp_docsplit_dev.conllu
        data/processed/panyo_original_adp_docsplit_test.conllu

Output:
    results/docsplit_vs_sentence_split_comparison.csv

Also runs the docsplit dataset statistics and writes:
    results/docsplit_dataset_statistics.csv
    results/docsplit_pos_distribution.csv
    results/docsplit_language_distribution.csv
    results/docsplit_boundary_statistics.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, get_misc_lang, iter_sentences

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
PROC    = DATA / "processed"

STEP1 = {
    "train": PROC / "panyo_original_adp_train.conllu",
    "dev":   PROC / "panyo_original_adp_dev.conllu",
    "test":  PROC / "panyo_original_adp_test.conllu",
}
STEP2 = {
    "train": PROC / "panyo_original_adp_docsplit_train.conllu",
    "dev":   PROC / "panyo_original_adp_docsplit_dev.conllu",
    "test":  PROC / "panyo_original_adp_docsplit_test.conllu",
}
STEP2_FULL = PROC / "panyo_original_adp_docsplit_full.conllu"

LANG_LABELS = ("KO", "EN", "MIXED", "OTHER")


# ---------------------------------------------------------------------------
# Collection helpers (stream-based, one pass)
# ---------------------------------------------------------------------------

def get_comment_value(comments: list[str], key: str) -> str:
    prefix = f"# {key} = "
    for c in comments:
        if c.startswith(prefix):
            return c[len(prefix):]
    return ""


def collect_split_stats(path: Path) -> dict:
    """Collect per-split statistics in one streaming pass."""
    n_sents    = 0
    n_tokens   = 0
    upos_cnt:  Counter = Counter()
    lang_cnt:  Counter = Counter()
    boundaries = 0
    sent_ids:  set[str] = set()
    doc_ids:   set[str] = set()

    for sent in iter_sentences(path):
        n_sents += 1
        sent_ids.add(get_comment_value(sent["comments"], "sent_id"))
        doc_id = get_comment_value(sent["comments"], "doc_id")
        if doc_id:
            doc_ids.add(doc_id)

        real_langs: list[str] = []
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            n_tokens += 1
            upos_cnt[tok["fields"][F_UPOS]] += 1
            lang = get_misc_lang(tok["fields"][F_MISC])
            if lang:
                lang_cnt[lang] += 1
            real_langs.append(lang)

        for i in range(len(real_langs) - 1):
            if {real_langs[i], real_langs[i + 1]} == {"KO", "EN"}:
                boundaries += 1

    return {
        "n_sents":    n_sents,
        "n_tokens":   n_tokens,
        "upos":       dict(upos_cnt),
        "lang":       dict(lang_cnt),
        "boundaries": boundaries,
        "sent_ids":   sent_ids,
        "doc_ids":    doc_ids,
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}" if total else "0.0"


# ---------------------------------------------------------------------------
# Docsplit statistics CSVs
# ---------------------------------------------------------------------------

def write_docsplit_stats(all_stats: dict[str, dict]) -> None:
    """Write four docsplit statistics CSVs to results/."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    # dataset_statistics
    ds_path = RESULTS / "docsplit_dataset_statistics.csv"
    with ds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sentences", "tokens",
                    "KO", "EN", "MIXED", "OTHER", "cs_boundaries"])
        for name, s in all_stats.items():
            w.writerow([
                name, s["n_sents"], s["n_tokens"],
                s["lang"].get("KO", 0), s["lang"].get("EN", 0),
                s["lang"].get("MIXED", 0), s["lang"].get("OTHER", 0),
                s["boundaries"],
            ])
    print(f"  Wrote {ds_path.name}")

    # pos_distribution
    all_tags = sorted({tag for s in all_stats.values() for tag in s["upos"]})
    pos_path = RESULTS / "docsplit_pos_distribution.csv"
    with pos_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + all_tags)
        for name, s in all_stats.items():
            w.writerow([name] + [s["upos"].get(tag, 0) for tag in all_tags])
    print(f"  Wrote {pos_path.name}")

    # language_distribution
    lang_path = RESULTS / "docsplit_language_distribution.csv"
    with lang_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "KO", "EN", "MIXED", "OTHER",
                    "KO_pct", "EN_pct", "MIXED_pct", "OTHER_pct"])
        for name, s in all_stats.items():
            total = sum(s["lang"].values())
            w.writerow([
                name,
                s["lang"].get("KO", 0), s["lang"].get("EN", 0),
                s["lang"].get("MIXED", 0), s["lang"].get("OTHER", 0),
                _pct(s["lang"].get("KO", 0), total),
                _pct(s["lang"].get("EN", 0), total),
                _pct(s["lang"].get("MIXED", 0), total),
                _pct(s["lang"].get("OTHER", 0), total),
            ])
    print(f"  Wrote {lang_path.name}")

    # boundary_statistics
    bnd_path = RESULTS / "docsplit_boundary_statistics.csv"
    with bnd_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sentences", "tokens", "cs_boundaries",
                    "boundaries_per_sentence", "boundaries_per_1k_tokens", "MIXED_tokens"])
        for name, s in all_stats.items():
            bps  = f"{s['boundaries'] / s['n_sents']:.3f}" if s["n_sents"] else ""
            bp1k = f"{1000 * s['boundaries'] / s['n_tokens']:.2f}" if s["n_tokens"] else ""
            w.writerow([
                name, s["n_sents"], s["n_tokens"], s["boundaries"],
                bps, bp1k, s["lang"].get("MIXED", 0),
            ])
    print(f"  Wrote {bnd_path.name}")


# ---------------------------------------------------------------------------
# Comparison CSV
# ---------------------------------------------------------------------------

def write_comparison(
    step1_stats: dict[str, dict],
    step2_stats: dict[str, dict],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "docsplit_vs_sentence_split_comparison.csv"

    rows = []
    for split in ("train", "dev", "test"):
        s1 = step1_stats.get(split, {})
        s2 = step2_stats.get(split, {})

        # Overlap check: sent_ids
        s1_sids = s1.get("sent_ids", set())
        s2_sids = s2.get("sent_ids", set())

        # For within-split leakage, check if any sent_id in this split appears in another split
        # We compute this globally; here just report per-split sent/doc counts
        s2_dids = s2.get("doc_ids", set())

        def tok_pct(stats: dict, lang: str) -> str:
            total = stats.get("n_tokens", 0)
            return _pct(stats.get("lang", {}).get(lang, 0), total)

        rows.append({
            "split":                  split,
            "step":                   "step1_sentence",
            "sentences":              s1.get("n_sents", ""),
            "tokens":                 s1.get("n_tokens", ""),
            "KO_pct":                 tok_pct(s1, "KO"),
            "EN_pct":                 tok_pct(s1, "EN"),
            "MIXED_pct":              tok_pct(s1, "MIXED"),
            "OTHER_pct":              tok_pct(s1, "OTHER"),
            "cs_boundaries":          s1.get("boundaries", ""),
            "boundaries_per_sent":    f"{s1['boundaries']/s1['n_sents']:.3f}" if s1.get("n_sents") else "",
            "unique_sent_ids":        len(s1_sids),
            "unique_doc_ids":         "N/A",
            "note":                   "",
        })
        rows.append({
            "split":                  split,
            "step":                   "step2_document",
            "sentences":              s2.get("n_sents", ""),
            "tokens":                 s2.get("n_tokens", ""),
            "KO_pct":                 tok_pct(s2, "KO"),
            "EN_pct":                 tok_pct(s2, "EN"),
            "MIXED_pct":              tok_pct(s2, "MIXED"),
            "OTHER_pct":              tok_pct(s2, "OTHER"),
            "cs_boundaries":          s2.get("boundaries", ""),
            "boundaries_per_sent":    f"{s2['boundaries']/s2['n_sents']:.3f}" if s2.get("n_sents") else "",
            "unique_sent_ids":        len(s2_sids),
            "unique_doc_ids":         len(s2_dids),
            "note":                   "",
        })

    fieldnames = [
        "split", "step", "sentences", "tokens",
        "KO_pct", "EN_pct", "MIXED_pct", "OTHER_pct",
        "cs_boundaries", "boundaries_per_sent",
        "unique_sent_ids", "unique_doc_ids", "note",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Cross-split leakage check
# ---------------------------------------------------------------------------

def check_leakage(split_stats: dict[str, dict], label: str) -> None:
    """Check that no sent_id or doc_id appears in more than one split."""
    all_sent_ids_per_split = {k: v["sent_ids"] for k, v in split_stats.items()}
    all_doc_ids_per_split  = {k: v["doc_ids"]  for k, v in split_stats.items()}

    splits = list(split_stats.keys())
    sent_overlap = 0
    doc_overlap  = 0
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            sa, sb = splits[i], splits[j]
            s_overlap = all_sent_ids_per_split[sa] & all_sent_ids_per_split[sb]
            d_overlap = all_doc_ids_per_split[sa]  & all_doc_ids_per_split[sb]
            if s_overlap:
                print(f"  [{label}] [WARN] sent_id overlap between {sa} and {sb}: {len(s_overlap)} IDs")
                sent_overlap += len(s_overlap)
            if d_overlap:
                print(f"  [{label}] [WARN] doc_id overlap between {sa} and {sb}: {len(d_overlap)} IDs")
                doc_overlap += len(d_overlap)

    if not sent_overlap:
        print(f"  [{label}] sent_id: no overlap across splits [OK]")
    if all_doc_ids_per_split.get("train"):  # only check if doc_ids exist
        if not doc_overlap:
            print(f"  [{label}] doc_id:  no overlap across splits [OK]")
    else:
        print(f"  [{label}] doc_id:  not present (sentence-level split)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Collect Step 1 stats ----
    print("Collecting Step 1 (sentence-level) statistics ...")
    step1_stats: dict[str, dict] = {}
    for split, path in STEP1.items():
        if not path.exists():
            print(f"  SKIP {split}: {path.name} not found")
            continue
        print(f"  {split} ...")
        step1_stats[split] = collect_split_stats(path)

    # ---- Collect Step 2 stats ----
    print("Collecting Step 2 (document-level) statistics ...")
    step2_stats: dict[str, dict] = {}
    all_docsplit_stats: dict[str, dict] = {}

    for split, path in STEP2.items():
        if not path.exists():
            print(f"  SKIP {split}: {path.name} not found")
            continue
        print(f"  {split} ...")
        s = collect_split_stats(path)
        step2_stats[split] = s
        all_docsplit_stats[f"panyo_docsplit_{split}"] = s

    if STEP2_FULL.exists():
        print(f"  full ...")
        all_docsplit_stats["panyo_docsplit_full"] = collect_split_stats(STEP2_FULL)

    # ---- Cross-split leakage checks ----
    print("\nCross-split leakage check (Step 1) ...")
    check_leakage(step1_stats, "step1")

    print("Cross-split leakage check (Step 2) ...")
    check_leakage(step2_stats, "step2")

    # ---- Write outputs ----
    print("\nWriting docsplit statistics CSVs ...")
    if all_docsplit_stats:
        write_docsplit_stats(all_docsplit_stats)

    print("Writing comparison CSV ...")
    write_comparison(step1_stats, step2_stats)

    # ---- Console summary ----
    print()
    print("=== Split comparison (tokens) ===")
    header = f"{'Split':<5}  {'Step':<16}  {'Sents':>7}  {'Tokens':>9}  "
    header += f"{'KO%':>5}  {'EN%':>5}  {'Bnd/sent':>9}"
    print(header)
    print("-" * 65)
    for split in ("train", "dev", "test"):
        for label, stats in [("step1", step1_stats), ("step2", step2_stats)]:
            s = stats.get(split)
            if not s:
                continue
            total = s["n_tokens"]
            ko_pct  = 100 * s["lang"].get("KO", 0) / total if total else 0
            en_pct  = 100 * s["lang"].get("EN", 0) / total if total else 0
            bps     = s["boundaries"] / s["n_sents"] if s["n_sents"] else 0
            print(
                f"{split:<5}  {label:<16}  {s['n_sents']:>7,}  {s['n_tokens']:>9,}  "
                f"{ko_pct:>4.1f}%  {en_pct:>4.1f}%  {bps:>9.3f}"
            )

    print()
    print("Note: Sentence-level split (Step 1) may allow paragraph/document leakage.")
    print("      Document-level split (Step 2) keeps JSONL records together.")
    print("      Use document-level split for main experiments if validation passes.")


if __name__ == "__main__":
    main()
