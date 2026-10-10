"""
dataset_statistics.py - Generate dataset statistics CSV files.

Reads:
    data/processed/panyo_original_adp_lang_labeled.conllu  (full labeled corpus)
    data/processed/panyo_original_adp_train.conllu
    data/processed/panyo_original_adp_dev.conllu
    data/processed/panyo_original_adp_test.conllu
    data/en_gum-ud-{train,dev,test}.conllu           (monolingual reference; optional)
    data/ko_kaist-ud-{train,dev,test}.conllu          (monolingual reference; optional)

Writes:
    results/dataset_statistics.csv    - sentence/token counts and language breakdown
    results/pos_distribution.csv      - UPOS tag distribution per dataset
    results/language_distribution.csv - language label counts and percentages
    results/boundary_statistics.csv   - code-switching boundary counts

Code-switching boundary definition (Step 1, simple adjacent-only version):
    A boundary is counted when two directly adjacent real tokens in the same sentence
    have labels KO and EN (in either order). OTHER and MIXED tokens between them are
    NOT skipped; they break the adjacency. This is documented as a limitation.
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

# Files that have Lang= labels
PANYO_FILES: dict[str, Path] = {
    "panyo_full":  DATA / "processed" / "panyo_original_adp_lang_labeled.conllu",
    "panyo_train": DATA / "processed" / "panyo_original_adp_train.conllu",
    "panyo_dev":   DATA / "processed" / "panyo_original_adp_dev.conllu",
    "panyo_test":  DATA / "processed" / "panyo_original_adp_test.conllu",
}

# Monolingual reference files (no Lang= labels)
MONO_FILES: dict[str, Path] = {
    "en_gum_train":   DATA / "en_gum-ud-train.conllu",
    "en_gum_dev":     DATA / "en_gum-ud-dev.conllu",
    "en_gum_test":    DATA / "en_gum-ud-test.conllu",
    "ko_kaist_train": DATA / "ko_kaist-ud-train.conllu",
    "ko_kaist_dev":   DATA / "ko_kaist-ud-dev.conllu",
    "ko_kaist_test":  DATA / "ko_kaist-ud-test.conllu",
}

LANG_LABELS = ("KO", "EN", "MIXED", "OTHER")


def _collect_stats(name: str, path: Path, has_lang: bool) -> dict:
    """
    Iterate through a CoNLL-U file once and collect all statistics.
    Returns a dict with keys: name, sentences, tokens, upos, lang, boundaries.
    """
    n_sents    = 0
    n_tokens   = 0
    upos_cnt: Counter = Counter()
    lang_cnt: Counter = Counter()
    boundaries = 0

    for sent in iter_sentences(path):
        n_sents += 1
        real_langs: list[str] = []

        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            n_tokens += 1
            upos_cnt[tok["fields"][F_UPOS]] += 1

            if has_lang:
                lang = get_misc_lang(tok["fields"][F_MISC])
                if lang:
                    lang_cnt[lang] += 1
                real_langs.append(lang)

        # Count direct adjacent KO<->EN transitions
        if has_lang:
            for i in range(len(real_langs) - 1):
                a, b = real_langs[i], real_langs[i + 1]
                if {a, b} == {"KO", "EN"}:
                    boundaries += 1

    return {
        "name":       name,
        "sentences":  n_sents,
        "tokens":     n_tokens,
        "upos":       dict(upos_cnt),
        "lang":       dict(lang_cnt),
        "boundaries": boundaries,
        "has_lang":   has_lang,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []

    print("Reading Panyo processed files ...")
    for name, path in PANYO_FILES.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"  {name} ...")
        all_stats.append(_collect_stats(name, path, has_lang=True))

    print("Reading monolingual reference files ...")
    for name, path in MONO_FILES.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"  {name} ...")
        all_stats.append(_collect_stats(name, path, has_lang=False))

    # -----------------------------------------------------------------------
    # 1. dataset_statistics.csv
    # -----------------------------------------------------------------------
    ds_path = RESULTS / "dataset_statistics.csv"
    with ds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "sentences", "tokens",
            "KO", "EN", "MIXED", "OTHER", "cs_boundaries",
        ])
        for s in all_stats:
            row = [
                s["name"], s["sentences"], s["tokens"],
                s["lang"].get("KO",    "") if s["has_lang"] else "",
                s["lang"].get("EN",    "") if s["has_lang"] else "",
                s["lang"].get("MIXED", "") if s["has_lang"] else "",
                s["lang"].get("OTHER", "") if s["has_lang"] else "",
                s["boundaries"]             if s["has_lang"] else "",
            ]
            w.writerow(row)
    print(f"\nWrote {ds_path}")

    # -----------------------------------------------------------------------
    # 2. pos_distribution.csv
    # -----------------------------------------------------------------------
    all_upos_tags = sorted({tag for s in all_stats for tag in s["upos"]})
    pos_path = RESULTS / "pos_distribution.csv"
    with pos_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + all_upos_tags)
        for s in all_stats:
            w.writerow([s["name"]] + [s["upos"].get(tag, 0) for tag in all_upos_tags])
    print(f"Wrote {pos_path}")

    # -----------------------------------------------------------------------
    # 3. language_distribution.csv
    # -----------------------------------------------------------------------
    lang_path = RESULTS / "language_distribution.csv"
    with lang_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "KO", "EN", "MIXED", "OTHER",
            "KO_pct", "EN_pct", "MIXED_pct", "OTHER_pct",
        ])
        for s in all_stats:
            if not s["has_lang"]:
                continue
            total = sum(s["lang"].values())
            def pct(k: str) -> str:
                return f"{100 * s['lang'].get(k, 0) / total:.1f}" if total else ""
            w.writerow([
                s["name"],
                s["lang"].get("KO",    0),
                s["lang"].get("EN",    0),
                s["lang"].get("MIXED", 0),
                s["lang"].get("OTHER", 0),
                pct("KO"), pct("EN"), pct("MIXED"), pct("OTHER"),
            ])
    print(f"Wrote {lang_path}")

    # -----------------------------------------------------------------------
    # 4. boundary_statistics.csv
    # -----------------------------------------------------------------------
    bnd_path = RESULTS / "boundary_statistics.csv"
    with bnd_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "sentences", "tokens",
            "cs_boundaries", "boundaries_per_sentence",
            "MIXED_tokens",
        ])
        for s in all_stats:
            if not s["has_lang"]:
                continue
            bps = (
                f"{s['boundaries'] / s['sentences']:.3f}"
                if s["sentences"] else ""
            )
            w.writerow([
                s["name"],
                s["sentences"],
                s["tokens"],
                s["boundaries"],
                bps,
                s["lang"].get("MIXED", 0),
            ])
    print(f"Wrote {bnd_path}")

    # -----------------------------------------------------------------------
    # Console summary for panyo_full
    # -----------------------------------------------------------------------
    full = next((s for s in all_stats if s["name"] == "panyo_full"), None)
    if full:
        print()
        print("=== Panyo corpus summary ===")
        print(f"  Sentences : {full['sentences']:,}")
        print(f"  Tokens    : {full['tokens']:,}")
        total = sum(full["lang"].values())
        for lbl in LANG_LABELS:
            cnt = full["lang"].get(lbl, 0)
            pct = 100 * cnt / total if total else 0
            print(f"  {lbl:<6s}    : {cnt:>8,}  ({pct:.1f}%)")
        print(f"  CS boundaries: {full['boundaries']:,}")
        print(f"  Boundaries/sentence: {full['boundaries']/full['sentences']:.3f}")


if __name__ == "__main__":
    main()
