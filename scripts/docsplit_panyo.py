"""
docsplit_panyo.py - Document-level 80/10/10 train/dev/test split of the Panyo corpus.

Reads the sentence→doc mapping produced by align_jsonl_conllu.py, groups sentences
into documents, shuffles documents with seed 42, then assigns documents to splits
targeting 80/10/10 by token count.

Inputs:
    data/processed/panyo_original_adp_lang_labeled.conllu
    data/processed/panyo_jsonl_doc_sentence_map.csv

Outputs:
    data/processed/panyo_original_adp_docsplit_full.conllu   (all sentences, with doc_id)
    data/processed/panyo_original_adp_docsplit_train.conllu
    data/processed/panyo_original_adp_docsplit_dev.conllu
    data/processed/panyo_original_adp_docsplit_test.conllu

Step 1 files are NOT touched:
    panyo_original_adp_train.conllu
    panyo_original_adp_dev.conllu
    panyo_original_adp_test.conllu

Each sentence in the output gets two new comment lines prepended:
    # doc_id = jsonl_<N>
    # split  = train|dev|test          (split files only; full gets no # split line)

Original # sent_id, # text, and # split = train/dev/test comments are preserved.
No token fields are modified.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, get_misc_lang, iter_sentences, write_sentences

ROOT      = Path(__file__).parent.parent
DATA      = ROOT / "data"
RESULTS   = ROOT / "results"
PROC      = DATA / "processed"

LABELED_PATH = PROC / "panyo_original_adp_lang_labeled.conllu"
MAP_PATH     = PROC / "panyo_jsonl_doc_sentence_map.csv"

OUT_FULL  = PROC / "panyo_original_adp_docsplit_full.conllu"
OUT_TRAIN = PROC / "panyo_original_adp_docsplit_train.conllu"
OUT_DEV   = PROC / "panyo_original_adp_docsplit_dev.conllu"
OUT_TEST  = PROC / "panyo_original_adp_docsplit_test.conllu"

# Step 1 files — must not be overwritten
STEP1_FILES = {
    PROC / "panyo_original_adp_train.conllu",
    PROC / "panyo_original_adp_dev.conllu",
    PROC / "panyo_original_adp_test.conllu",
}

SEED         = 42
TRAIN_TARGET = 0.80
DEV_TARGET   = 0.10
LANG_LABELS  = ("KO", "EN", "MIXED", "OTHER")


# ---------------------------------------------------------------------------
# Load mapping
# ---------------------------------------------------------------------------

def load_map(path: Path) -> dict[str, str]:
    """Return {sent_id: doc_id} from the sentence map CSV."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["sent_id"]] = row["doc_id"]
    return mapping


def load_map_ordered(path: Path) -> list[dict]:
    """Return rows in CSV order."""
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Sentence stats helpers
# ---------------------------------------------------------------------------

def count_cs_boundaries(real_langs: list[str]) -> int:
    n = 0
    for i in range(len(real_langs) - 1):
        a, b = real_langs[i], real_langs[i + 1]
        if {a, b} == {"KO", "EN"}:
            n += 1
    return n


def sent_stats(sent: dict) -> dict:
    lang_cnt: Counter = Counter()
    upos_cnt: Counter = Counter()
    real_langs: list[str] = []
    n_tokens = 0
    for tok in sent["tokens"]:
        if not tok["is_real"]:
            continue
        n_tokens += 1
        upos_cnt[tok["fields"][F_UPOS]] += 1
        lang = get_misc_lang(tok["fields"][F_MISC])
        if lang:
            lang_cnt[lang] += 1
        real_langs.append(lang)
    boundaries = count_cs_boundaries(real_langs)
    return {
        "n_tokens": n_tokens,
        "lang":     dict(lang_cnt),
        "upos":     dict(upos_cnt),
        "boundaries": boundaries,
    }


# ---------------------------------------------------------------------------
# Comment helpers
# ---------------------------------------------------------------------------

def prepend_comments(sent: dict, new_comments: list[str]) -> dict:
    """Return a shallow-copy sentence with new_comments prepended."""
    return {
        "comments": new_comments + sent["comments"],
        "tokens":   sent["tokens"],
    }


def get_sent_id(sent: dict) -> str:
    for c in sent["comments"]:
        if c.startswith("# sent_id = "):
            return c[len("# sent_id = "):]
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Safety: confirm we will not overwrite Step 1 files
    for p in [OUT_FULL, OUT_TRAIN, OUT_DEV, OUT_TEST]:
        if p in STEP1_FILES:
            sys.exit(f"ERROR: output path {p} would overwrite a Step 1 file!")

    if not LABELED_PATH.exists():
        sys.exit(f"ERROR: {LABELED_PATH} not found")
    if not MAP_PATH.exists():
        sys.exit(f"ERROR: {MAP_PATH} not found.\nRun scripts/align_jsonl_conllu.py first.")

    # -----------------------------------------------------------------------
    # 1. Load sentence→doc mapping
    # -----------------------------------------------------------------------
    print("Loading sentence->doc mapping ...")
    sent_to_doc = load_map(MAP_PATH)
    unassigned  = sum(1 for v in sent_to_doc.values() if v == "UNASSIGNED")
    print(f"  {len(sent_to_doc):,} sentences mapped  ({unassigned:,} UNASSIGNED)")

    # -----------------------------------------------------------------------
    # 2. Load all labeled sentences in order
    # -----------------------------------------------------------------------
    print("Loading labeled sentences ...")
    all_sents = list(iter_sentences(LABELED_PATH))
    print(f"  {len(all_sents):,} sentences")

    # -----------------------------------------------------------------------
    # 3. Group sentences into documents (preserve sentence order within doc)
    # -----------------------------------------------------------------------
    doc_sents:   dict[str, list[dict]] = {}  # doc_id → [sent, ...]
    doc_order:   list[str]             = []  # insertion-order doc_ids
    unassigned_sents: list[dict]       = []

    for sent in all_sents:
        sid = get_sent_id(sent)
        doc_id = sent_to_doc.get(sid, "UNASSIGNED")
        if doc_id == "UNASSIGNED":
            unassigned_sents.append(sent)
            continue
        if doc_id not in doc_sents:
            doc_sents[doc_id] = []
            doc_order.append(doc_id)
        doc_sents[doc_id].append(sent)

    n_docs = len(doc_order)
    total_tokens = sum(
        sum(1 for t in s["tokens"] if t["is_real"])
        for slist in doc_sents.values()
        for s in slist
    )
    print(f"  {n_docs:,} documents  |  {total_tokens:,} tokens  |  {len(unassigned_sents):,} unassigned sentences")

    if unassigned_sents:
        print(f"  [WARN] {len(unassigned_sents):,} sentences are UNASSIGNED and will be excluded from splits.")

    # -----------------------------------------------------------------------
    # 4. Shuffle documents with fixed seed
    # -----------------------------------------------------------------------
    rng = random.Random(SEED)
    shuffled_doc_ids = doc_order[:]
    rng.shuffle(shuffled_doc_ids)

    # -----------------------------------------------------------------------
    # 5. Assign documents to splits by cumulative token count (80/10/10)
    # -----------------------------------------------------------------------
    train_tok_target = int(total_tokens * TRAIN_TARGET)
    dev_tok_target   = int(total_tokens * (TRAIN_TARGET + DEV_TARGET))

    doc_split: dict[str, str] = {}
    cumulative = 0
    for doc_id in shuffled_doc_ids:
        n_tok = sum(1 for s in doc_sents[doc_id] for t in s["tokens"] if t["is_real"])
        if cumulative < train_tok_target:
            doc_split[doc_id] = "train"
        elif cumulative < dev_tok_target:
            doc_split[doc_id] = "dev"
        else:
            doc_split[doc_id] = "test"
        cumulative += n_tok

    split_doc_counts  = Counter(doc_split.values())
    split_sent_counts: Counter = Counter()
    split_tok_counts:  Counter = Counter()
    split_lang_counts: dict[str, Counter] = {s: Counter() for s in ("train", "dev", "test")}
    split_boundaries:  Counter = Counter()

    # -----------------------------------------------------------------------
    # 6. Write full output (all docs, in original order, with # doc_id prepended)
    # -----------------------------------------------------------------------
    print(f"Writing {OUT_FULL.name} ...")
    full_sents_out: list[dict] = []
    for doc_id in doc_order:
        for sent in doc_sents[doc_id]:
            full_sents_out.append(prepend_comments(sent, [f"# doc_id = {doc_id}"]))
    write_sentences(full_sents_out, OUT_FULL)
    print(f"  {len(full_sents_out):,} sentences")

    # -----------------------------------------------------------------------
    # 7. Write train/dev/test splits
    # -----------------------------------------------------------------------
    split_sents_out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for doc_id in doc_order:
        split = doc_split[doc_id]
        for sent in doc_sents[doc_id]:
            s_stats = sent_stats(sent)
            split_sent_counts[split] += 1
            split_tok_counts[split]  += s_stats["n_tokens"]
            split_boundaries[split]  += s_stats["boundaries"]
            for lbl, cnt in s_stats["lang"].items():
                split_lang_counts[split][lbl] += cnt

            new_sent = prepend_comments(
                sent,
                [f"# doc_id = {doc_id}", f"# split = {split}"],
            )
            split_sents_out[split].append(new_sent)

    for split_name, out_path in [
        ("train", OUT_TRAIN), ("dev", OUT_DEV), ("test", OUT_TEST)
    ]:
        print(f"Writing {out_path.name} ...")
        write_sentences(split_sents_out[split_name], out_path)

    # -----------------------------------------------------------------------
    # 8. Print summary
    # -----------------------------------------------------------------------
    print()
    print("=== Document-level split summary ===")
    print(f"{'Split':<6}  {'Docs':>6}  {'Sents':>7}  {'Tokens':>9}  "
          f"{'KO':>7}  {'EN':>7}  {'MIXED':>5}  {'OTHER':>7}  {'Boundaries':>10}")
    print("-" * 80)
    for split in ("train", "dev", "test"):
        lc = split_lang_counts[split]
        tok = split_tok_counts[split]
        print(
            f"{split:<6}  "
            f"{split_doc_counts[split]:>6,}  "
            f"{split_sent_counts[split]:>7,}  "
            f"{tok:>9,}  "
            f"{lc.get('KO',0):>7,}  "
            f"{lc.get('EN',0):>7,}  "
            f"{lc.get('MIXED',0):>5,}  "
            f"{lc.get('OTHER',0):>7,}  "
            f"{split_boundaries[split]:>10,}"
        )
    print()
    total_out = sum(split_tok_counts.values())
    for split in ("train", "dev", "test"):
        pct = 100 * split_tok_counts[split] / total_out if total_out else 0
        bps = split_boundaries[split] / split_sent_counts[split] if split_sent_counts[split] else 0
        print(f"  {split}: {pct:.1f}% of tokens  |  {bps:.3f} boundaries/sentence")

    if unassigned_sents:
        print(f"\n[WARN] {len(unassigned_sents):,} sentences were UNASSIGNED and excluded.")


if __name__ == "__main__":
    main()
