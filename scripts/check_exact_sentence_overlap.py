#!/usr/bin/env python3
"""
check_exact_sentence_overlap.py
Exact normalized sentence-overlap audit for the Panyo document-level splits.

Reads panyo_original_adp_docsplit_{train,dev,test}.conllu produced by
docsplit_panyo.py.  Does NOT read GUM or KAIST files.  Does NOT alter any
split assignments, annotations, predictions, or experimental results.

Reports:
  1. train-dev exact sentence overlap
  2. train-test exact sentence overlap
  3. dev-test exact sentence overlap
  4. Within-split duplicate sentence counts
  5. Up to 20 example overlaps with sent_id and doc_id

Output: outputs/exact_sentence_overlap.txt  (also printed to stdout)
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT          = Path(__file__).parent.parent
PROC          = ROOT / "data" / "processed"
OUT_DIR       = ROOT / "outputs"

DEFAULT_TRAIN = PROC / "panyo_original_adp_docsplit_train.conllu"
DEFAULT_DEV   = PROC / "panyo_original_adp_docsplit_dev.conllu"
DEFAULT_TEST  = PROC / "panyo_original_adp_docsplit_test.conllu"

MAX_EXAMPLES  = 20

# A sentence key is (sent_id, doc_id).
SentKey = tuple[str, str]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Unicode NFKC + whitespace collapsing (case preserved)."""
    nfkc = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", nfkc).strip()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def read_split(path: Path) -> dict[str, list[SentKey]]:
    """
    Parse a docsplit CoNLL-U file.

    Returns {normalized_text: [(sent_id, doc_id), ...]} with one entry per
    sentence occurrence.  Text is taken from the '# text = ...' comment when
    present; otherwise reconstructed from FORM fields of real tokens.
    """
    result: dict[str, list[SentKey]] = defaultdict(list)
    sent_id:   str        = ""
    doc_id:    str        = ""
    text_line: str | None = None
    forms:     list[str]  = []

    def flush() -> None:
        nonlocal sent_id, doc_id, text_line, forms
        if not forms and text_line is None:
            return
        raw  = text_line if text_line is not None else " ".join(forms)
        norm = normalize(raw)
        if norm:
            result[norm].append((sent_id or "(unknown)", doc_id or "(unknown)"))
        sent_id    = ""
        doc_id     = ""
        text_line  = None
        forms      = []

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            elif line.startswith("# doc_id = "):
                doc_id = line[len("# doc_id = "):]
            elif line.startswith("# text = "):
                text_line = line[len("# text = "):]
            elif line.startswith("#"):
                pass  # other comments (# split = ..., etc.) ignored
            elif line == "":
                flush()
            else:
                cols = line.split("\t")
                # skip multiword ranges (e.g. "1-2") and empty nodes (e.g. "1.1")
                if len(cols) >= 2 and "-" not in cols[0] and "." not in cols[0]:
                    forms.append(cols[1])

    flush()  # last sentence when file has no trailing blank line
    return dict(result)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def within_dupe_stats(data: dict[str, list[SentKey]]) -> tuple[int, int]:
    """Return (unique_texts_that_appear_more_than_once, total_extra_occurrences)."""
    texts_dupe = sum(1 for v in data.values() if len(v) > 1)
    extra      = sum(len(v) - 1 for v in data.values() if len(v) > 1)
    return texts_dupe, extra


def cross_overlap(
    data_a: dict[str, list[SentKey]],
    data_b: dict[str, list[SentKey]],
) -> tuple[int, list[tuple[str, list[SentKey], list[SentKey]]]]:
    """
    Return (overlap_count, examples).

    examples is a list of up to MAX_EXAMPLES tuples:
        (norm_text, a_occurrences, b_occurrences)
    """
    overlap_keys = sorted(set(data_a) & set(data_b))
    examples = [
        (k, data_a[k], data_b[k])
        for k in overlap_keys[:MAX_EXAMPLES]
    ]
    return len(overlap_keys), examples


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(splits: dict[str, dict[str, list[SentKey]]]) -> list[str]:
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln("=== Panyo Document-Level Split: Exact Sentence Overlap Audit ===")
    ln()
    ln("Normalization: Unicode NFKC + whitespace collapsing (case preserved).")
    ln("Source files:  panyo_original_adp_docsplit_{train,dev,test}.conllu")
    ln("               (GUM and KAIST files are NOT used in this audit.)")
    ln()

    # ------------------------------------------------------------------
    # 4. Within-split duplicates + split sizes
    # ------------------------------------------------------------------
    ln("--- Split sizes and within-split duplicate counts ---")
    for name in ("train", "dev", "test"):
        data       = splits[name]
        total_occ  = sum(len(v) for v in data.values())
        unique     = len(data)
        texts_dupe, extra = within_dupe_stats(data)
        ln(
            f"  {name:5s}: {total_occ:8,} sentences | "
            f"{unique:8,} unique normalized texts | "
            f"{texts_dupe:,} texts appear >1x ({extra:,} extra occurrences)"
        )
    ln()

    # ------------------------------------------------------------------
    # 1-3. Cross-split overlap counts
    # ------------------------------------------------------------------
    pairs = [
        ("train", "dev"),
        ("train", "test"),
        ("dev",   "test"),
    ]
    ln("--- Cross-split exact overlap (unique normalized sentence texts) ---")
    pair_results: dict[tuple[str, str], tuple[int, list]] = {}
    for na, nb in pairs:
        count, examples = cross_overlap(splits[na], splits[nb])
        pair_results[(na, nb)] = (count, examples)
        ln(f"  {na} n {nb}:  {count:,} overlapping sentences")
    ln()

    # ------------------------------------------------------------------
    # 5. Example overlaps
    # ------------------------------------------------------------------
    ln(f"--- Examples (up to {MAX_EXAMPLES} per cross-split pair) ---")
    for na, nb in pairs:
        count, examples = pair_results[(na, nb)]
        ln()
        if not examples:
            ln(f"[{na} n {nb}]  No overlaps found.")
            continue
        shown = len(examples)
        ln(f"[{na} n {nb}]  {count:,} overlapping sentences; showing {shown}")
        for i, (norm_text, a_occ, b_occ) in enumerate(examples, 1):
            display = norm_text[:120] + ("..." if len(norm_text) > 120 else "")
            ln(f"  [{i}] TEXT: {display}")
            for sid, did in a_occ[:3]:
                ln(f"        {na}: sent_id={sid}  doc_id={did}")
            for sid, did in b_occ[:3]:
                ln(f"        {nb}: sent_id={sid}  doc_id={did}")

    ln()
    total_cross = sum(c for c, _ in pair_results.values())
    ln(f"Total cross-split overlapping sentence count (sum of three pairs): {total_cross:,}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact normalized sentence-overlap audit for Panyo document-level splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN,
                        help="Train split CoNLL-U file")
    parser.add_argument("--dev",   type=Path, default=DEFAULT_DEV,
                        help="Dev split CoNLL-U file")
    parser.add_argument("--test",  type=Path, default=DEFAULT_TEST,
                        help="Test split CoNLL-U file")
    parser.add_argument("--out",   type=Path,
                        default=OUT_DIR / "exact_sentence_overlap.txt",
                        help="Output summary file")
    args = parser.parse_args()

    for name, path in [("train", args.train), ("dev", args.dev), ("test", args.test)]:
        if not path.exists():
            sys.exit(
                f"ERROR: {name} split file not found: {path}\n"
                "Run scripts/docsplit_panyo.py first."
            )

    print("Reading split files ...")
    splits = {
        "train": read_split(args.train),
        "dev":   read_split(args.dev),
        "test":  read_split(args.test),
    }

    report_lines = build_report(splits)

    # Print to stdout (replace un-encodable chars for terminals that can't handle Korean)
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    for line in report_lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode(enc, errors="replace").decode(enc))

    # Write UTF-8 output file
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {args.out}")


if __name__ == "__main__":
    main()
