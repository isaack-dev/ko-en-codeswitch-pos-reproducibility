"""
validate_docsplit.py - Validate the Step 2 document-level split files.

Does NOT modify scripts/validate_conllu.py (which covers Step 1 files).

Checks:
  1. Each generated docsplit .conllu file has valid 10-column lines.
  2. Each real token has Lang=... in MISC.
  3. No UPOS labels were changed compared with panyo_original_adp_lang_labeled.conllu.
  4. No sent_id appears in more than one split (train/dev/test).
  5. No doc_id appears in more than one split.
  6. train + dev + test sentences cover exactly the same set as the docsplit_full file.
  7. ADP labels were not converted to PART.

Usage (from repo root):
    python scripts/validate_docsplit.py

Exit 0 = all checks passed, 1 = at least one check failed.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, iter_sentences

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"

LABELED_PATH  = PROC / "panyo_original_adp_lang_labeled.conllu"
FULL_PATH     = PROC / "panyo_original_adp_docsplit_full.conllu"

SPLIT_PATHS = {
    "train": PROC / "panyo_original_adp_docsplit_train.conllu",
    "dev":   PROC / "panyo_original_adp_docsplit_dev.conllu",
    "test":  PROC / "panyo_original_adp_docsplit_test.conllu",
}

VALID_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

MAX_ERR_SHOWN = 20


def get_comment_value(comments: list[str], key: str) -> str:
    prefix = f"# {key} = "
    for c in comments:
        if c.startswith(prefix):
            return c[len(prefix):]
    return ""


# ---------------------------------------------------------------------------
# Check 1+2: 10-column structure AND Lang= present
# ---------------------------------------------------------------------------

def check_structure_and_lang(path: Path) -> tuple[int, int, set[str], set[str]]:
    """
    Returns (errors, n_sents, sent_id_set, doc_id_set).
    Checks 10-column structure, valid UPOS, and Lang= in MISC for real tokens.
    """
    errors  = 0
    shown   = 0
    n_sents = 0
    sent_ids: set[str] = set()
    doc_ids:  set[str] = set()

    for sent in iter_sentences(path):
        n_sents += 1
        sid = get_comment_value(sent["comments"], "sent_id")
        did = get_comment_value(sent["comments"], "doc_id")
        if sid:
            sent_ids.add(sid)
        if did:
            doc_ids.add(did)

        for tok in sent["tokens"]:
            fields = tok["fields"]
            if len(fields) != 10:
                if shown < MAX_ERR_SHOWN:
                    print(f"    [FAIL] {path.name} sent {n_sents}: wrong field count {len(fields)}")
                    shown += 1
                errors += 1
                continue
            if not tok["is_real"]:
                continue
            upos = fields[F_UPOS]
            if upos not in VALID_UPOS:
                if shown < MAX_ERR_SHOWN:
                    print(f"    [FAIL] {path.name} sent {n_sents} tok {fields[0]}: invalid UPOS '{upos}'")
                    shown += 1
                errors += 1
            misc = fields[F_MISC]
            if not any(p.startswith("Lang=") for p in misc.split("|")):
                if shown < MAX_ERR_SHOWN:
                    print(f"    [FAIL] {path.name} sent {n_sents} form='{fields[1]}': missing Lang=")
                    shown += 1
                errors += 1

    if shown >= MAX_ERR_SHOWN and errors > shown:
        print(f"    ... (suppressed {errors - shown} additional errors)")
    return errors, n_sents, sent_ids, doc_ids


# ---------------------------------------------------------------------------
# Check 3: UPOS unchanged vs original labeled file
# ---------------------------------------------------------------------------

def check_upos_unchanged(orig_path: Path, docsplit_path: Path) -> int:
    """
    Compare UPOS sequences incrementally. Returns number of differing sentences.
    Reads both files in parallel without loading everything into memory.
    """
    errors = 0
    shown  = 0

    # Build a map from sent_id → UPOS sequence from the original labeled file
    print(f"    Loading original UPOS map from {orig_path.name} ...")
    orig_upos: dict[str, list[str]] = {}
    for sent in iter_sentences(orig_path):
        sid   = get_comment_value(sent["comments"], "sent_id")
        upos  = [t["fields"][F_UPOS] for t in sent["tokens"] if t["is_real"]]
        orig_upos[sid] = upos

    for sent in iter_sentences(docsplit_path):
        sid  = get_comment_value(sent["comments"], "sent_id")
        upos = [t["fields"][F_UPOS] for t in sent["tokens"] if t["is_real"]]
        expected = orig_upos.get(sid)
        if expected is None:
            if shown < MAX_ERR_SHOWN:
                print(f"    [FAIL] sent_id '{sid}' not found in original")
                shown += 1
            errors += 1
        elif upos != expected:
            if shown < MAX_ERR_SHOWN:
                print(f"    [FAIL] sent_id '{sid}': UPOS changed  orig={expected}  new={upos}")
                shown += 1
            errors += 1

    return errors


# ---------------------------------------------------------------------------
# Check 4+5: No sent_id or doc_id overlap across splits
# ---------------------------------------------------------------------------

def check_cross_split_overlap(split_data: dict[str, tuple]) -> int:
    """
    split_data: {split_name: (sent_ids, doc_ids)}
    Returns total overlap count.
    """
    errors = 0
    splits = list(split_data.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            na, nb = splits[i], splits[j]
            sa, da = split_data[na]
            sb, db = split_data[nb]

            sent_overlap = sa & sb
            doc_overlap  = da & db

            if sent_overlap:
                print(f"    [FAIL] sent_id overlap {na}n{nb}: {len(sent_overlap)} IDs")
                errors += len(sent_overlap)
            else:
                print(f"    [OK]   sent_id: no overlap {na}n{nb}")

            if da and db:
                if doc_overlap:
                    print(f"    [FAIL] doc_id overlap {na}n{nb}: {len(doc_overlap)} IDs")
                    errors += len(doc_overlap)
                else:
                    print(f"    [OK]   doc_id:  no overlap {na}n{nb}")

    return errors


# ---------------------------------------------------------------------------
# Check 6: train+dev+test covers exactly the same sentences as docsplit_full
# ---------------------------------------------------------------------------

def check_coverage(full_path: Path, split_paths: dict[str, Path]) -> int:
    errors = 0

    full_sids: set[str] = set()
    for sent in iter_sentences(full_path):
        sid = get_comment_value(sent["comments"], "sent_id")
        full_sids.add(sid)

    union_sids: set[str] = set()
    for p in split_paths.values():
        if p.exists():
            for sent in iter_sentences(p):
                sid = get_comment_value(sent["comments"], "sent_id")
                union_sids.add(sid)

    only_in_full  = full_sids - union_sids
    only_in_splits = union_sids - full_sids

    if only_in_full:
        print(f"    [FAIL] {len(only_in_full)} sent_ids in full but not in any split")
        errors += 1
    else:
        print(f"    [OK]   all full sent_ids appear in at least one split")

    if only_in_splits:
        print(f"    [FAIL] {len(only_in_splits)} sent_ids in splits but not in full")
        errors += 1
    else:
        print(f"    [OK]   no extra sent_ids in splits")

    return errors


# ---------------------------------------------------------------------------
# Check 7: No ADP → PART conversion
# ---------------------------------------------------------------------------

def check_no_adp_to_part(path: Path) -> int:
    part_count = 0
    for sent in iter_sentences(path):
        for tok in sent["tokens"]:
            if tok["is_real"] and tok["fields"][F_UPOS] == "PART":
                part_count += 1
    if part_count > 0:
        print(f"    [FAIL] {part_count} PART tokens found — ADP may have been converted")
    else:
        print(f"    [OK]   No PART tokens found")
    return part_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check(path: Path) -> tuple[bool, int, set, set]:
    """Run structure+lang check and print result. Returns (passed, n_sents, sids, dids)."""
    label = path.name
    if not path.exists():
        print(f"  [SKIP] {label}: file not found")
        return True, 0, set(), set()
    print(f"  Checking {label} ...")
    errors, n_sents, sids, dids = check_structure_and_lang(path)
    if errors == 0:
        print(f"    [PASS] {n_sents:,} sentences, 0 structure/lang errors")
    else:
        print(f"    [FAIL] {n_sents:,} sentences, {errors} errors")
    return errors == 0, n_sents, sids, dids


def main() -> None:
    all_pass = True

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Check 1+2: Structure and Lang= on docsplit files")
    print("=" * 60)
    split_ids: dict[str, tuple[set, set]] = {}

    passed, _, full_sids, full_dids = _check(FULL_PATH)
    if not passed:
        all_pass = False

    for split_name, path in SPLIT_PATHS.items():
        passed, _, sids, dids = _check(path)
        if not passed:
            all_pass = False
        split_ids[split_name] = (sids, dids)

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Check 3: UPOS unchanged vs panyo_original_adp_lang_labeled.conllu")
    print("=" * 60)
    if LABELED_PATH.exists():
        for split_name, path in SPLIT_PATHS.items():
            if not path.exists():
                continue
            print(f"  Comparing {path.name} ...")
            errs = check_upos_unchanged(LABELED_PATH, path)
            if errs == 0:
                print(f"    [PASS] No UPOS changes in {split_name}")
            else:
                print(f"    [FAIL] {errs} sentences have changed UPOS in {split_name}")
                all_pass = False
    else:
        print(f"  [SKIP] {LABELED_PATH.name} not found")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Check 4+5: No sent_id / doc_id overlap across splits")
    print("=" * 60)
    overlap_errors = check_cross_split_overlap(split_ids)
    if overlap_errors > 0:
        all_pass = False

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Check 6: train+dev+test covers exactly docsplit_full sentences")
    print("=" * 60)
    if FULL_PATH.exists():
        cov_errors = check_coverage(FULL_PATH, SPLIT_PATHS)
        if cov_errors > 0:
            all_pass = False
    else:
        print(f"  [SKIP] {FULL_PATH.name} not found")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Check 7: No ADP->PART conversion")
    print("=" * 60)
    for split_name, path in SPLIT_PATHS.items():
        if not path.exists():
            continue
        print(f"  Checking {path.name} ...")
        part_errs = check_no_adp_to_part(path)
        if part_errs > 0:
            all_pass = False

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: ALL DOCSPLIT CHECKS PASSED")
    else:
        print("RESULT: ONE OR MORE DOCSPLIT CHECKS FAILED")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
