"""
validate_conllu.py - Validate CoNLL-U files generated in Step 1.

Checks performed on generated Panyo files:
  1. Every token line has exactly 10 tab-separated fields.
  2. Real tokens have a valid UPOS tag.
  3. Real tokens have Lang=... in the MISC field.
  4. Sentence boundaries (blank lines) are properly formed.

Additional check:
  5. UPOS comparison: original vs. language-labeled file.
     Verifies that no UPOS labels were changed during language labeling.

Usage (from repo root):
    python scripts/validate_conllu.py

Exit code 0 = all checks passed, 1 = one or more checks failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, iter_sentences

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

VALID_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

# Generated files that must have Lang= on every real token
GENERATED_FILES = [
    DATA / "processed" / "panyo_original_adp_lang_labeled.conllu",
    DATA / "processed" / "panyo_original_adp_train.conllu",
    DATA / "processed" / "panyo_original_adp_dev.conllu",
    DATA / "processed" / "panyo_original_adp_test.conllu",
]

ORIGINAL_PANYO = DATA / "kor_en_panyo_corpus.conllu"
LABELED_PANYO  = DATA / "processed" / "panyo_original_adp_lang_labeled.conllu"

# Maximum number of individual errors to print per file (to avoid flooding the console)
MAX_ERRORS_SHOWN = 20


def _validate_file(path: Path) -> tuple[int, int]:
    """
    Validate a generated CoNLL-U file.

    Returns (error_count, sentence_count).
    Prints individual errors up to MAX_ERRORS_SHOWN.
    """
    errors       = 0
    shown        = 0
    n_sents      = 0

    for sent in iter_sentences(path):
        n_sents += 1
        for tok in sent["tokens"]:
            fields = tok["fields"]

            # Check 10 columns (iter_sentences already raises on mismatch, but belt-and-suspenders)
            if len(fields) != 10:
                if shown < MAX_ERRORS_SHOWN:
                    print(
                        f"    [FAIL] sentence {n_sents}: "
                        f"expected 10 fields, got {len(fields)}: {fields}"
                    )
                    shown += 1
                errors += 1
                continue

            if not tok["is_real"]:
                continue  # skip multiword tokens and empty nodes

            # Check valid UPOS
            upos = fields[F_UPOS]
            if upos not in VALID_UPOS:
                if shown < MAX_ERRORS_SHOWN:
                    print(
                        f"    [FAIL] sentence {n_sents} token {fields[0]}: "
                        f"invalid UPOS '{upos}'"
                    )
                    shown += 1
                errors += 1

            # Check Lang= present in MISC
            misc = fields[F_MISC]
            if not any(p.startswith("Lang=") for p in misc.split("|")):
                if shown < MAX_ERRORS_SHOWN:
                    print(
                        f"    [FAIL] sentence {n_sents} token {fields[0]} "
                        f"form='{fields[1]}': missing Lang= in MISC ('{misc}')"
                    )
                    shown += 1
                errors += 1

    if shown == MAX_ERRORS_SHOWN and errors > shown:
        print(f"    ... (suppressed {errors - shown} additional errors)")

    return errors, n_sents


def _compare_upos(orig_path: Path, labeled_path: Path) -> int:
    """
    Compare UPOS tags between original and labeled file sentence-by-sentence.

    Returns the number of sentences with UPOS differences.
    Reads both files incrementally without loading them fully into memory.
    """
    errors  = 0
    shown   = 0
    sent_no = 0

    orig_gen    = iter_sentences(orig_path)
    labeled_gen = iter_sentences(labeled_path)

    for orig_sent, lab_sent in zip(orig_gen, labeled_gen):
        sent_no += 1
        orig_upos = [t["fields"][F_UPOS] for t in orig_sent["tokens"] if t["is_real"]]
        lab_upos  = [t["fields"][F_UPOS] for t in lab_sent["tokens"]  if t["is_real"]]

        if orig_upos != lab_upos:
            errors += 1
            if shown < MAX_ERRORS_SHOWN:
                print(
                    f"    [FAIL] sentence {sent_no}: UPOS changed "
                    f"orig={orig_upos} labeled={lab_upos}"
                )
                shown += 1

    # Check for leftover sentences in either generator
    orig_remaining = sum(1 for _ in orig_gen)
    lab_remaining  = sum(1 for _ in labeled_gen)
    if orig_remaining or lab_remaining:
        print(
            f"    [FAIL] sentence count mismatch: "
            f"original has {orig_remaining} extra, labeled has {lab_remaining} extra"
        )
        errors += 1

    return errors


def _check(path: Path) -> bool:
    """Validate one file and print PASS/FAIL. Returns True if no errors."""
    label = f"{path.parent.name}/{path.name}"
    if not path.exists():
        print(f"  [SKIP] {label}: file not found")
        return True  # missing file is not a generated-file failure (may not be created yet)

    print(f"  Checking {label} ...")
    errors, n_sents = _validate_file(path)
    if errors == 0:
        print(f"    [PASS] {n_sents:,} sentences, 0 errors")
    else:
        print(f"    [FAIL] {n_sents:,} sentences, {errors} errors")
    return errors == 0


def main() -> None:
    all_pass = True

    print("=" * 60)
    print("Validating generated Panyo files (Lang= required on all real tokens)")
    print("=" * 60)
    for path in GENERATED_FILES:
        if not _check(path):
            all_pass = False

    print()
    print("=" * 60)
    print("Comparing UPOS: original panyo vs. language-labeled file")
    print("=" * 60)
    if ORIGINAL_PANYO.exists() and LABELED_PANYO.exists():
        print(
            f"  Comparing {ORIGINAL_PANYO.name} vs {LABELED_PANYO.name} ..."
        )
        upos_errors = _compare_upos(ORIGINAL_PANYO, LABELED_PANYO)
        if upos_errors == 0:
            print("    [PASS] No UPOS changes detected")
        else:
            print(f"    [FAIL] {upos_errors} sentence(s) have changed UPOS labels")
            all_pass = False
    else:
        missing = []
        if not ORIGINAL_PANYO.exists():
            missing.append(ORIGINAL_PANYO.name)
        if not LABELED_PANYO.exists():
            missing.append(LABELED_PANYO.name)
        print(f"  [SKIP] Missing: {', '.join(missing)}")

    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
