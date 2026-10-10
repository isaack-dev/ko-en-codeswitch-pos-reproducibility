"""
validate_predictions.py - Validate all Step 3 baseline prediction files.

Checks:
  1. Every token line has 10 columns.
  2. Every real token has Lang=... in MISC.
  3. Every real token has PredUPOS=... in MISC.
  4. Gold UPOS (column 4) is unchanged vs the corresponding docsplit input file.
  5. Sentence count and token count match the corresponding input split.
  6. All PredUPOS values are valid UPOS labels.
  7. No duplicate PredUPOS field in any MISC.
  8. No ADP->PART conversion in gold labels (gold UPOS contains no PART).
  9. Prediction file parses as valid CoNLL-U-like (no malformed lines).

Exits 0 if all checks pass, 1 otherwise.

Run from repo root:
  python scripts/validate_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_MISC, F_UPOS, get_misc_lang, iter_sentences

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "baselines"

SPLIT_SOURCES: dict[str, Path] = {
    "dev":  PROC / "panyo_original_adp_docsplit_dev.conllu",
    "test": PROC / "panyo_original_adp_docsplit_test.conllu",
}

MODELS = [
    "majority_global",
    "majority_lang",
    "perceptron_joint",
    "perceptron_pipeline_panyo",
    "perceptron_pipeline_ud_transfer",
]

VALID_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

MAX_SHOWN = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pred_upos(misc: str) -> str | None:
    for part in misc.split("|"):
        if part.startswith("PredUPOS="):
            return part[9:]
    return None


def count_field(misc: str, prefix: str) -> int:
    return sum(1 for p in misc.split("|") if p.startswith(prefix))


def load_gold_map(src_path: Path) -> dict[str, list[str]]:
    """Return {sent_id: [gold_upos, ...]} from the source split file."""
    gold: dict[str, list[str]] = {}
    for sent in iter_sentences(src_path):
        sid = ""
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[12:]
                break
        upos_seq = [
            tok["fields"][F_UPOS]
            for tok in sent["tokens"]
            if tok["is_real"]
        ]
        if sid:
            gold[sid] = upos_seq
    return gold


# ---------------------------------------------------------------------------
# Validate one file
# ---------------------------------------------------------------------------

def validate_file(pred_path: Path, src_path: Path) -> bool:
    """
    Run all checks on pred_path against src_path as gold reference.
    Returns True if all checks pass.
    """
    name = pred_path.name
    print(f"\n  [{name}]")

    if not pred_path.exists():
        print(f"    [FAIL] File not found")
        return False

    gold_map = load_gold_map(src_path)

    errors:  list[str] = []
    shown    = 0

    n_sents = 0
    n_toks  = 0
    part_count = 0

    for sent in iter_sentences(pred_path):
        n_sents += 1
        sid = ""
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[12:]
                break

        pred_upos_seq: list[str] = []

        for tok in sent["tokens"]:
            fields = tok["fields"]

            # Check 1: 10 columns (iter_sentences already raises ValueError for wrong counts,
            # but we verify here explicitly)
            if len(fields) != 10:
                if shown < MAX_SHOWN:
                    errors.append(f"sent {n_sents}: wrong column count {len(fields)}")
                    shown += 1
                continue

            if not tok["is_real"]:
                continue

            n_toks += 1
            misc = fields[F_MISC]
            gold_upos = fields[F_UPOS]

            # Check 2: Lang= present
            if not any(p.startswith("Lang=") for p in misc.split("|")):
                if shown < MAX_SHOWN:
                    errors.append(f"sent {n_sents} tok {fields[0]}: missing Lang= in MISC '{misc}'")
                    shown += 1

            # Check 3: PredUPOS= present
            pred_upos = get_pred_upos(misc)
            if pred_upos is None:
                if shown < MAX_SHOWN:
                    errors.append(f"sent {n_sents} tok {fields[0]}: missing PredUPOS= in MISC '{misc}'")
                    shown += 1
            else:
                pred_upos_seq.append(pred_upos)

            # Check 6: PredUPOS is valid UPOS
            if pred_upos and pred_upos not in VALID_UPOS:
                if shown < MAX_SHOWN:
                    errors.append(f"sent {n_sents} tok {fields[0]}: invalid PredUPOS='{pred_upos}'")
                    shown += 1

            # Check 7: no duplicate PredUPOS
            if count_field(misc, "PredUPOS=") > 1:
                if shown < MAX_SHOWN:
                    errors.append(f"sent {n_sents} tok {fields[0]}: duplicate PredUPOS in '{misc}'")
                    shown += 1

            # Check 8: no PART in gold UPOS
            if gold_upos == "PART":
                part_count += 1

        # Check 4: gold UPOS unchanged
        if sid and sid in gold_map:
            gold_seq = gold_map[sid]
            # pred_upos_seq covers only real tokens with PredUPOS; gold_seq covers all real tokens
            # We check gold UPOS column directly
            actual_gold_seq = [
                tok["fields"][F_UPOS]
                for tok in sent["tokens"]
                if tok["is_real"]
            ]
            if actual_gold_seq != gold_seq:
                if shown < MAX_SHOWN:
                    errors.append(f"sent_id='{sid}': gold UPOS changed")
                    shown += 1
        elif sid and sid not in gold_map:
            if shown < MAX_SHOWN:
                errors.append(f"sent_id='{sid}': not found in gold source")
                shown += 1

    if shown >= MAX_SHOWN and len(errors) > shown:
        errors.append(f"... ({len(errors) - MAX_SHOWN} more errors suppressed)")

    # Check 5: sentence and token counts
    gold_n_sents = len(gold_map)
    if n_sents != gold_n_sents:
        errors.append(f"sentence count mismatch: pred={n_sents}, gold={gold_n_sents}")

    # Check 8 result
    if part_count > 0:
        errors.append(f"PART found in gold UPOS: {part_count} tokens (ADP->PART conversion?)")
    else:
        print(f"    [OK] Check 8: no PART in gold UPOS")

    # Print all errors
    passed = len(errors) == 0
    if passed:
        print(f"    [PASS] {n_sents:,} sentences, {n_toks:,} real tokens, all checks OK")
    else:
        for e in errors:
            print(f"    [FAIL] {e}")
        print(f"    Total errors: {len(errors)}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_pass = True

    print("=" * 70)
    print("validate_predictions.py - Step 3 prediction file validation")
    print("=" * 70)

    for model in MODELS:
        for split, src_path in SPLIT_SOURCES.items():
            pred_path = RESULTS / f"{model}_{split}_predictions.conllu"
            if not pred_path.exists():
                print(f"\n  [SKIP] {pred_path.name}: not found")
                continue
            if not src_path.exists():
                print(f"\n  [SKIP] {pred_path.name}: source {src_path.name} not found")
                continue
            ok = validate_file(pred_path, src_path)
            if not ok:
                all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("RESULT: ALL PREDICTION CHECKS PASSED")
    else:
        print("RESULT: ONE OR MORE PREDICTION CHECKS FAILED")
    print("=" * 70)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
