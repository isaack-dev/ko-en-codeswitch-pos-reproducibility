"""
label_language.py - Add token-level language labels to the Panyo corpus.

Input:  data/kor_en_panyo_corpus.conllu
Output: data/processed/panyo_original_adp_lang_labeled.conllu

Language labels added to the MISC column (field 9) of each real token:
    Lang=KO     - token contains Hangul, no Latin
    Lang=EN     - token contains Latin, no Hangul
    Lang=MIXED  - token contains both Hangul and Latin
    Lang=OTHER  - punctuation, digits, symbols; neither Hangul nor Latin

Rules:
  - Preserves all original UPOS labels exactly (including ADP for Korean particles).
  - Preserves all existing MISC content; appends Lang= if MISC already has values.
  - Skips multiword tokens (ID contains '-') and empty nodes (ID contains '.').
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_MISC, iter_sentences, set_misc_lang, write_sentences

ROOT   = Path(__file__).parent.parent
INPUT  = ROOT / "data" / "kor_en_panyo_corpus.conllu"
OUTPUT = ROOT / "data" / "processed" / "panyo_original_adp_lang_labeled.conllu"


# ---------------------------------------------------------------------------
# Character-level language detection
# ---------------------------------------------------------------------------

def _has_hangul(s: str) -> bool:
    """Return True if s contains any Hangul character."""
    for c in s:
        cp = ord(c)
        if (0xAC00 <= cp <= 0xD7A3 or   # Syllables (precomposed)
                0x1100 <= cp <= 0x11FF or   # Jamo
                0x3130 <= cp <= 0x318F or   # Compatibility Jamo
                0xA960 <= cp <= 0xA97F or   # Jamo Extended-A
                0xD7B0 <= cp <= 0xD7FF):    # Jamo Extended-B
            return True
    return False


def _has_latin(s: str) -> bool:
    """Return True if s contains any ASCII Latin letter [A-Za-z]."""
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in s)


def classify_lang(form: str) -> str:
    """
    Classify a token form into one of KO / EN / MIXED / OTHER.

    KO    : has Hangul, no Latin
    EN    : has Latin, no Hangul
    MIXED : has both Hangul and Latin (e.g. 'Python으로', 'debug하는')
    OTHER : punctuation, pure numbers, symbols - neither Hangul nor Latin
    """
    h = _has_hangul(form)
    la = _has_latin(form)
    if h and la:
        return "MIXED"
    if h:
        return "KO"
    if la:
        return "EN"
    return "OTHER"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT.exists():
        sys.exit(f"ERROR: input file not found: {INPUT}")

    counts: dict[str, int] = {"KO": 0, "EN": 0, "MIXED": 0, "OTHER": 0}
    n_sents  = 0
    n_tokens = 0
    labeled: list[dict] = []

    for sent in iter_sentences(INPUT):
        n_sents += 1
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            n_tokens += 1
            lang = classify_lang(tok["fields"][F_FORM])
            counts[lang] += 1
            tok["fields"][F_MISC] = set_misc_lang(tok["fields"][F_MISC], lang)
        labeled.append(sent)

    write_sentences(labeled, OUTPUT)

    # Summary
    print(f"Sentences  : {n_sents:,}")
    print(f"Real tokens: {n_tokens:,}")
    for tag in ("KO", "EN", "MIXED", "OTHER"):
        pct = 100.0 * counts[tag] / n_tokens if n_tokens else 0.0
        print(f"  {tag:<6s}: {counts[tag]:>8,}  ({pct:.1f}%)")
    print(f"\nWrote: {OUTPUT}")


if __name__ == "__main__":
    main()
