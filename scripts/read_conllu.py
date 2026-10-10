"""
read_conllu.py - Pure-Python CoNLL-U reader and writer.

A sentence is a dict with two keys:
    "comments" : list[str]  - comment lines preserved verbatim (no trailing newline)
    "tokens"   : list[dict] - every non-blank, non-comment line in the sentence block

Each token dict:
    "fields"  : list[str]  - the 10 tab-separated CoNLL-U fields
    "is_real" : bool       - True iff the token ID is a plain integer
                             (False for multiword ranges like "1-2" and empty nodes like "1.1")

Field index constants for convenience:
    F_ID, F_FORM, F_LEMMA, F_UPOS, F_XPOS,
    F_FEATS, F_HEAD, F_DEPREL, F_DEPS, F_MISC
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

# Field index constants
F_ID     = 0
F_FORM   = 1
F_LEMMA  = 2
F_UPOS   = 3
F_XPOS   = 4
F_FEATS  = 5
F_HEAD   = 6
F_DEPREL = 7
F_DEPS   = 8
F_MISC   = 9


def _is_real_token(token_id: str) -> bool:
    """Return True for plain integer IDs; False for multiword ranges and empty nodes."""
    return token_id.isdigit()


def iter_sentences(path: Path) -> Generator[dict, None, None]:
    """
    Yield sentences from a CoNLL-U file one at a time (memory-efficient).

    Handles files that do not end with a trailing blank line.
    Raises ValueError if any token line does not have exactly 10 fields.
    """
    comments: list[str] = []
    tokens: list[dict] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith("#"):
                comments.append(line)
            elif line == "":
                if comments or tokens:
                    yield {"comments": comments, "tokens": tokens}
                    comments = []
                    tokens = []
            else:
                parts = line.split("\t")
                if len(parts) != 10:
                    raise ValueError(
                        f"{path}:{lineno}: expected 10 tab-separated fields, "
                        f"got {len(parts)}: {line!r}"
                    )
                tokens.append({
                    "fields":  parts,
                    "is_real": _is_real_token(parts[F_ID]),
                })

    # Yield last sentence if file does not end with a blank line
    if comments or tokens:
        yield {"comments": comments, "tokens": tokens}


def read_sentences(path: Path) -> list[dict]:
    """Load all sentences from a CoNLL-U file into a list."""
    return list(iter_sentences(path))


def sentence_to_str(sentence: dict) -> str:
    """Serialize a sentence dict back to a CoNLL-U block ending with a blank line."""
    lines = sentence["comments"] + [
        "\t".join(tok["fields"]) for tok in sentence["tokens"]
    ]
    return "\n".join(lines) + "\n\n"


def write_sentences(sentences: list[dict], path: Path) -> None:
    """Write a list of sentence dicts to a CoNLL-U file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sentence in sentences:
            fh.write(sentence_to_str(sentence))


def get_misc_lang(misc: str) -> str:
    """Extract the Lang=XX value from a MISC field string, or '' if absent."""
    for part in misc.split("|"):
        if part.startswith("Lang="):
            return part[5:]
    return ""


def set_misc_lang(misc: str, lang: str) -> str:
    """
    Add or replace Lang=XX in a MISC field string.

    - If misc is "_", returns "Lang=XX".
    - Otherwise, removes any existing Lang=XX entry and appends the new one.
    """
    label = f"Lang={lang}"
    if misc == "_":
        return label
    parts = [p for p in misc.split("|") if not p.startswith("Lang=")]
    parts.append(label)
    return "|".join(parts)
