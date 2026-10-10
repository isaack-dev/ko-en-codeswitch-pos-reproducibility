"""
diagnostic_utils.py - Shared helpers for Step 4 diagnostic scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import (
    F_ID, F_FORM, F_LEMMA, F_UPOS, F_MISC,
    get_misc_lang, iter_sentences,
)

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "baselines"
DIAG    = ROOT / "results" / "diagnostic_analysis"

TRAIN_FILE = PROC / "panyo_original_adp_docsplit_train.conllu"
DEV_FILE   = PROC / "panyo_original_adp_docsplit_dev.conllu"
TEST_FILE  = PROC / "panyo_original_adp_docsplit_test.conllu"

ALL_MODELS = [
    "majority_global",
    "majority_lang",
    "perceptron_joint",
    "perceptron_pipeline_panyo",
    "perceptron_pipeline_ud_transfer",
]

PANYO_MODELS = ["perceptron_joint", "perceptron_pipeline_panyo"]


def get_pred_upos(misc: str) -> str:
    for part in misc.split("|"):
        if part.startswith("PredUPOS="):
            return part[9:]
    return ""


def get_doc_id(comments: list[str]) -> str:
    for c in comments:
        if c.startswith("# doc_id = "):
            return c[11:]
    return ""


def get_sent_id(comments: list[str]) -> str:
    for c in comments:
        if c.startswith("# sent_id = "):
            return c[12:]
    return ""


# ---------------------------------------------------------------------------
# Token record
# ---------------------------------------------------------------------------

class TokenRecord:
    """One real token from a prediction file."""
    __slots__ = (
        "sent_id", "doc_id", "tok_idx", "form", "lemma",
        "gold_upos", "pred_upos", "lang",
        "prev_form", "prev_lang", "prev_gold",
        "next_form", "next_lang", "next_gold",
        "sentence_context",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def iter_token_records(pred_path: Path) -> Iterator[TokenRecord]:
    """
    Yield one TokenRecord per real token from a prediction CoNLL-U file.
    Fills prev/next context within the same sentence.
    """
    for sent in iter_sentences(pred_path):
        sid  = get_sent_id(sent["comments"])
        did  = get_doc_id(sent["comments"])
        real = [t for t in sent["tokens"] if t["is_real"]]
        text = " ".join(t["fields"][F_FORM] for t in real)

        for i, tok in enumerate(real):
            f  = tok["fields"]
            misc = f[F_MISC]
            lang = get_misc_lang(misc)
            pred = get_pred_upos(misc)

            p = real[i-1]["fields"] if i > 0 else None
            n = real[i+1]["fields"] if i < len(real)-1 else None

            yield TokenRecord(
                sent_id  = sid,
                doc_id   = did,
                tok_idx  = i,
                form     = f[F_FORM],
                lemma    = f[F_LEMMA],
                gold_upos= f[F_UPOS],
                pred_upos= pred,
                lang     = lang,
                prev_form= p[F_FORM] if p else "",
                prev_lang= get_misc_lang(p[F_MISC]) if p else "",
                prev_gold= p[F_UPOS] if p else "",
                next_form= n[F_FORM] if n else "",
                next_lang= get_misc_lang(n[F_MISC]) if n else "",
                next_gold= n[F_UPOS] if n else "",
                sentence_context=text,
            )


def pred_path(model: str, split: str) -> Path:
    return RESULTS / f"{model}_{split}_predictions.conllu"


def diag_path(name: str) -> Path:
    return DIAG / name


def acc(correct: int, total: int) -> float:
    return round(correct / total, 6) if total else 0.0


def macro_f1(tp_map: dict, fp_map: dict, fn_map: dict, tags: list) -> float:
    f1s = []
    for t in tags:
        tp = tp_map.get(t, 0)
        fp = fp_map.get(t, 0)
        fn = fn_map.get(t, 0)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return round(sum(f1s) / len(f1s), 6) if f1s else 0.0
