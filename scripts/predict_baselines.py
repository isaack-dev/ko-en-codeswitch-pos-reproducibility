"""
predict_baselines.py - Generate prediction files for all Step 3 baseline models.

For each model and each split (dev, test), reads the docsplit CoNLL-U file,
appends PredUPOS=<tag> to the MISC column of real tokens, and writes a new
prediction file.

Gold UPOS in column 4 is NEVER modified.
Multiword tokens and empty nodes are preserved without modification.

Run from repo root:
  python scripts/predict_baselines.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import (
    F_FORM, F_MISC, F_UPOS,
    get_misc_lang, iter_sentences, write_sentences,
)
from perceptron_tagger import AveragedPerceptron

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "models" / "baselines"
RESULTS = ROOT / "results" / "baselines"

DEV_IN  = PROC / "panyo_original_adp_docsplit_dev.conllu"
TEST_IN = PROC / "panyo_original_adp_docsplit_test.conllu"

SPLITS: dict[str, Path] = {"dev": DEV_IN, "test": TEST_IN}


# ---------------------------------------------------------------------------
# MISC helpers
# ---------------------------------------------------------------------------

def set_pred_upos(misc: str, tag: str) -> str:
    """Append or replace PredUPOS=<tag> in a MISC field string."""
    if misc == "_":
        return f"PredUPOS={tag}"
    parts = [p for p in misc.split("|") if not p.startswith("PredUPOS=")]
    parts.append(f"PredUPOS={tag}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Sentence-level helpers
# ---------------------------------------------------------------------------

def sent_to_real_tokens(sent: dict) -> list[tuple]:
    """Extract (form, lang) tuples for real tokens; returns (index, form, lang)."""
    result: list[tuple] = []
    for i, tok in enumerate(sent["tokens"]):
        if tok["is_real"]:
            fields = tok["fields"]
            result.append((i, fields[F_FORM], get_misc_lang(fields[F_MISC])))
    return result


def annotate_sent(sent: dict, tag_by_tok_idx: dict[int, str]) -> dict:
    """
    Return a deep copy of sent with PredUPOS added to MISC for real tokens.
    tag_by_tok_idx: maps token list index → predicted UPOS tag.
    """
    new_tokens = []
    for i, tok in enumerate(sent["tokens"]):
        new_tok = {"fields": list(tok["fields"]), "is_real": tok["is_real"]}
        if tok["is_real"] and i in tag_by_tok_idx:
            new_tok["fields"][F_MISC] = set_pred_upos(
                tok["fields"][F_MISC], tag_by_tok_idx[i]
            )
        new_tokens.append(new_tok)
    return {"comments": sent["comments"], "tokens": new_tokens}


# ---------------------------------------------------------------------------
# Majority predictors
# ---------------------------------------------------------------------------

class MajorityGlobalPredictor:
    def __init__(self, model: dict) -> None:
        self.tag = model["global_tag"]

    def predict_sents(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            tag_map = {i: self.tag for i, _, _ in sent_to_real_tokens(sent)}
            out.append(annotate_sent(sent, tag_map))
        return out


class MajorityLangPredictor:
    def __init__(self, model: dict) -> None:
        self.lang_tags  = model["lang_tags"]
        self.global_tag = model["global_tag"]

    def predict_sents(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            tag_map = {}
            for i, _, lang in sent_to_real_tokens(sent):
                tag_map[i] = self.lang_tags.get(lang, self.global_tag)
            out.append(annotate_sent(sent, tag_map))
        return out


# ---------------------------------------------------------------------------
# Perceptron predictors
# ---------------------------------------------------------------------------

class PerceptronPredictor:
    """Wraps a single AveragedPerceptron for joint prediction."""

    def __init__(self, tagger: AveragedPerceptron) -> None:
        self.tagger = tagger

    def predict_sents(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            real_toks = sent_to_real_tokens(sent)  # [(i, form, lang)]
            untagged  = [(form, lang) for _, form, lang in real_toks]
            preds     = self.tagger.predict(untagged) if untagged else []
            tag_map   = {real_toks[k][0]: preds[k] for k in range(len(real_toks))}
            out.append(annotate_sent(sent, tag_map))
        return out


class PipelinePanyoPredictor:
    """
    Language-routing pipeline using Panyo-trained per-language taggers.

    KO   tokens → ko_tagger
    EN   tokens → en_tagger
    OTHER tokens → other_tagger
    MIXED tokens → global_fallback (MIXED=0 in this corpus, but handled)

    Each tagger predicts over the full sentence context; only the predictions
    for the routed language tokens are used.
    """

    def __init__(
        self,
        ko_tagger:    AveragedPerceptron,
        en_tagger:    AveragedPerceptron,
        other_tagger: AveragedPerceptron,
        fallback_tag: str,
    ) -> None:
        self.ko    = ko_tagger
        self.en    = en_tagger
        self.other = other_tagger
        self.fallback = fallback_tag

    def predict_sents(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            real_toks = sent_to_real_tokens(sent)  # [(i, form, lang)]
            if not real_toks:
                out.append(annotate_sent(sent, {}))
                continue

            untagged = [(form, lang) for _, form, lang in real_toks]

            # Each tagger predicts for the full sentence; we pick by language
            ko_preds    = self.ko.predict(untagged)
            en_preds    = self.en.predict(untagged)
            other_preds = self.other.predict(untagged)

            tag_map: dict[int, str] = {}
            for k, (tok_idx, _, lang) in enumerate(real_toks):
                if lang == "KO":
                    tag_map[tok_idx] = ko_preds[k]
                elif lang == "EN":
                    tag_map[tok_idx] = en_preds[k]
                elif lang == "OTHER":
                    tag_map[tok_idx] = other_preds[k]
                else:
                    tag_map[tok_idx] = self.fallback  # MIXED or unknown

            out.append(annotate_sent(sent, tag_map))
        return out


class PipelineUDTransferPredictor:
    """
    UD-transfer pipeline.

    EN tokens → en_gum_tagger  (trained on GUM, no Lang= features)
    KO tokens → ko_kaist_tagger (trained on KAIST, no Lang= features)
    OTHER/MIXED → other_fallback_tag (Panyo OTHER majority)

    Expected to show annotation mismatch effects (documented feature).
    """

    def __init__(
        self,
        en_tagger:  AveragedPerceptron,
        ko_tagger:  AveragedPerceptron,
        other_tag:  str,
    ) -> None:
        self.en    = en_tagger
        self.ko    = ko_tagger
        self.other = other_tag

    def predict_sents(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            real_toks = sent_to_real_tokens(sent)
            if not real_toks:
                out.append(annotate_sent(sent, {}))
                continue

            untagged = [(form, lang) for _, form, lang in real_toks]

            en_preds = self.en.predict(untagged)
            ko_preds = self.ko.predict(untagged)

            tag_map: dict[int, str] = {}
            for k, (tok_idx, _, lang) in enumerate(real_toks):
                if lang == "EN":
                    tag_map[tok_idx] = en_preds[k]
                elif lang == "KO":
                    tag_map[tok_idx] = ko_preds[k]
                else:
                    tag_map[tok_idx] = self.other

            out.append(annotate_sent(sent, tag_map))
        return out


# ---------------------------------------------------------------------------
# Helpers: load model JSON
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_perceptron(path: Path) -> AveragedPerceptron:
    print(f"  Loading {path.name} ...")
    return AveragedPerceptron.load(path)


# ---------------------------------------------------------------------------
# Write prediction file
# ---------------------------------------------------------------------------

def write_predictions(predictor, split_path: Path, out_path: Path) -> int:
    """Predict and write. Returns number of sentences written."""
    print(f"  Predicting on {split_path.name} ...")
    sentences = list(iter_sentences(split_path))
    annotated = predictor.predict_sents(sentences)
    write_sentences(annotated, out_path)
    print(f"    -> {out_path.name} ({len(annotated):,} sents)")
    return len(annotated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Verify input model directory
    if not MODELS.exists():
        sys.exit(f"ERROR: {MODELS} not found. Run train_baselines.py first.")

    summary_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # A. Global majority
    # -----------------------------------------------------------------------
    print("\n=== A. Majority Global ===")
    m_path = MODELS / "majority_global.json"
    if m_path.exists():
        predictor = MajorityGlobalPredictor(load_json(m_path))
        for split, src in SPLITS.items():
            out = RESULTS / f"majority_global_{split}_predictions.conllu"
            n   = write_predictions(predictor, src, out)
            summary_rows.append({"model": "majority_global", "split": split,
                                  "file": out.name, "sentences": n, "valid": "?"})
    else:
        print(f"  SKIP: {m_path.name} not found")

    # -----------------------------------------------------------------------
    # B. Language-aware majority
    # -----------------------------------------------------------------------
    print("\n=== B. Majority Lang ===")
    m_path = MODELS / "majority_lang.json"
    if m_path.exists():
        predictor = MajorityLangPredictor(load_json(m_path))
        for split, src in SPLITS.items():
            out = RESULTS / f"majority_lang_{split}_predictions.conllu"
            n   = write_predictions(predictor, src, out)
            summary_rows.append({"model": "majority_lang", "split": split,
                                  "file": out.name, "sentences": n, "valid": "?"})
    else:
        print(f"  SKIP: {m_path.name} not found")

    # -----------------------------------------------------------------------
    # C. Joint perceptron
    # -----------------------------------------------------------------------
    print("\n=== C. Perceptron Joint ===")
    m_path = MODELS / "perceptron_joint.json"
    if m_path.exists():
        tagger    = load_perceptron(m_path)
        predictor = PerceptronPredictor(tagger)
        for split, src in SPLITS.items():
            out = RESULTS / f"perceptron_joint_{split}_predictions.conllu"
            n   = write_predictions(predictor, src, out)
            summary_rows.append({"model": "perceptron_joint", "split": split,
                                  "file": out.name, "sentences": n, "valid": "?"})
    else:
        print(f"  SKIP: {m_path.name} not found")

    # -----------------------------------------------------------------------
    # D. Panyo-internal pipeline
    # -----------------------------------------------------------------------
    print("\n=== D. Pipeline Panyo ===")
    paths_ok = all(
        (MODELS / f"perceptron_pipeline_panyo_{l}.json").exists()
        for l in ("ko", "en", "other")
    )
    if paths_ok:
        ko_t    = load_perceptron(MODELS / "perceptron_pipeline_panyo_ko.json")
        en_t    = load_perceptron(MODELS / "perceptron_pipeline_panyo_en.json")
        other_t = load_perceptron(MODELS / "perceptron_pipeline_panyo_other.json")

        # Fallback for MIXED (which is 0 in this corpus, but handled gracefully)
        # Use global majority tag from the global model if available
        gm_path = MODELS / "majority_global.json"
        fallback = load_json(gm_path)["global_tag"] if gm_path.exists() else "NOUN"

        predictor = PipelinePanyoPredictor(ko_t, en_t, other_t, fallback)
        for split, src in SPLITS.items():
            out = RESULTS / f"perceptron_pipeline_panyo_{split}_predictions.conllu"
            n   = write_predictions(predictor, src, out)
            summary_rows.append({"model": "perceptron_pipeline_panyo", "split": split,
                                  "file": out.name, "sentences": n, "valid": "?"})
    else:
        print(f"  SKIP: one or more pipeline_panyo model files missing")

    # -----------------------------------------------------------------------
    # E. UD-transfer pipeline
    # -----------------------------------------------------------------------
    print("\n=== E. Pipeline UD Transfer ===")
    en_path = MODELS / "perceptron_pipeline_ud_en_gum.json"
    ko_path = MODELS / "perceptron_pipeline_ud_ko_kaist.json"

    if en_path.exists() and ko_path.exists():
        en_t = load_perceptron(en_path)
        ko_t = load_perceptron(ko_path)

        # OTHER fallback: majority OTHER tag from Panyo lang model
        ml_path    = MODELS / "majority_lang.json"
        other_fallback = "PUNCT"  # most common OTHER tag in Panyo
        if ml_path.exists():
            ml = load_json(ml_path)
            other_fallback = ml["lang_tags"].get("OTHER", "PUNCT")

        predictor = PipelineUDTransferPredictor(en_t, ko_t, other_fallback)
        for split, src in SPLITS.items():
            out = RESULTS / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
            n   = write_predictions(predictor, src, out)
            summary_rows.append({"model": "perceptron_pipeline_ud_transfer", "split": split,
                                  "file": out.name, "sentences": n, "valid": "?"})
    else:
        print(f"  SKIP: UD-transfer model files not found")
        for split in SPLITS:
            summary_rows.append({"model": "perceptron_pipeline_ud_transfer", "split": split,
                                  "file": "N/A", "sentences": 0, "valid": "SKIPPED"})

    # -----------------------------------------------------------------------
    # Write model_prediction_summary.csv
    # -----------------------------------------------------------------------
    import csv
    summary_path = RESULTS / "model_prediction_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "file", "sentences", "valid"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
