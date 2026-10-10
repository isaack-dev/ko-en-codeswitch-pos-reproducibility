"""
train_baselines.py - Train and save all Step 3 baseline POS tagger models.

Models trained:
  A. majority_global       - single most frequent UPOS in Panyo train
  B. majority_lang         - most frequent UPOS per Lang label in Panyo train
  C. perceptron_joint      - averaged perceptron on all Panyo train tokens
  D. perceptron_pipeline_panyo_ko    - perceptron updated only on KO tokens
     perceptron_pipeline_panyo_en    - perceptron updated only on EN tokens
     perceptron_pipeline_panyo_other - perceptron updated only on OTHER tokens
  E. perceptron_pipeline_ud_en_gum   - perceptron trained on GUM English train
     perceptron_pipeline_ud_ko_kaist - perceptron trained on KAIST Korean train

Inputs:
  data/processed/panyo_original_adp_docsplit_train.conllu
  data/en_gum-ud-train.conllu
  data/ko_kaist-ud-train.conllu

Outputs:
  models/baselines/*.json
  results/baselines/training_log.md

Run from repo root:
  python scripts/train_baselines.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_MISC, F_UPOS, get_misc_lang, iter_sentences
from perceptron_tagger import AveragedPerceptron

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
PROC    = DATA / "processed"
MODELS  = ROOT / "models" / "baselines"
RESULTS = ROOT / "results" / "baselines"

PANYO_TRAIN = PROC / "panyo_original_adp_docsplit_train.conllu"
GUM_TRAIN   = DATA / "en_gum-ud-train.conllu"
KAIST_TRAIN = DATA / "ko_kaist-ud-train.conllu"

N_EPOCHS = 5
SEED     = 42


# ---------------------------------------------------------------------------
# CoNLL-U → perceptron format
# ---------------------------------------------------------------------------

def load_tagged_sents(path: Path, lang_override: str | None = None) -> list[list[tuple]]:
    """
    Return a list of sentences.
    Each sentence: [(form, lang, gold_tag), ...] for real tokens only.

    lang_override: if set, every token gets this constant lang value instead
    of reading Lang= from MISC. Used for monolingual UD corpora (GUM, KAIST),
    which carry no Lang= labels, so the lang feature at training time matches
    the exact Lang= value ("EN" / "KO") used for the same-language Panyo
    tokens at inference time (see get_misc_lang() in read_conllu.py and
    classify_lang() in label_language.py for the shared KO/EN/MIXED/OTHER
    vocabulary).
    """
    sents: list[list[tuple]] = []
    for sent in iter_sentences(path):
        toks: list[tuple] = []
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            fields = tok["fields"]
            form   = fields[F_FORM]
            upos   = fields[F_UPOS]
            lang   = lang_override if lang_override is not None else get_misc_lang(fields[F_MISC])
            toks.append((form, lang, upos))
        if toks:
            sents.append(toks)
    return sents


# ---------------------------------------------------------------------------
# Majority models
# ---------------------------------------------------------------------------

def train_majority_global(tagged_sents: list[list[tuple]]) -> dict:
    counts: Counter = Counter()
    for sent in tagged_sents:
        for _, _, tag in sent:
            counts[tag] += 1
    global_tag = counts.most_common(1)[0][0]
    return {
        "type":       "MajorityGlobal",
        "global_tag": global_tag,
        "counts":     dict(counts.most_common()),
    }


def train_majority_lang(tagged_sents: list[list[tuple]]) -> dict:
    per_lang: dict[str, Counter] = {}
    global_cnt: Counter = Counter()
    for sent in tagged_sents:
        for _, lang, tag in sent:
            global_cnt[tag] += 1
            if lang not in per_lang:
                per_lang[lang] = Counter()
            per_lang[lang][tag] += 1

    lang_tags: dict[str, str] = {}
    lang_counts: dict[str, dict] = {}
    for lang, cnt in per_lang.items():
        lang_tags[lang] = cnt.most_common(1)[0][0]
        lang_counts[lang] = dict(cnt.most_common())

    global_tag = global_cnt.most_common(1)[0][0]
    return {
        "type":        "MajorityLang",
        "global_tag":  global_tag,
        "lang_tags":   lang_tags,
        "lang_counts": lang_counts,
    }


# ---------------------------------------------------------------------------
# Training log helpers
# ---------------------------------------------------------------------------

_log_lines: list[str] = []

def _log(line: str = "") -> None:
    print(line)
    _log_lines.append(line)

def _write_log() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "training_log.md"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\nTraining log written to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    _log("# Step 3 Training Log")
    _log(f"\n**Date:** 2026-06-20")
    _log(f"**Epochs:** {N_EPOCHS}  |  **Seed:** {SEED}")

    # -----------------------------------------------------------------------
    # Load Panyo training data
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## Panyo Training Data")
    print(f"\nLoading {PANYO_TRAIN.name} ...")
    t0 = time.time()
    panyo_sents = load_tagged_sents(PANYO_TRAIN)
    n_panyo_tok = sum(len(s) for s in panyo_sents)
    _log(f"\n- File: `{PANYO_TRAIN.name}`")
    _log(f"- Sentences: {len(panyo_sents):,}")
    _log(f"- Real tokens: {n_panyo_tok:,}")
    _log(f"- Load time: {time.time()-t0:.1f}s")

    # Collect tag vocab and per-lang counts
    all_tags: set[str] = set()
    lang_cnt: Counter = Counter()
    for sent in panyo_sents:
        for _, lang, tag in sent:
            all_tags.add(tag)
            lang_cnt[lang] += 1
    _log(f"- UPOS tags: {sorted(all_tags)}")
    _log(f"- Lang distribution: { {l: lang_cnt[l] for l in sorted(lang_cnt)} }")

    # -----------------------------------------------------------------------
    # A. Global majority
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## A. Global Majority Baseline")
    t0 = time.time()
    maj_global = train_majority_global(panyo_sents)
    path = MODELS / "majority_global.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(maj_global, f, ensure_ascii=False, indent=2)
    _log(f"\n- Global majority tag: **{maj_global['global_tag']}**")
    _log(f"- Top tags: { {k: v for k, v in list(maj_global['counts'].items())[:5]} }")
    _log(f"- Saved: `{path.name}` ({path.stat().st_size // 1024} KB)")
    _log(f"- Time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # B. Language-aware majority
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## B. Language-Aware Majority Baseline")
    t0 = time.time()
    maj_lang = train_majority_lang(panyo_sents)
    path = MODELS / "majority_lang.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(maj_lang, f, ensure_ascii=False, indent=2)
    _log(f"\n- Majority tag per lang: {maj_lang['lang_tags']}")
    _log(f"- Fallback (global) tag: {maj_lang['global_tag']}")
    for lang, cnt in maj_lang["lang_counts"].items():
        top3 = list(cnt.items())[:3]
        _log(f"  - {lang}: top-3 = {top3}")
    _log(f"- Saved: `{path.name}` ({path.stat().st_size // 1024} KB)")
    _log(f"- Time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # C. Joint perceptron on all Panyo train
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## C. Joint Perceptron (all Panyo tokens)")
    print("\nTraining joint perceptron ...")
    t0 = time.time()
    joint = AveragedPerceptron()
    joint.train(panyo_sents, n_epochs=N_EPOCHS, seed=SEED, lang_filter=None)
    elapsed = time.time() - t0
    mb = joint.save(MODELS / "perceptron_joint.json")
    _log(f"\n- Training sentences: {len(panyo_sents):,}")
    _log(f"- Training tokens: {n_panyo_tok:,}")
    _log(f"- Epochs: {N_EPOCHS}")
    _log(f"- Training time: {elapsed:.1f}s")
    _log(f"- Model size: {mb} MB")
    _log(f"- Feature vocab: {len(joint.weights):,}")
    _log(f"- Tag set: {joint.tags}")

    # -----------------------------------------------------------------------
    # D. Panyo-internal pipeline perceptrons
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## D. Panyo-Internal Pipeline Perceptrons")
    _log("\nTraining uses the full sentence context but updates only for the")
    _log("specified language. MIXED=0 in this corpus; no MIXED tagger is trained.")

    for lang_lbl, model_name in [
        ("KO",    "perceptron_pipeline_panyo_ko"),
        ("EN",    "perceptron_pipeline_panyo_en"),
        ("OTHER", "perceptron_pipeline_panyo_other"),
    ]:
        n_toks = sum(1 for s in panyo_sents for _, l, _ in s if l == lang_lbl)
        _log(f"\n### D.{lang_lbl} tagger ({n_toks:,} update tokens)")
        print(f"\nTraining pipeline [{lang_lbl}] perceptron ...")
        t0 = time.time()
        m  = AveragedPerceptron()
        m.train(panyo_sents, n_epochs=N_EPOCHS, seed=SEED, lang_filter=lang_lbl)
        elapsed = time.time() - t0
        mb = m.save(MODELS / f"{model_name}.json")
        _log(f"- Training time: {elapsed:.1f}s")
        _log(f"- Model size: {mb} MB")
        _log(f"- Feature vocab: {len(m.weights):,}")
        _log(f"- Tag set: {m.tags}")

    # -----------------------------------------------------------------------
    # E. UD-transfer pipeline (GUM English + KAIST Korean)
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## E. UD-Transfer Pipeline Perceptrons")
    _log("\nEach corpus is monolingual, so every token is assigned the same")
    _log("constant Lang= value used for that language's Panyo tokens at")
    _log("inference (GUM -> EN, KAIST -> KO), keeping the lang feature")
    _log("representation identical between training and inference.")

    UD_LANG_OVERRIDE = {GUM_TRAIN: "EN", KAIST_TRAIN: "KO"}

    for train_path, model_name, corpus_label in [
        (GUM_TRAIN,   "perceptron_pipeline_ud_en_gum",   "GUM English"),
        (KAIST_TRAIN, "perceptron_pipeline_ud_ko_kaist",  "KAIST Korean"),
    ]:
        if not train_path.exists():
            _log(f"\n### {corpus_label}: SKIPPED ({train_path.name} not found)")
            continue

        print(f"\nLoading {train_path.name} ...")
        t0 = time.time()
        ud_sents = load_tagged_sents(train_path, lang_override=UD_LANG_OVERRIDE[train_path])
        n_ud_tok = sum(len(s) for s in ud_sents)
        ud_tags  = sorted({tag for s in ud_sents for _, _, tag in s})
        _log(f"\n### {corpus_label}")
        _log(f"- File: `{train_path.name}`")
        _log(f"- Sentences: {len(ud_sents):,}")
        _log(f"- Tokens: {n_ud_tok:,}")
        _log(f"- UPOS tags: {ud_tags}")

        print(f"Training {corpus_label} UD-transfer perceptron ...")
        m = AveragedPerceptron()
        m.train(ud_sents, n_epochs=N_EPOCHS, seed=SEED, lang_filter=None)
        elapsed = time.time() - t0
        mb = m.save(MODELS / f"{model_name}.json")
        _log(f"- Training time: {elapsed:.1f}s")
        _log(f"- Model size: {mb} MB")
        _log(f"- Feature vocab: {len(m.weights):,}")

    # -----------------------------------------------------------------------
    # Summary of models
    # -----------------------------------------------------------------------
    _log("\n---")
    _log("\n## Model File Summary")
    _log("\n| File | Size |")
    _log("|---|---|")
    for p in sorted(MODELS.glob("*.json")):
        kb = p.stat().st_size // 1024
        size_str = f"{kb//1024} MB" if kb >= 1024 else f"{kb} KB"
        _log(f"| `{p.name}` | {size_str} |")

    _write_log()
    print("\nAll models saved to models/baselines/")


if __name__ == "__main__":
    main()
