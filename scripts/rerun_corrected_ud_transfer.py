"""
rerun_corrected_ud_transfer.py - Rerun only the UD-transfer-affected experiments
after fixing the language-feature mismatch in scripts/train_baselines.py.

Does NOT touch models/baselines/*.json, results/baselines/*, or
results/diagnostic_analysis/* (all original outputs are preserved). Does NOT
retrain the in-domain models (Joint Perceptron, Panyo-internal Pipeline,
majority baselines) -- they are unaffected by the fix (their data-loading
call site, load_tagged_sents(PANYO_TRAIN), is unchanged) and are reused
read-only from models/baselines/ where needed (e.g. the OTHER-token fallback
tag).

Everything this script produces is written under outputs/corrected_ud_transfer/.

Run from repo root:
  python scripts/rerun_corrected_ud_transfer.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import train_baselines as tb  # noqa: E402  (already fixed: load_tagged_sents)
import predict_baselines as pb  # noqa: E402
import diagnostic_ud_mapping as dm  # noqa: E402
import evaluate_baselines as eb  # noqa: E402
from diagnostic_seen_unseen import build_train_vocab  # noqa: E402
from diagnostic_boundary_analysis import (  # noqa: E402
    _is_ko_en_boundary, boundary_direction, region_labels,
)
from diagnostic_utils import TRAIN_FILE, iter_token_records, acc as diag_acc  # noqa: E402
from perceptron_tagger import AveragedPerceptron  # noqa: E402
from read_conllu import F_FORM, F_MISC, F_UPOS, get_misc_lang, iter_sentences  # noqa: E402

ROOT       = Path(__file__).parent.parent
MODELS_ORIG = ROOT / "models" / "baselines"          # read-only (unchanged)
RESULTS_ORIG = ROOT / "results" / "baselines"          # read-only (unchanged)
DIAG_ORIG   = ROOT / "results" / "diagnostic_analysis"  # read-only (unchanged)

OUT         = ROOT / "outputs" / "corrected_ud_transfer"
MODELS_OUT  = OUT / "models"
PRED_OUT    = OUT / "predictions"
METRICS_OUT = OUT / "metrics"
SEEN_OUT    = OUT / "seen_unseen"
BOUND_OUT   = OUT / "boundary"
COMPARE_OUT = OUT / "comparison"

SPLITS = ["dev", "test"]
UD_LANG_OVERRIDE = {tb.GUM_TRAIN: "EN", tb.KAIST_TRAIN: "KO"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for d in (MODELS_OUT, PRED_OUT, METRICS_OUT, SEEN_OUT, BOUND_OUT, COMPARE_OUT):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Retrain ONLY the two UD-transfer perceptrons (corrected lang feature)
# ---------------------------------------------------------------------------

def retrain_ud_transfer_models() -> dict:
    info: dict = {}
    for train_path, model_name, corpus_label in [
        (tb.GUM_TRAIN,   "perceptron_pipeline_ud_en_gum",  "GUM English"),
        (tb.KAIST_TRAIN, "perceptron_pipeline_ud_ko_kaist", "KAIST Korean"),
    ]:
        lang = UD_LANG_OVERRIDE[train_path]
        print(f"\n[retrain] Loading {train_path.name} (lang_override={lang}) ...")
        t0 = time.time()
        ud_sents = tb.load_tagged_sents(train_path, lang_override=lang)
        n_tok = sum(len(s) for s in ud_sents)
        n_blank = sum(1 for s in ud_sents for _, l, _ in s if l == "")
        assert n_blank == 0, f"{train_path.name}: {n_blank} tokens still have blank lang"

        print(f"[retrain] Training {corpus_label} UD-transfer perceptron "
              f"(epochs={tb.N_EPOCHS}, seed={tb.SEED}) ...")
        m = AveragedPerceptron()
        m.train(ud_sents, n_epochs=tb.N_EPOCHS, seed=tb.SEED, lang_filter=None)
        elapsed = time.time() - t0
        out_path = MODELS_OUT / f"{model_name}.json"
        mb = m.save(out_path)

        info[model_name] = {
            "corpus": corpus_label,
            "source_file": str(train_path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256_of(train_path),
            "lang_override": lang,
            "sentences": len(ud_sents),
            "tokens": n_tok,
            "n_epochs": tb.N_EPOCHS,
            "seed": tb.SEED,
            "training_seconds": round(elapsed, 1),
            "model_size_mb": mb,
            "feature_vocab": len(m.weights),
            "tags": m.tags,
            "output_model_path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        }
    return info


# ---------------------------------------------------------------------------
# Step 2: Generate corrected UD-transfer pipeline predictions
# ---------------------------------------------------------------------------

def generate_corrected_predictions() -> dict:
    en_t = AveragedPerceptron.load(MODELS_OUT / "perceptron_pipeline_ud_en_gum.json")
    ko_t = AveragedPerceptron.load(MODELS_OUT / "perceptron_pipeline_ud_ko_kaist.json")

    ml_path = MODELS_ORIG / "majority_lang.json"  # unchanged, read-only
    other_fallback = "PUNCT"
    if ml_path.exists():
        ml = pb.load_json(ml_path)
        other_fallback = ml["lang_tags"].get("OTHER", "PUNCT")

    predictor = pb.PipelineUDTransferPredictor(en_t, ko_t, other_fallback)
    src_by_split = {"dev": pb.DEV_IN, "test": pb.TEST_IN}
    out_paths: dict[str, Path] = {}
    for split, src in src_by_split.items():
        out = PRED_OUT / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        pb.write_predictions(predictor, src, out)
        out_paths[split] = out
    return {
        "other_fallback_tag": other_fallback,
        "other_fallback_source": "models/baselines/majority_lang.json (unchanged)",
        "predictions": {s: str(p.relative_to(ROOT)).replace("\\", "/") for s, p in out_paths.items()},
    }


def verify_token_identity_preserved() -> dict:
    """Prove doc_id/sent_id/token identity/reference UPOS are unchanged
    between the ORIGINAL (buggy) and CORRECTED UD-transfer prediction files
    -- only PredUPOS may differ."""
    report: dict = {}
    for split in SPLITS:
        orig = RESULTS_ORIG / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        corr = PRED_OUT / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        if not orig.exists():
            report[split] = {"status": "SKIPPED", "reason": f"{orig} not found"}
            continue

        def keyed(path: Path) -> dict[tuple, str]:
            out = {}
            for sent in iter_sentences(path):
                sid = did = None
                for c in sent["comments"]:
                    if c.startswith("# doc_id = "):
                        did = c[len("# doc_id = "):]
                    if c.startswith("# sent_id = "):
                        sid = c[len("# sent_id = "):]
                real = [t for t in sent["tokens"] if t["is_real"]]
                for i, tok in enumerate(real):
                    out[(did, sid, i)] = tok["fields"][F_UPOS]
            return out

        orig_keys = keyed(orig)
        corr_keys = keyed(corr)
        same_keys = set(orig_keys) == set(corr_keys)
        gold_matches = same_keys and all(orig_keys[k] == corr_keys[k] for k in orig_keys)
        report[split] = {
            "status": "OK" if (same_keys and gold_matches) else "MISMATCH",
            "orig_tokens": len(orig_keys),
            "corrected_tokens": len(corr_keys),
            "identical_token_keys": same_keys,
            "identical_gold_upos": gold_matches,
        }
        assert same_keys, f"{split}: token key sets differ between original and corrected predictions"
        assert gold_matches, f"{split}: gold UPOS differs between original and corrected predictions"
    return report


# ---------------------------------------------------------------------------
# Step 3: Generate mapping-variant predictions (conservative / panyo_coarse / ko_oriented)
# ---------------------------------------------------------------------------

def generate_mapping_variants() -> dict:
    dev_path = PRED_OUT / "perceptron_pipeline_ud_transfer_dev_predictions.conllu"
    r_without = dm.evaluate_mapped(dev_path, scheme=3, en_adj_map=False)
    r_with    = dm.evaluate_mapped(dev_path, scheme=3, en_adj_map=True)
    en_adj_map = r_with["accuracy"] > r_without["accuracy"]
    print(f"[mapping] EN ADJ->NOUN in scheme 3 (ko_oriented): {'YES' if en_adj_map else 'NO'} "
          f"(dev acc without={r_without['accuracy']:.4f}, with={r_with['accuracy']:.4f})")

    out_paths: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        src = PRED_OUT / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        out_paths[split] = {}
        for scheme in (1, 2, 3):
            dst = PRED_OUT / f"ud_transfer_{dm.SCHEME_NAMES[scheme]}_{split}_predictions.conllu"
            dm.write_mapped_conllu(src, dst, scheme, en_adj_map if scheme == 3 else False)
            out_paths[split][dm.SCHEME_NAMES[scheme]] = str(dst.relative_to(ROOT)).replace("\\", "/")
            print(f"  -> {dst.name}")
    return {"en_adj_map_scheme3": en_adj_map, "mapped_predictions": out_paths}


# ---------------------------------------------------------------------------
# Step 4: Main metrics (mirrors evaluate_baselines.py, scoped to the corrected
# raw UD-transfer model only, matching the original script's scope)
# ---------------------------------------------------------------------------

def compute_main_metrics() -> None:
    overall_rows: list[dict] = []
    pos_rows: list[dict] = []
    lang_rows: list[dict] = []
    model = "perceptron_pipeline_ud_transfer"

    for split in SPLITS:
        pred_path = PRED_OUT / f"{model}_{split}_predictions.conllu"
        records = eb.read_predictions(pred_path)
        corr, tot = eb.compute_accuracy(records)
        corr_np, tot_np = eb.compute_accuracy(records, exclude_gold="PUNCT")
        acc_v = round(corr / tot, 4) if tot else 0.0
        acc_np = round(corr_np / tot_np, 4) if tot_np else 0.0
        pc = eb.compute_per_class(records)
        mf1 = eb.compute_macro_f1(pc)
        wf1 = eb.compute_weighted_f1(pc)

        overall_rows.append({
            "model": model, "split": split, "tokens": tot, "correct": corr,
            "accuracy": acc_v, "non_punct_tokens": tot_np, "non_punct_correct": corr_np,
            "non_punct_accuracy": acc_np, "macro_f1": mf1, "weighted_f1": wf1,
        })
        for tag, m in pc.items():
            pos_rows.append({"model": model, "split": split, "tag": tag, **{
                k: m[k] for k in ("precision", "recall", "f1", "support", "tp", "fp", "fn")
            }})
        for lang, (lc, lt) in eb.compute_lang_accuracy(records).items():
            lang_rows.append({
                "model": model, "split": split, "lang": lang,
                "correct": lc, "total": lt, "accuracy": round(lc / lt, 4) if lt else 0.0,
            })

        all_tags, conf = eb.compute_confusion(records)
        conf_path = METRICS_OUT / f"confusion_matrix_{model}_{split}.csv"
        with conf_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gold\\pred"] + all_tags)
            for g in all_tags:
                w.writerow([g] + [conf.get((g, p), 0) for p in all_tags])
        print(f"  -> {conf_path.name}")

    eb._write_csv(METRICS_OUT / "overall_results.csv",
                  ["model", "split", "tokens", "correct", "accuracy", "non_punct_tokens",
                   "non_punct_correct", "non_punct_accuracy", "macro_f1", "weighted_f1"],
                  overall_rows)
    eb._write_csv(METRICS_OUT / "pos_wise_f1.csv",
                  ["model", "split", "tag", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
                  pos_rows)
    eb._write_csv(METRICS_OUT / "language_wise_accuracy.csv",
                  ["model", "split", "lang", "correct", "total", "accuracy"],
                  lang_rows)
    print(f"  -> overall_results.csv, pos_wise_f1.csv, language_wise_accuracy.csv")


# ---------------------------------------------------------------------------
# Step 5: Mapping-scheme metrics (mirrors diagnostic_ud_mapping.py exactly)
# ---------------------------------------------------------------------------

def compute_mapping_metrics(en_adj_map: bool) -> None:
    all_rows: list[dict] = []
    lang_rows: list[dict] = []
    pos_rows: list[dict] = []
    remap_rows: list[dict] = []

    for split in SPLITS:
        src = PRED_OUT / f"perceptron_pipeline_ud_transfer_{split}_predictions.conllu"
        for scheme_id, scheme_name in dm.SCHEME_NAMES.items():
            ea = en_adj_map if scheme_id == 3 else False
            r = dm.evaluate_mapped(src, scheme=scheme_id, en_adj_map=ea)
            all_rows.append({
                "scheme": scheme_name, "split": split, "accuracy": r["accuracy"],
                "non_punct_accuracy": r["non_punct_accuracy"], "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"], "total": r["total"], "correct": r["correct"],
            })
            for lang, (t, c) in r["lang_stats"].items():
                lang_rows.append({"scheme": scheme_name, "split": split, "lang": lang,
                                   "total": t, "correct": c, "accuracy": diag_acc(c, t)})
            for tag, v in r["tag_f1"].items():
                pos_rows.append({"scheme": scheme_name, "split": split, "tag": tag, **v})
            for (o, mp), cnt in r["remap_counts"].items():
                remap_rows.append({"scheme": scheme_name, "split": split,
                                    "original_pred": o, "mapped_pred": mp, "count": cnt})

    with (METRICS_OUT / "ud_transfer_mapping_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "split", "accuracy", "non_punct_accuracy",
                                            "macro_f1", "weighted_f1", "total", "correct"])
        w.writeheader(); w.writerows(all_rows)
    with (METRICS_OUT / "ud_transfer_mapping_by_lang.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "split", "lang", "total", "correct", "accuracy"])
        w.writeheader(); w.writerows(lang_rows)
    with (METRICS_OUT / "ud_transfer_mapping_by_pos.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "split", "tag", "precision", "recall", "f1", "support"])
        w.writeheader(); w.writerows(pos_rows)
    with (METRICS_OUT / "ud_transfer_mapping_confusion_before_after.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "split", "original_pred", "mapped_pred", "count"])
        w.writeheader(); w.writerows(remap_rows)
    print("  -> ud_transfer_mapping_results.csv, _by_lang.csv, _by_pos.csv, _confusion_before_after.csv")


# ---------------------------------------------------------------------------
# Step 6: Seen/unseen diagnostics (scoped to raw UD-transfer, matching
# diagnostic_seen_unseen.py's original per-model scope)
# ---------------------------------------------------------------------------

def compute_seen_unseen() -> None:
    print("[seen/unseen] Building training vocabulary from Panyo train (unchanged)...")
    seen_lower, _, _ = build_train_vocab(TRAIN_FILE)

    acc_rows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    lang_rows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    pos_rows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    model = "perceptron_pipeline_ud_transfer"

    for split in SPLITS:
        p = PRED_OUT / f"{model}_{split}_predictions.conllu"
        for tr in iter_token_records(p):
            lo = tr.form.lower()
            cond = "seen" if lo in seen_lower else "unseen"
            ok = 1 if tr.gold_upos == tr.pred_upos else 0
            acc_rows[(model, split, cond)][0] += 1
            acc_rows[(model, split, cond)][1] += ok
            lang_rows[(model, split, cond, tr.lang)][0] += 1
            lang_rows[(model, split, cond, tr.lang)][1] += ok
            pos_rows[(model, split, cond, tr.gold_upos)][0] += 1
            pos_rows[(model, split, cond, tr.gold_upos)][1] += ok

    with (SEEN_OUT / "seen_unseen_accuracy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "total", "correct", "accuracy", "error_count"])
        for (m, s, c), (t, cor) in sorted(acc_rows.items()):
            w.writerow([m, s, c, t, cor, diag_acc(cor, t), t - cor])
    with (SEEN_OUT / "seen_unseen_by_lang.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "lang", "total", "correct", "accuracy", "error_count"])
        for (m, s, c, lg), (t, cor) in sorted(lang_rows.items()):
            w.writerow([m, s, c, lg, t, cor, diag_acc(cor, t), t - cor])
    with (SEEN_OUT / "seen_unseen_by_pos.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "gold_upos", "total", "correct", "accuracy", "error_count"])
        for (m, s, c, up), (t, cor) in sorted(pos_rows.items()):
            w.writerow([m, s, c, up, t, cor, diag_acc(cor, t), t - cor])
    print("  -> seen_unseen_accuracy.csv, _by_lang.csv, _by_pos.csv")


# ---------------------------------------------------------------------------
# Step 7: Boundary diagnostics (scoped to raw UD-transfer)
# ---------------------------------------------------------------------------

def compute_boundary() -> None:
    region_acc: dict[tuple, list] = defaultdict(lambda: [0, 0])
    dir_acc: dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0])
    model = "perceptron_pipeline_ud_transfer"

    for split in SPLITS:
        p = PRED_OUT / f"{model}_{split}_predictions.conllu"
        for sent in iter_sentences(p):
            real = [t for t in sent["tokens"] if t["is_real"]]
            if not real:
                continue
            n = len(real)
            langs = [get_misc_lang(t["fields"][F_MISC]) for t in real]
            golds = [t["fields"][F_UPOS] for t in real]
            preds = [eb.get_pred_upos(t["fields"][F_MISC]) for t in real]
            regions = region_labels(langs)

            for i in range(n):
                ok = 1 if golds[i] == preds[i] else 0
                region_acc[(model, split, regions[i])][0] += 1
                region_acc[(model, split, regions[i])][1] += ok

            for i in range(n - 1):
                if not _is_ko_en_boundary(langs[i], langs[i + 1]):
                    continue
                direction = boundary_direction(langs[i], langs[i + 1])
                l_ok = 1 if golds[i] == preds[i] else 0
                r_ok = 1 if golds[i + 1] == preds[i + 1] else 0
                both = 1 if l_ok and r_ok else 0
                dir_acc[(model, split, direction)][0] += 1
                dir_acc[(model, split, direction)][1] += both
                dir_acc[(model, split, direction)][2] += l_ok
                dir_acc[(model, split, direction)][3] += r_ok

    with (BOUND_OUT / "boundary_region_accuracy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "region", "total", "correct", "accuracy", "error_count"])
        for (m, s, r), (t, c) in sorted(region_acc.items()):
            w.writerow([m, s, r, t, c, diag_acc(c, t), t - c])
    with (BOUND_OUT / "boundary_direction_accuracy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "direction", "total_boundary_pairs", "pair_both_correct",
                     "pair_accuracy", "left_token_acc", "right_token_acc"])
        for (m, s, d), (n_p, both, ls, rs) in sorted(dir_acc.items()):
            w.writerow([m, s, d, n_p, both, diag_acc(both, n_p), diag_acc(ls, n_p), diag_acc(rs, n_p)])
    print("  -> boundary_region_accuracy.csv, boundary_direction_accuracy.csv")


# ---------------------------------------------------------------------------
# Config record
# ---------------------------------------------------------------------------

def write_config(train_info: dict, pred_info: dict, map_info: dict, verify_info: dict) -> None:
    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fix_description": (
            "scripts/train_baselines.py::load_tagged_sents() previously passed "
            "lang='' for every token of monolingual UD corpora (GUM, KAIST) "
            "during UD-transfer training, while inference (predict_baselines.py) "
            "used real Lang=EN/KO values from the Panyo corpus, so all "
            "lang-derived perceptron features (lang=, p1.lang=, n1.lang=, "
            "p1.tag+lang=) never fired at inference. Fixed by adding a "
            "lang_override parameter: GUM training tokens now get lang='EN', "
            "KAIST training tokens now get lang='KO' -- identical string "
            "representation to get_misc_lang() at inference."
        ),
        "modified_source_files": ["scripts/train_baselines.py"],
        "unmodified_scripts_reused": [
            "scripts/perceptron_tagger.py", "scripts/predict_baselines.py",
            "scripts/diagnostic_ud_mapping.py", "scripts/evaluate_baselines.py",
            "scripts/diagnostic_seen_unseen.py", "scripts/diagnostic_boundary_analysis.py",
            "scripts/statistical_tests.py (paired-testing implementation, commit c4560bf -- no bug found, not altered)",
        ],
        "not_rerun": {
            "in_domain_models": [
                "majority_global", "majority_lang", "perceptron_joint", "perceptron_pipeline_panyo",
            ],
            "reason": (
                "These models never call the fixed code path (they load Panyo "
                "training data via load_tagged_sents(PANYO_TRAIN), which is "
                "unchanged -- it still reads real Lang= values from MISC). "
                "The fix only touches the two UD-transfer call sites. Rerunning "
                "them would reproduce bit-identical results at the cost of "
                "several minutes of retraining, so it was skipped."
            ),
            "diagnostic_residual_errors": (
                "scripts/diagnostic_residual_errors.py is scoped to "
                "PANYO_MODELS = [perceptron_joint, perceptron_pipeline_panyo] "
                "only; it never touches UD-transfer predictions, so it is "
                "unaffected by this fix and was not rerun."
            ),
        },
        "training_config": {
            "n_epochs": tb.N_EPOCHS,
            "seed": tb.SEED,
            "ud_release": "2.16",
            "ud_release_note": (
                "UD release version is not embedded in the CoNLL-U file headers "
                "themselves (verified: no '# udpipe'/version comment in "
                "en_gum-ud-train.conllu or ko_kaist-ud-train.conllu). Recorded "
                "here per instruction; repository inspection did not surface a "
                "different documented release, so 2.16 is used unchanged."
            ),
            "source_files_used": {
                "en_gum_train": {
                    "path": str(tb.GUM_TRAIN.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_of(tb.GUM_TRAIN),
                },
                "ko_kaist_train": {
                    "path": str(tb.KAIST_TRAIN.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_of(tb.KAIST_TRAIN),
                },
                "panyo_docsplit_dev": {
                    "path": str(pb.DEV_IN.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_of(pb.DEV_IN),
                },
                "panyo_docsplit_test": {
                    "path": str(pb.TEST_IN.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_of(pb.TEST_IN),
                },
                "panyo_docsplit_train_for_seen_unseen": {
                    "path": str(TRAIN_FILE.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_of(TRAIN_FILE),
                },
                "majority_lang_model_for_other_fallback": {
                    "path": "models/baselines/majority_lang.json",
                    "sha256": sha256_of(MODELS_ORIG / "majority_lang.json"),
                },
            },
        },
        "retrained_models": train_info,
        "corrected_predictions": pred_info,
        "mapping_variants": map_info,
        "token_identity_verification": verify_info,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT / 'config.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("=" * 70)
    print("STEP 1: Retrain corrected UD-transfer perceptrons")
    print("=" * 70)
    train_info = retrain_ud_transfer_models()

    print("\n" + "=" * 70)
    print("STEP 2: Generate corrected UD-transfer pipeline predictions")
    print("=" * 70)
    pred_info = generate_corrected_predictions()

    print("\n" + "=" * 70)
    print("STEP 2b: Verify doc_id/sent_id/token identity/gold UPOS preserved")
    print("=" * 70)
    verify_info = verify_token_identity_preserved()
    for split, r in verify_info.items():
        print(f"  {split}: {r}")

    print("\n" + "=" * 70)
    print("STEP 3: Generate mapping-variant predictions (conservative / panyo_coarse / ko_oriented)")
    print("=" * 70)
    map_info = generate_mapping_variants()

    print("\n" + "=" * 70)
    print("STEP 4: Main test metrics (accuracy, non-punct acc, macro F1, weighted F1, confusion)")
    print("=" * 70)
    compute_main_metrics()

    print("\n" + "=" * 70)
    print("STEP 5: Mapping-scheme metrics")
    print("=" * 70)
    compute_mapping_metrics(map_info["en_adj_map_scheme3"])

    print("\n" + "=" * 70)
    print("STEP 6: Seen/unseen diagnostics")
    print("=" * 70)
    compute_seen_unseen()

    print("\n" + "=" * 70)
    print("STEP 7: Boundary diagnostics")
    print("=" * 70)
    compute_boundary()

    print("\n" + "=" * 70)
    print("STEP 8: Write machine-readable configuration record")
    print("=" * 70)
    write_config(train_info, pred_info, map_info, verify_info)

    print("\nDone. All outputs under outputs/corrected_ud_transfer/.")


if __name__ == "__main__":
    main()
