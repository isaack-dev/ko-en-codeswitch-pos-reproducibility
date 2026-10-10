"""Separate descriptive v1.2 evaluation; never trains or changes frozen sources.

Runs reference gates before reading new correctness. Only PredUPOS is used for
models. The canonical silver source alone uses the CoNLL-U UPOS column.
"""
from __future__ import annotations

import collections
import importlib.metadata
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import audit_v12_reference as audit

ROOT, OUT = audit.ROOT, audit.OUT
sys.path.insert(0, str(ROOT / "scripts"))
from read_conllu import iter_sentences
from diagnostic_ud_mapping import apply_map, BASE_MAP_1, BASE_MAP_2
from build_test_only_adjudicated_reference import compute_per_class, macro_f1, weighted_f1

CLEAN = ROOT / "outputs/final_clean_reproducibility_run"
OLD_MANIFEST = CLEAN / "human_model_evaluation/model_evaluation_manifest.json"
OLD_ENV = CLEAN / "environment/environment.json"
UD_CONFIG = ROOT / "outputs/corrected_ud_transfer/config.json"
STRATA = ("1", "2", "3", "4", "5")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": audit.sha(path)}


def source_key(row):
    return row["source_sentence_id"], int(row["token_ordinal"])


def misc_item(misc, name):
    vals = [part[len(name)+1:] for part in misc.split("|") if part.startswith(name+"=")]
    audit.check(len(vals) <= 1, f"Duplicate {name} MISC field")
    return vals[0] if vals else ""


def load_stream(path, selected_sids, canonical=False):
    """Inspect the entire file, keeping selected original tokens only in memory."""
    selected, seen_sids, tags, n_tokens = {}, set(), set(), 0
    for sentence in iter_sentences(path):
        meta = {}
        for comment in sentence["comments"]:
            for name in ("sent_id", "doc_id", "text", "split"):
                prefix = f"# {name} = "
                if comment.startswith(prefix):
                    audit.check(name not in meta, f"Duplicate sentence {name}: {path}")
                    meta[name] = comment[len(prefix):]
        sid = meta.get("sent_id")
        audit.check(sid and sid not in seen_sids, f"Missing/duplicate sentence ID: {path}/{sid}")
        seen_sids.add(sid)
        if not canonical:
            audit.check(meta.get("split") == "test" and bool(meta.get("doc_id")), f"Not frozen TEST: {sid}")
        ids, ordinal = set(), 0
        for token in sentence["tokens"]:
            if not token["is_real"]:
                continue
            f = token["fields"]
            audit.check(f[0] not in ids, f"Duplicate original token ID: {sid}/{f[0]}")
            ids.add(f[0])
            ordinal += 1
            pred = f[3] if canonical else misc_item(f[9], "PredUPOS")
            audit.check(pred in audit.UPOS, f"Missing/invalid prediction: {path}/{sid}/{f[0]}")
            tags.add(pred)
            n_tokens += 1
            if sid in selected_sids:
                selected[sid, ordinal] = {"original_token_id": f[0], "token": f[1],
                    "sentence": meta.get("text"), "source_document_id": meta.get("doc_id"),
                    "prediction": pred, "retained_source_upos": f[3], "lang": misc_item(f[9], "Lang")}
    audit.check(selected_sids <= seen_sids, f"Missing selected sentences: {path}")
    return selected, {"file_sentences": len(seen_sids), "file_real_tokens": n_tokens,
                      "observed_prediction_inventory": sorted(tags)}


def align(reference, stream, canonical=False):
    keys = [source_key(row) for row in reference]
    audit.check(len(keys) == len(set(keys)), "Duplicate reference source token key")
    audit.check(set(keys) == set(stream), "Incomplete or extra selected prediction tokens")
    predictions = []
    for row in reference:
        token = stream[source_key(row)]
        for field in ("original_token_id", "token", "sentence"):
            audit.check(token[field] == row[field], f"Prediction source {field} differs: {source_key(row)}")
        if not canonical or token["source_document_id"] is not None:
            audit.check(token["source_document_id"] == row["source_document_id"], "Prediction document mismatch")
        audit.check(token["prediction"] in audit.UPOS, "Invalid prediction inventory")
        predictions.append(token["prediction"])
    return predictions


def provenance(reference):
    """Gate 2: check registered hashes and lineage before model scoring."""
    registered, env, ud = read_json(OLD_MANIFEST), read_json(OLD_ENV), read_json(UD_CONFIG)
    for name, expected in env["reused_script_checksums_sha256"].items():
        audit.check(audit.sha(ROOT/"scripts"/name) == expected, f"Frozen module changed: {name}")
    audit.check(ud["mapping_variants"]["en_adj_map_scheme3"] is True, "Frozen mapping configuration differs")
    canonical_path = ROOT/"data/kor_en_panyo_corpus.conllu"
    # The clean environment's 'canonical_key' is the historical sample XLSX,
    # not the full CoNLL-U corpus. Use the pre-access preservation baseline.
    baseline = audit.preserve()
    audit.check(audit.sha(canonical_path) == baseline["files"][canonical_path.relative_to(ROOT).as_posix()], "Canonical silver changed")
    sids = {r["source_sentence_id"] for r in reference}
    inventory, predictions, streams, alignment = [], {}, {}, {}
    specifications = [("canonical_silver", {"display": "Canonical silver", "prediction_file": str(canonical_path)})]
    specifications += list(registered["models"].items())
    for name, spec in specifications:
        path = Path(spec["prediction_file"])
        path = path if path.is_absolute() else ROOT/path
        checkpoints = []
        item = {"system": name, "display": spec["display"], "prediction_file": path.relative_to(ROOT).as_posix(),
            "prediction_sha256": "", "prediction_field": "UPOS (canonical only)" if name == "canonical_silver" else "MISC:PredUPOS",
            "status": "unavailable", "alignment_tokens": 0, "reason": "", "checkpoints": "[]",
            "configuration": "canonical corpus" if name == "canonical_silver" else "frozen clean run, training seed=42; no regeneration",
            "frozen_manifest": OLD_MANIFEST.relative_to(ROOT).as_posix(), "frozen_manifest_sha256": audit.sha(OLD_MANIFEST),
            "label_inventory": "", "generator_script": "scripts/predict_baselines.py"}
        if name == "canonical_silver":
            item["generator_script"] = "preserved canonical corpus; original web-assisted generation not exactly reproducible"
        try:
            item["prediction_sha256"] = audit.sha(path)
            if name != "canonical_silver":
                audit.check(item["prediction_sha256"] == spec["prediction_file_sha256"], "Registered prediction hash mismatch")
                cp = ROOT/spec["model_artifact"]
                audit.check(audit.sha(cp) == spec["model_artifact_sha256"], "Registered checkpoint hash mismatch")
                checkpoints = [cp]
                if name == "pangyo_pipeline_perceptron":
                    checkpoints += [cp.with_name(f"perceptron_pipeline_panyo_{lang}.json") for lang in ("en", "other")]
                    item["configuration"] += "; full-context KO/EN/OTHER routing; fallback=NOUN"
                if name.startswith("ud_transfer"):
                    checkpoints += [cp.with_name("perceptron_pipeline_ud_en_gum.json"), cp.with_name("majority_lang.json")]
                    item["configuration"] += "; corrected KAIST=KO/GUM=EN; OTHER fallback=PUNCT"
                item["checkpoints"] = json.dumps([record(c) for c in checkpoints], sort_keys=True)
            stream, details = load_stream(path, sids, canonical=name == "canonical_silver")
            preds = align(reference, stream, canonical=name == "canonical_silver")
            if name == "canonical_silver":
                test_stream, _ = load_stream(audit.sampling.SPLITS["test"], sids, canonical=True)
                audit.check(preds == align(reference, test_stream, canonical=True), "Canonical and frozen TEST silver differ")
            if name.startswith("ud_transfer"):
                raw_name = "ud_transfer_no_mapping"
                corrected = ROOT/"outputs/corrected_ud_transfer/predictions"/path.name
                audit.check(audit.sha(corrected) == item["prediction_sha256"], "Corrected lineage stream differs")
                if name != raw_name:
                    scheme = {"ud_transfer_conservative": 1, "ud_transfer_panyo_coarse": 2, "ud_transfer_ko_oriented": 3}[name]
                    audit.check(raw_name in streams, "Raw UD stream unavailable; cannot certify mapping")
                    raw = streams[raw_name]
                    for k, token in stream.items():
                        audit.check(token["lang"] == raw[k]["lang"], "Frozen routing language changed")
                        audit.check(token["prediction"] == apply_map(raw[k]["prediction"], raw[k]["lang"], scheme, True), "Frozen mapping stream differs")
                    item["configuration"] += f"; scheme={scheme}; frozen EN_ADJ_to_NOUN=true for scheme3"
                    item["generator_script"] += "; scripts/diagnostic_ud_mapping.py:apply_map"
            item.update(status="available", alignment_tokens=len(preds), label_inventory="|".join(details["observed_prediction_inventory"]))
            streams[name], predictions[name], alignment[name] = stream, preds, details
        except (audit.ValidationFailure, FileNotFoundError, ValueError) as error:
            item["reason"] = str(error)
            if name == "canonical_silver":
                raise
        inventory.append(item)
    audit.write_csv("frozen_prediction_inventory.csv", inventory)
    provenance_record = {"registered_model_manifest": record(OLD_MANIFEST), "clean_environment": record(OLD_ENV),
        "corrected_ud_configuration": record(UD_CONFIG), "corrected_command_log": record(ROOT/"outputs/corrected_ud_transfer/command_log.txt"),
        "clean_run_manifest": record(CLEAN/"manifest.json"),
        "frozen_script_hashes_verified": env["reused_script_checksums_sha256"],
        "mapping_definitions": {"conservative": BASE_MAP_1, "coarse": BASE_MAP_2,
            "ko_oriented": "unchanged apply_map scheme3, EN_ADJ flag=true from preserved corrected DEV selection"},
        "alignment": alignment, "inventory": inventory,
        "timing_limit": "Preserved clean-run records identify July configurations; hashes do not authenticate freeze time or nonaccess to new human outcomes."}
    (OUT/"prediction_provenance.json").write_bytes(audit.dump(provenance_record))
    return predictions, provenance_record


def scored(reference):
    audit.unique(reference)
    audit.unique(reference, source_key)
    indices = []
    for i, row in enumerate(reference):
        audit.check(row["reference_status"] in ("RESOLVED", "UNRESOLVED"), "Invalid reference status")
        if row["reference_status"] == "RESOLVED":
            audit.check(row["human_upos_v12"] in audit.UPOS, "Invalid resolved reference label")
            indices.append(i)
    return indices


def counts(reference, predictions, indices):
    audit.check(len(predictions) == len(reference) and all(p in audit.UPOS for p in predictions), "Missing/invalid prediction; denominator cannot shrink")
    n = len(indices)
    correct = sum(reference[i]["human_upos_v12"] == predictions[i] for i in indices)
    return {"correct": correct, "eligible_tokens": n, "accuracy": correct/n if n else "",
            "exact_accuracy": f"{correct}/{n}" if n else "undefined"}


def equal_stratum(parts):
    audit.check(len(parts) == 5 and all(p["eligible_tokens"] > 0 for p in parts), "Five nonempty strata required")
    return sum((Fraction(p["correct"], p["eligible_tokens"]) for p in parts), Fraction())/5


def evaluate(reference, predictions):
    indices = scored(reference)
    total = len(reference)
    primary, strata, equal, nonpunct, per_upos, confusion, coverage = [], [], [], [], [], [], []
    for system, preds in predictions.items():
        result = counts(reference, preds, indices)
        pairs = [(reference[i]["human_upos_v12"], preds[i]) for i in indices]
        classes = compute_per_class(pairs)
        primary.append({"system": system, **result, "full_sample_tokens": total, "resolved_tokens": len(indices),
            "unresolved_tokens": total-len(indices), "reference_coverage": len(indices)/total,
            "macro_f1_positive_reference_support": macro_f1(classes), "support_weighted_f1": weighted_f1(classes),
            "positive_reference_categories": sum(v["support"] > 0 for v in classes.values()),
            "reference_absent_labels": "|".join(tag for tag in audit.UPOS if classes.get(tag, {}).get("support", 0) == 0),
            "full_sample_lower_correct": result["correct"], "full_sample_upper_correct": result["correct"]+total-len(indices),
            "full_sample_bound_denominator": total, "algebraic_lower_bound": result["correct"]/total,
            "algebraic_upper_bound": (result["correct"]+total-len(indices))/total})
        nonpunct.append({"system": system, **counts(reference, preds, [i for i in indices if reference[i]["human_upos_v12"] != "PUNCT"]),
                         "exclusion": "human-reference PUNCT only"})
        parts = []
        for stratum in STRATA:
            eligible = [i for i in indices if reference[i]["stratum"] == stratum]
            full = sum(r["stratum"] == stratum for r in reference)
            part = {"system": system, "stratum": stratum, **counts(reference, preds, eligible),
                    "full_stratum_tokens": full, "unresolved_tokens": full-len(eligible)}
            strata.append(part)
            parts.append(part)
        mean = equal_stratum(parts)
        equal.append({"system": system, "strata": 5, "weight_each": "1/5", "equal_stratum_accuracy": float(mean),
                      "exact_mean_numerator": mean.numerator, "exact_mean_denominator": mean.denominator})
        matrix = collections.Counter(pairs)
        predicted = collections.Counter(preds[i] for i in indices)
        supported = collections.Counter(reference[i]["human_upos_v12"] for i in indices)
        for tag in audit.UPOS:
            cl = classes.get(tag, {"support": 0, "precision": 0., "recall": 0., "f1": 0.})
            tp = matrix[tag, tag]
            per_upos.append({"system": system, "upos": tag, **cl, "predicted_count": predicted[tag],
                "true_positive": tp, "false_positive": predicted[tag]-tp, "false_negative": supported[tag]-tp,
                "reference_absent": supported[tag] == 0, "included_in_macro": supported[tag] > 0,
                "zero_denominator_convention": "0 for undefined precision/recall/F1; absent reference excluded from macro"})
            for pred in audit.UPOS:
                confusion.append({"system": system, "human_upos": tag, "predicted_upos": pred, "tokens": matrix[tag, pred]})
        for group in ("ALL", *STRATA):
            full = [i for i, r in enumerate(reference) if group == "ALL" or r["stratum"] == group]
            resolved = [i for i in full if i in set(indices)]
            correct = counts(reference, preds, resolved)["correct"]
            coverage.append({"system": system, "group": group, "source_tokens": len(full), "resolved_tokens": len(resolved),
                "unresolved_tokens": len(full)-len(resolved), "coverage": len(resolved)/len(full),
                "correct_resolved": correct, "algebraic_lower_correct": correct,
                "algebraic_upper_correct": correct+len(full)-len(resolved), "bound_denominator": len(full),
                "algebraic_lower_bound": correct/len(full), "algebraic_upper_bound": (correct+len(full)-len(resolved))/len(full),
                "bound_interpretation": "full-sample algebraic sensitivity bounds, not confidence intervals"})
    audit.check(len({r["eligible_tokens"] for r in primary}) == 1, "Inconsistent paired-system denominator")
    outputs = {"v12_primary_results.csv": primary, "v12_stratum_results.csv": strata,
        "v12_equal_stratum_summary.csv": equal, "v12_nonpunct_results.csv": nonpunct,
        "v12_upos_metrics.csv": per_upos, "v12_confusion_matrices.csv": confusion, "v12_reference_coverage.csv": coverage}
    for name, rows in outputs.items():
        audit.write_csv(name, rows)
    return outputs


def historical():
    silver_path = CLEAN/"human_reference/human_silver_metrics.json"
    rep_path = CLEAN/"human_model_evaluation/model_metrics_representative.csv"
    diag_path = CLEAN/"human_model_evaluation/model_metrics_diagnostic.csv"
    legacy = read_json(silver_path)["human_vs_canonical"]
    rep, diag = audit.csv_rows(rep_path), audit.csv_rows(diag_path)
    text = "# Historical v1.1 context — kept separate\n\n"
    text += ("The historical independent annotations used v1.1. Its adjudication/post-consistency records reference a provisional v1.2 draft; this does not make those independent annotations prospective v1.2 data. The 60-sentence representative reference (1,063 tokens) and targeted 20-sentence diagnostic subset (816 tokens) remain distinct from each other and from the new 180-sentence v1.2 reference. No historical scores were recalculated or harmonized here.\n\n"
        "Historical accuracy includes punctuation. Historical non-PUNCT uses final human-reference PUNCT exclusion. Reported results below are read directly from preserved corrected clean-run files; existing confidence intervals/tests were neither rerun nor transferred to the new sample.\n\n"
        "| Source | Representative correct / 1,063 | Accuracy | Non-PUNCT correct / 924 | Accuracy |\n|:--|--:|--:|--:|--:|\n")
    s = legacy["Representative"]
    text += f"| Canonical silver | {s['correct']} | {100*s['agreement']:.4f}% | {s['non_punct_correct']} | {100*s['non_punct_agreement']:.4f}% |\n"
    for r in rep:
        text += f"| {r['model_display']} | {r['correct']} | {100*float(r['accuracy']):.4f}% | {r['non_punct_correct']} | {100*float(r['non_punct_accuracy']):.4f}% |\n"
    text += "\nDiagnostic-only historical sample (selection targeted; never pool):\n\n| Source | Correct / 816 | Accuracy |\n|:--|--:|--:|\n"
    s = legacy["Diagnostic"]
    text += f"| Canonical silver | {s['correct']} | {100*s['agreement']:.4f}% |\n"
    for r in diag:
        text += f"| {r['model_display']} | {r['correct']} | {100*float(r['accuracy']):.4f}% |\n"
    text += "\nNumerical differences across guideline versions, sample composition, and reference construction cannot establish a guideline effect or model improvement. Historical independent records, adjudication, scripts, configurations and outputs are protected by the preservation baseline.\n\nSources:\n\n"
    for path in (silver_path, rep_path, diag_path):
        text += f"- `{path.relative_to(ROOT).as_posix()}` — SHA-256 `{audit.sha(path)}`\n"
    text += "- Original materials: `outputs/human_validation/test_only/`; historical evaluator: `scripts/evaluate_models_on_test_only_human_reference.py`; reference builder: `scripts/build_test_only_adjudicated_reference.py`.\n"
    (OUT/"v11_historical_context.md").write_text(text, encoding="utf-8")


def documentation(outputs, validation):
    primary, nonpunct = outputs["v12_primary_results.csv"], {r["system"]: r for r in outputs["v12_nonpunct_results.csv"]}
    equal = {r["system"]: r for r in outputs["v12_equal_stratum_summary.csv"]}
    strata = {(r["system"], r["stratum"]): r for r in outputs["v12_stratum_results.csv"]}
    support = [r for r in outputs["v12_upos_metrics.csv"] if r["system"] == "canonical_silver"]
    methods = ("# Verified evaluation methods — preparation, not manuscript revision\n\n"
        "The separate v1.2 analysis follows `reports/analysis_plans/pangyo_v1.2/PANGYO_V12_ANALYSIS_PLAN_FREEZE_2026-10-01.md` unchanged. Two independent submissions per batch were source-validated, pre-adjudication agreement reconstructed, and every final decision reconciled to preserved blind adjudication/reconciliation records before model correctness was computed. Historical independence, actual nonaccess to keys, and timing remain researcher attestations rather than independently authenticated events.\n\n"
        "The frozen extension contains 180 TEST sentences from 180 distinct paragraph/document IDs, with 36 sentences per original length band (1–12, 13–15, 16–18, 19–22, 23+). Seed 20261001 and constrained document matching were fixed in the original sampler; old annotated documents/text duplicates and train/dev exact-normalized sentence overlaps were excluded. Source tokenization and IDs were unchanged. Article-level independence is not established by paragraph IDs. Batch A contains 60 sentences/1,046 tokens; Batch B 120/2,140. No DEV calibration, historical representative, or diagnostic tokens enter the new outcome.\n\n"
        "Primary accuracy is exact correct/scorable token count over combined A+B including punctuation. All 3,186 tokens are resolved; coverage is 100%. Five band accuracies and their equal 20% mean are secondary descriptive outcomes. Neither aggregation estimates population accuracy without design qualifications: equal sentence quotas yield unequal token counts and the matching procedure does not provide simple token inclusion probabilities. No population weights, new significance tests, bootstrap intervals, or model-winner claims were computed.\n\n"
        "Frozen predictions were selected from the corrected clean-run lineage and checked against registered prediction/checkpoint/module hashes. Model predictions come exclusively from CoNLL-U MISC `PredUPOS`, never the retained source UPOS column. Canonical silver is taken from `data/kor_en_panyo_corpus.conllu` and checked against frozen TEST silver. Exact sentence/document/context, original token IDs/forms, and real-token ordinals align all sources to all reference tokens. Mapped UD outputs agree with unchanged `apply_map`; Conservative changes PART→ADP, SYM→PUNCT, X→NOUN and leaves ADJ unchanged. Scheme3 retains the historically DEV-selected EN ADJ mapping flag. No retraining, regeneration or rule tuning occurred.\n\n"
        "Human non-PUNCT evaluation excludes only adjudicated human PUNCT, retaining erroneous PUNCT predictions. Human–human non-PUNCT excludes either annotator's PUNCT. Per-UPOS precision/recall/F1 and full 17×17 confusion matrices use the adjudicated reference. Supported classes with no predictions have F1=0. Macro F1 averages only positive-reference-support classes; weighted F1 uses reference support. Absent reference labels are disclosed; undefined per-label scores use zero. Every system uses the same scorable set. Coverage reports include full-sample algebraic bounds, which collapse to point values with zero unresolved tokens and are not confidence intervals. Missing model predictions make a system unavailable, not a smaller denominator.\n\n"
        "Suggested reporting: primary/secondary/non-PUNCT comparison table; five-band descriptive table; UPOS support/F1 table and confusion diagnostics; separate historical-context table. Exact counts and full-precision values are in the CSVs. Observed differences across versions are not causal effects. IAA describes annotation consistency, not independent semantic validity. Canonical web-assisted generation cannot be claimed exactly reproducible from model outputs.\n")
    (OUT/"evaluation_methods_for_paper.md").write_text(methods, encoding="utf-8")
    text = "# Verified v1.2 results — camera-ready preparation\n\nGates 0–3 passed; historical results remain separate. All 3,186 tokens are resolved and every included stream fully aligns.\n\n"
    m = validation["pre_adjudication"]
    text += f"Independent pre-adjudication agreement: {m['raw_agree']}/{m['tokens']} = {100*m['raw_agreement']:.4f}%; nominal unweighted κ={m['cohens_kappa_nominal_unweighted_17tags']:.10f}. Human–human non-PUNCT: {m['nonpunct_agree']}/{m['nonpunct_n']} = {100*m['nonpunct_agreement']:.4f}%. 136 original UPOS disagreements and 22 boundary judgments were resolved; three adjacent proposals produced no final label changes.\n\n"
    text += "| Source | Correct / 3,186 | Primary % | Equal-stratum % | Non-PUNCT correct / eligible | Non-PUNCT % | Supported macro F1 | Weighted F1 |\n|:--|--:|--:|--:|--:|--:|--:|--:|\n"
    for r in primary:
        n, e = nonpunct[r["system"]], equal[r["system"]]
        text += f"| {r['system']} | {r['correct']} | {100*r['accuracy']:.4f} | {100*e['equal_stratum_accuracy']:.4f} | {n['correct']}/{n['eligible_tokens']} | {100*n['accuracy']:.4f} | {r['macro_f1_positive_reference_support']:.6f} | {r['support_weighted_f1']:.6f} |\n"
    text += "\nFive prespecified strata (correct/eligible and accuracy):\n\n| Source | 1–12 | 13–15 | 16–18 | 19–22 | 23+ |\n|:--|--:|--:|--:|--:|--:|\n"
    for r in primary:
        parts = [strata[r["system"], s] for s in STRATA]
        text += "| "+r["system"]+" | "+" | ".join(f"{p['correct']}/{p['eligible_tokens']} ({100*p['accuracy']:.2f}%)" for p in parts)+" |\n"
    text += "\nReference UPOS support and canonical-silver diagnostics:\n\n| UPOS | Reference support | Silver precision | Silver recall | Silver F1 |\n|:--|--:|--:|--:|--:|\n"
    for r in support:
        text += f"| {r['upos']} | {r['support']} | {r['precision']:.6f} | {r['recall']:.6f} | {r['f1']:.6f} |\n"
    text += f"\nReference-absent labels: {primary[0]['reference_absent_labels'] or 'none'}. Full per-system metrics and all 289 confusion cells per system are retained in `v12_upos_metrics.csv` and `v12_confusion_matrices.csv`.\n\n"
    offdiag = sorted((r for r in outputs["v12_confusion_matrices.csv"] if r["system"] == "canonical_silver" and r["human_upos"] != r["predicted_upos"] and r["tokens"]), key=lambda r: (-r["tokens"], r["human_upos"], r["predicted_upos"]))[:10]
    text += "Largest canonical-silver confusions (human → silver; descriptive, not new mapping rules):\n\n"
    for r in offdiag:
        text += f"- {r['human_upos']} → {r['predicted_upos']}: {r['tokens']} tokens.\n"
    text += "\nReady to report: source counts, independently reconstructed IAA, documented resolution, exact descriptive system results and prespecified strata/UPOS diagnostics. Unsupported: authenticated preregistration timing, historical nonaccess/independence, causal guideline effects, population accuracy, inferential system superiority, and exact reproduction of original ChatGPT silver generation. Registered original calibration metric files are unavailable; plan-reported readiness numbers remain context only. Historical v1.1 values are in `v11_historical_context.md` and must not be pooled.\n\n"
    text += "Authoritative supporting outputs reside in `outputs/camera_ready/v12_integration/`: `reference_validation_report.md`, `annotation_process_audit.md`, `input_and_provenance_manifest.json`, `prediction_provenance.json`, `frozen_prediction_inventory.csv`, the seven `v12_*.csv` evaluation tables, `test_report.txt`, and `README.md`. `aligned_v12_predictions.csv` is INTERNAL ONLY; never send to annotators.\n"
    (OUT/"camera_ready_result_summary.md").write_text(text, encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("openpyxl", "numpy")},
        "commands": ["python -B scripts/camera_ready/evaluate_v12_reference.py", "python -B scripts/camera_ready/test_v12_integration.py"],
        "scripts": [record(ROOT/"scripts/camera_ready"/name) for name in ("audit_v12_reference.py", "evaluate_v12_reference.py", "test_v12_integration.py") if (ROOT/"scripts/camera_ready"/name).exists()],
        "legacy_modules": [record(ROOT/"scripts"/name) for name in ("read_conllu.py", "build_test_only_adjudicated_reference.py", "diagnostic_ud_mapping.py")],
        "randomness": "No new random sampling or inference; selection seed 20261001 and model seed42 are historical fixed inputs."}
    (OUT/"execution_environment.json").write_bytes(audit.dump(environment))
    (OUT/"README.md").write_text(
        "# v1.2 integration outputs — RESEARCHER ONLY\n\n"
        "Run from `C:\\Python\\Programs\\CCIR` with the existing Python 3.12 environment (openpyxl and numpy; see `execution_environment.json`). No installs, training, regeneration, calibration scoring or manuscript edits are required.\n\n"
        "```powershell\npython -B scripts/camera_ready/evaluate_v12_reference.py\npython -B scripts/camera_ready/test_v12_integration.py\n```\n\n"
        "The evaluator reruns gates 0–1 before accessing model correctness, certifies gate2 prediction hashes/identity and writes only new integration reports. The test runner checks synthetic edge cases, the full frozen-input integration, and two repeated executions for byte-identical serialized result/report outputs. Do not run archived packaging scripts. All 786 preexisting file hashes (including inputs) must match `preservation_checksums.json`; prior working-tree edits are preserved. Source/protocol mismatch stops processing. A model with unverifiable predictions is unavailable; it cannot silently lose unmatched tokens.\n\n"
        "`camera_ready_result_summary.md` and `evaluation_methods_for_paper.md` provide verified preparation materials. Source audit: `input_and_provenance_manifest.json`, `reference_validation.json`, `reference_validation_report.md`, `annotation_process_audit.md`, `sampling_verification_log.txt`, IAA/confusion and decision trace. Prediction lineage: `prediction_provenance.json`, `frozen_prediction_inventory.csv`. Exact results: `v12_primary_results.csv`, `v12_stratum_results.csv`, `v12_equal_stratum_summary.csv`, `v12_nonpunct_results.csv`, `v12_upos_metrics.csv`, `v12_confusion_matrices.csv`, `v12_reference_coverage.csv`. Historical context is separate in `v11_historical_context.md`. Test outcome is recorded only after successful actual execution in `test_report.txt` and `repeatability_validation.json`.\n\n"
        "**Never distribute** aligned predictions, human responses/notes, reference labels, decision records or these internal archives to annotators. The full aligned CSV contains human labels, canonical silver and eight model streams. Historical human data, DEV calibration and v1.2 TEST remain separate. All outcomes are descriptive; no new weights, intervals, hypothesis tests, mapping development or causal claims. See the audits for documentary and historical-timing limitations.\n", encoding="utf-8")


def run():
    reference, manifest, validation = audit.run()
    predictions, pred_provenance = provenance(reference)
    aligned = [{**{f: row[f] for f in audit.SOURCE_FIELDS}, "human_upos_v12": row["human_upos_v12"],
        "reference_status": row["reference_status"], "tokenization_issue_v12": row["tokenization_issue_v12"],
        **{system: preds[i] for system, preds in predictions.items()}} for i, row in enumerate(reference)]
    audit.write_csv("aligned_v12_predictions.csv", aligned)
    results = evaluate(reference, predictions)
    historical()
    documentation(results, validation)
    manifest["prediction_evaluation_provenance"] = pred_provenance
    manifest["new_evaluation_scripts"] = read_json(OUT/"execution_environment.json")["scripts"]
    (OUT/"input_and_provenance_manifest.json").write_bytes(audit.dump(manifest))
    audit.preserve()
    blocker = OUT/"evaluation_blocker.json"
    if blocker.exists():
        previous = read_json(blocker)
        if not previous.get("resolved"):
            previous.update(resolved=True, active=False, resolution="Completed gated evaluation passed. Initial canonical_key lookup referred to historical XLSX; corrected evaluator to compare actual full-corpus preservation checksum. No frozen source was changed.")
            blocker.write_bytes(audit.dump(previous))
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = run()
        print(json.dumps({"gates_0_3_passed": True, "results": result["v12_primary_results.csv"]}, indent=2))
    except (audit.ValidationFailure, FileNotFoundError, KeyError) as error:
        (OUT/"evaluation_blocker.json").write_bytes(audit.dump({"error": str(error), "complete": False}))
        print("STOP: material validation failure:", error)
        raise SystemExit(1)
