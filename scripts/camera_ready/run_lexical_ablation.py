"""Controlled full/identity/affix ablations of original joint and routed models.

The unchanged tagger's feature callable is filtered in an isolated process;
training, averaging, tag vocabulary, tie-breaking and decoding are untouched.
DEV precedes the explicit silver TEST gate. Never scores any human reference.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import additional_diagnostics as d
from predict_baselines import PerceptronPredictor, PipelinePanyoPredictor
from read_conllu import read_sentences, write_sentences, get_misc_lang

ORIGINAL_FEATURES=d.original.get_features
IDENTITY=("w=","p1=","n1=")
AFFIX=("pre1=","pre2=","pre3=","suf1=","suf2=","suf3=")
ARCHITECTURES=("joint","routed")
MODEL_NAMES={"joint":"perceptron_joint","KO":"perceptron_pipeline_panyo_ko",
             "EN":"perceptron_pipeline_panyo_en","OTHER":"perceptron_pipeline_panyo_other"}


def features(config,*args):
    original=ORIGINAL_FEATURES(*args)
    removed=() if config=="A" else IDENTITY if config=="B" else IDENTITY+AFFIX
    d.require(config in "ABC","Unknown ablation configuration")
    return [f for f in original if not f.startswith(removed)] if removed else original


@contextlib.contextmanager
def feature_configuration(config):
    previous=d.original.get_features
    d.original.get_features=ORIGINAL_FEATURES if config=="A" else lambda *args:features(config,*args)
    try:
        yield
    finally:
        d.original.get_features=previous


def freeze():
    d.preserve()
    config={"experiment_id":"PANGYO-CR-POSTREVIEW-LEXICAL-ABLATION-001","configurations":{
        "A":{"description":"original full-feature baseline","remove_prefixes":[]},
        "B":{"description":"remove current/neighbor word identities; retain character affixes","remove_prefixes":list(IDENTITY)},
        "C":{"description":"remove word identities and all current-token prefixes/suffixes","remove_prefixes":list(IDENTITY+AFFIX)}},
        "architectures":list(ARCHITECTURES),"epochs":d.N_EPOCHS,"seed":d.SEED,
        "training":"original load_tagged_sents, same frozen TRAIN sentence order; unchanged original train with local random.Random(42)",
        "routed":"full sentence context, updates only matching KO/EN/OTHER; all-training sorted tag vocabulary; full-sentence language-specific decoding; fallback=NOUN",
        "retained_features":["bias","lang=","shape=","hgl=","lat=","dig=","p1.lang=","n1.lang=","p1.tag=","p2.tag=","p1.tag+lang="],
        "boundary_note":"Removing p1=/n1= also removes BOS/EOS values carried by those identity features; no replacement boundary feature is introduced.",
        "unchanged":"Perceptron updates, averaging, 9-decimal weight rounding, greedy history, tag ties, initialization, epochs, RNG, labels, source tokenization, corpus and split",
        "seen_definition":"exact Python Unicode FORM.lower() membership in frozen Pangyo TRAIN, across all languages; unseen complement",
        "nonpunct_definition":"exclude only frozen silver-reference UPOS=PUNCT; predictions do not determine eligibility",
        "test_groups":["ALL/punctuation included","ALL/nonpunct","seen/punctuation included","unseen/punctuation included","seen/nonpunct","unseen/nonpunct"],
        "gates":["verify original modules/split hashes","A checkpoint reproduction and exact DEV predictions/counts","B/C DEV verification","record source/config hashes before silver TEST","TEST A exact frozen-output reproduction before interpreting B/C"],
        "scope":"Post-review descriptive silver-reference diagnostics; new models are not inserted into frozen v1.2 evaluation; no human scoring, new mapping, model selection or significance testing.",
        "known_outcome_access":"Frozen full-feature silver outcomes and v1.2 human outcomes were already available; this freeze does not claim original prospective preregistration."}
    d.put_json("ablation_configurations.json",config)
    paths=[d.ROOT/"scripts/camera_ready"/s for s in d.SCRIPTS]+[d.ROOT/"scripts"/s for s in
        ("perceptron_tagger.py","train_baselines.py","predict_baselines.py","read_conllu.py")]
    source={"config":d.record(d.OUT/"ablation_configurations.json"),"source_files":[d.record(p) for p in paths],
        "split_files":[d.record(p) for p in d.SPLITS.values()],
        "partition_audit":d.record(d.OUT/"partition_and_existing_unseen_audit.json")}
    if (d.OUT/"ablation_freeze_record.json").exists():
        old=d.read_json(d.OUT/"ablation_freeze_record.json")
        d.require(old["frozen_content"]==source,"Existing ablation source/config freeze differs")
    else:
        d.put_json("ablation_freeze_record.json",{"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frozen_content":source,"timestamp_caveat":"Local execution record, not third-party preregistration"})
        # Exact byte copies of ONLY new ablation/audit/test code under authorized new output root.
        for path in paths[:3]:
            d.put("frozen_source/"+path.name,path.read_bytes())
    print("Configurations/code/checksums frozen before new TEST execution",flush=True)


def verify_freeze():
    freeze=d.read_json(d.OUT/"ablation_freeze_record.json")["frozen_content"]
    for entry in [freeze["config"],*freeze["source_files"],*freeze["split_files"],freeze["partition_audit"]]:
        d.require(d.sha(d.ROOT/entry["path"])==entry["sha256"],f"Ablation frozen content changed: {entry['path']}")
    return freeze


def metric_rows(config,architecture,split,sentences,vocabulary):
    counters={(condition,punct):[0,0] for condition in ("ALL","seen","unseen") for punct in ("included","nonpunct")}
    for sent in sentences:
        for token in sent["tokens"]:
            if not token["is_real"]:
                continue
            f=token["fields"]
            values=[v[9:] for v in f[9].split("|") if v.startswith("PredUPOS=")]
            d.require(len(values)==1,"Missing/duplicate PredUPOS")
            pred=values[0]
            d.require(pred in {"ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"},"Invalid predicted UPOS")
            condition="seen" if f[1].lower() in vocabulary else "unseen"
            for c in ("ALL",condition):
                for punctuation in (("included","nonpunct") if f[3]!="PUNCT" else ("included",)):
                    counters[c,punctuation][0]+=1
                    counters[c,punctuation][1]+=pred==f[3]
    return [{"configuration":config,"architecture":architecture,"split":split,"condition":condition,"punctuation":punct,
        "eligible_tokens":n,"correct":correct,"errors":n-correct,"accuracy":correct/n if n else "",
        "exact_accuracy":f"{correct}/{n}","reference":"frozen silver"} for (condition,punct),(n,correct) in sorted(counters.items())]


def verify_prediction_source(source,predicted):
    d.require(len(source)==len(predicted),"Prediction sentence count mismatch")
    for sent,pred in zip(source,predicted):
        d.require(sent["comments"]==pred["comments"] and len(sent["tokens"])==len(pred["tokens"]),"Prediction context/structure changed")
        for a,b in zip(sent["tokens"],pred["tokens"]):
            d.require(a["is_real"]==b["is_real"] and a["fields"][:9]==b["fields"][:9],"Prediction original token/label altered")
            old=[v for v in a["fields"][9].split("|") if not v.startswith("PredUPOS=") and v!="_"]
            new=[v for v in b["fields"][9].split("|") if not v.startswith("PredUPOS=") and v!="_"]
            d.require(old==new,"Prediction original MISC altered")


def original_prediction_file(architecture,split):
    model="perceptron_joint" if architecture=="joint" else "perceptron_pipeline_panyo"
    return d.CLEAN/f"baselines/results/{model}_{split}_predictions.conllu"


def verify_a_counts(results,split):
    old=d.rows(d.CLEAN/"baselines/results/overall_results.csv")
    unseen=d.rows(d.CLEAN/"baselines/diagnostics/seen_unseen_accuracy.csv")
    for row in results:
        model="perceptron_joint" if row["architecture"]=="joint" else "perceptron_pipeline_panyo"
        if row["condition"]=="ALL":
            expected=next(r for r in old if r["model"]==model and r["split"]==split)
            n,c=("tokens","correct") if row["punctuation"]=="included" else ("non_punct_tokens","non_punct_correct")
        elif row["punctuation"]=="included":
            expected=next(r for r in unseen if r["model"]==model and r["split"]==split and r["condition"]==row["condition"])
            n,c="total","correct"
        else:
            continue  # exact A prediction identity already establishes these extra cells
        d.require((row["eligible_tokens"],row["correct"])==(int(expected[n]),int(expected[c])),"Full-feature A does not reproduce original metrics")


def stage(config,split):
    frozen=verify_freeze()
    d.preserve()
    completed=d.OUT/f"{split}_validation/{config}.json"
    if completed.exists():
        record=d.read_json(completed)
        for entry in record["outputs"]:
            d.require(d.sha(d.ROOT/entry["path"])==entry["sha256"],"Completed stage artifact changed")
        print(f"Verified existing {config}/{split} stage without overwriting",flush=True)
        return
    if config!="A":
        d.require((d.OUT/"dev_validation/A.json").exists(),"A DEV reproduction must pass before ablated training")
    if split=="test":
        for c in "ABC":
            d.require((d.OUT/f"dev_validation/{c}.json").exists(),"All DEV stages must pass before TEST")
        if config!="A":
            d.require((d.OUT/"test_validation/A.json").exists(),"A TEST reproduction must pass before ablated TEST")
        d.put_json("silver_test_execution_gate.json",{"config_freeze":d.record(d.OUT/"ablation_freeze_record.json"),
            "dev_validations":[d.record(d.OUT/f"dev_validation/{c}.json") for c in "ABC"],
            "source_hashes_verified_before_test":True,"human_reference_evaluation_performed":False})
    models={}
    training_records=[]
    with feature_configuration(config):
        if split=="dev":
            train=d.load_tagged_sents(d.SPLITS["train"])
            for name,original_name in MODEL_NAMES.items():
                path=d.OUT/f"models/{config}/{original_name}.json"
                if path.exists():
                    raise ValueError(f"Incomplete existing model stage; refusing overwrite: {path}")
                model=d.original.AveragedPerceptron()
                print(f"Training {config}/{name}, original five epochs/seed42",flush=True)
                start=time.perf_counter()
                log=io.StringIO()
                with contextlib.redirect_stdout(log):
                    model.train(train,n_epochs=d.N_EPOCHS,seed=d.SEED,lang_filter=None if name=="joint" else name)
                    model.save(path)
                d.put(f"training_logs/{config}_{name}.txt",log.getvalue())
                d.require(not any(f.startswith(IDENTITY if config=="B" else IDENTITY+AFFIX) for f in model.weights) if config!="A" else True,"Removed feature in trained weights")
                if config=="A":
                    original_path=d.CLEAN/f"baselines/models/{original_name}.json"
                    d.require(d.sha(path)==d.sha(original_path),f"A checkpoint fails exact reproduction: {name}")
                training_records.append({"model":name,"training_seconds":time.perf_counter()-start,"feature_count":len(model.weights),
                    "epochs":d.N_EPOCHS,"seed":d.SEED,"steps":sum(map(len,train))*d.N_EPOCHS,"model_file":d.record(path),
                    "exact_original_checkpoint_match":config=="A"})
                models[name]=model
                print(f"Completed {config}/{name}: {len(model.weights)} learned features; {training_records[-1]['training_seconds']:.1f}s",flush=True)
            d.put_json(f"training_records/{config}.json",training_records)
        else:
            models={name:d.original.AveragedPerceptron.load(d.OUT/f"models/{config}/{original_name}.json") for name,original_name in MODEL_NAMES.items()}
        source=read_sentences(d.SPLITS[split])
        predictors={"joint":PerceptronPredictor(models["joint"]),
                    "routed":PipelinePanyoPredictor(models["KO"],models["EN"],models["OTHER"],"NOUN")}
        vocab={f.lower() for sent in d.load_tagged_sents(d.SPLITS["train"]) for f,l,t in sent}
        results,outputs=[],[]
        for architecture,predictor in predictors.items():
            predicted=predictor.predict_sents(source)
            verify_prediction_source(source,predicted)
            path=d.OUT/f"predictions/{config}/{architecture}_{split}_predictions.conllu"
            d.require(not path.exists(),f"Refusing prediction overwrite: {path}")
            write_sentences(predicted,path)
            reopened=read_sentences(path)
            d.require(reopened==predicted,"Prediction reopen mismatch")
            if config=="A":
                d.require(d.sha(path)==d.sha(original_prediction_file(architecture,split)),f"A {architecture}/{split} predictions not byte-identical to frozen original")
            results+=metric_rows(config,architecture,split,predicted,vocab)
            outputs.append(d.record(path))
        if config=="A":
            verify_a_counts(results,split)
        d.put_csv(f"metrics/{config}_{split}.csv",results)
        outputs.append(d.record(d.OUT/f"metrics/{config}_{split}.csv"))
        outputs += [d.record(d.OUT/f"models/{config}/{name}.json") for name in MODEL_NAMES.values()]
        verify_freeze()
        d.preserve()
        d.put_json(f"{split}_validation/{config}.json",{"configuration":config,"split":split,"original_source_unchanged":True,
            "source_token_identity_checked":True,"predictions_reopened_verified":True,"reference":"frozen silver only",
            "original_A_checkpoint_and_prediction_identity":config=="A","original_A_counts_match":config=="A",
            "frozen_config_sha256":frozen["config"]["sha256"],"outputs":outputs})
        print(f"Validated {config}/{split}; silver accuracy: "+str([(r['architecture'],r['exact_accuracy']) for r in results if r['condition']=='ALL' and r['punctuation']=='included']),flush=True)


def finalize():
    verify_freeze()
    preservation=d.preserve()
    result=[]
    for config in "ABC":
        for split in ("dev","test"):
            d.require((d.OUT/f"{split}_validation/{config}.json").exists(),"Incomplete ablation experiment")
            result+=d.rows(d.OUT/f"metrics/{config}_{split}.csv")
    d.put_csv("ablation_results.csv",result)
    test=d.read_json(d.OUT/"test_validation_report.json")
    d.require(test["passed"],"Diagnostics tests have not passed")
    lookup={(r["configuration"],r["architecture"],r["split"],r["condition"],r["punctuation"]):r for r in result}
    report="# Lexical ablation interpretation — post-review silver diagnostics\n\n"
    report+="Both original in-domain architectures were trained under A (full), B (remove w=/p1=/n1= identities), and C (remove identities plus six affixes). Original five epochs, seed42, sentence ordering/shuffling, greedy history, averaging, labels, supplied Lang and routing were preserved. A reproduces all four original checkpoint files and both DEV/TEST prediction streams byte-identically, including all exact archived metric counts. B/C DEV verification preceded TEST; source/config freeze hashes were checked before every stage. No DEV score was used to select an epoch, seed, feature or configuration.\n\n| Architecture | Configuration | TEST overall | TEST non-PUNCT | TEST seen | TEST unseen |\n|:--|:--|--:|--:|--:|--:|\n"
    for architecture in ARCHITECTURES:
        for config in "ABC":
            cells=[lookup[config,architecture,"test",c,p] for c,p in (("ALL","included"),("ALL","nonpunct"),("seen","included"),("unseen","included"))]
            report+=f"| {architecture} | {config} | "+" | ".join(f"{r['correct']}/{r['eligible_tokens']} ({100*float(r['accuracy']):.6f}%)" for r in cells)+" |\n"
    report+="\nAccuracy changes relative to full A (percentage points; descriptive, no inferential tests):\n\n"
    for architecture in ARCHITECTURES:
        a=float(lookup["A",architecture,"test","ALL","included"]["accuracy"])
        b=float(lookup["B",architecture,"test","ALL","included"]["accuracy"])
        c=float(lookup["C",architecture,"test","ALL","included"]["accuracy"])
        report+=f"- {architecture}: A→B {100*(b-a):+.6f} pp; B→C {100*(c-b):+.6f} pp; A→C {100*(c-a):+.6f} pp.\n"
    report+="\nA→B isolates removal of direct current/neighbor identities while affix information remains. B→C removes those current-token character affixes as well. The six affixes cover up to three Python Unicode characters and can encode whole forms for short tokens, so B must not be described as lexically blind. C still retains shape, Hangul/Latin/digit indicators, supplied current/neighbor Lang and predicted tag history; these carry indirect lexical/corpus regularities. Removing neighbor identities also removes their BOS/EOS values; no substitute boundary features were added. Changes in predicted history and learned weight support are expected downstream consequences of the ablation, not changes to the decoder.\n\nThese accuracy changes quantify dependence on feature groups on the fixed silver benchmark. Neither small nor large changes prove deterministic annotation-rule reproduction, semantic correctness, or human-valid generalization. Training-label regularity is in-sample; form singletons are automatically label-consistent. Whole-form unseen tests still allow shared affixes/context. Language-ID reliance is not isolated. Joint/routed descriptive comparisons are available, but supplied language and original silver scheme remain part of both conditions. No additional multilingual sequence tagger was trained.\n\nAll experiments are additional post-review descriptive diagnostics after inspection of existing human outcomes, not part of the frozen prospective v1.2 evaluation. Human-reference performance of new B/C models was not computed; that requires a separately documented exploratory analysis. Historical v1.1, frozen v1.2 models/evaluation, manuscript and supplementary archives are unchanged. `ablation_results.csv` contains exact numerators/denominators including seen/unseen non-PUNCT intersections for both DEV and TEST.\n"
    d.put("ablation_interpretation_report.md",report)
    coverage=[{"reviewer_concern":"A: annotation reliability","addressed":"v1.2 discrepancy contributions, silver NOUN share, length bands, traceable examples; existing independent IAA/adjudication audit","remaining":"Scheme-specific human correctness; sparse UPOS support; no newly tuned mapping rules","outputs":"annotation_discrepancy_report.md; silver_human_pos_confusions.csv; frozen_mapping_decomposition_report.md"},
        {"reviewer_concern":"B: near-perfect silver accuracy","addressed":"Authoritative unseen-token audit, empirical train label regularity, controlled A/B/C ablations in joint and routed models","remaining":"Does not prove annotation-rule reproduction; supplied language-ID contribution not isolated; human performance of new models unmeasured","outputs":"training_regularity_and_unseen_audit.md; ablation_results.csv; ablation_interpretation_report.md"},
        {"reviewer_concern":"C: small human reference","addressed":"Existing separate 180-sentence v1.2 extension; stratum and sparse-category diagnostics","remaining":"No population weighting or new uncertainty tests; rare UPOS labels have little support; historical data cannot be pooled","outputs":"../v12_integration/camera_ready_result_summary.md; annotation_discrepancy_report.md"},
        {"reviewer_concern":"D: scope and joint/routed distinction","addressed":"Same feature interventions in both original architectures; exact baseline reproduction and held-out silver subgroup counts","remaining":"No contemporary multilingual sequence model or mechanism-isolating routing/LID ablation; no inferential superiority claim","outputs":"ablation_results.csv; ablation_interpretation_report.md"},
        {"reviewer_concern":"E: reproducibility","addressed":"Frozen inputs/code/config hashes, exact A checkpoint/prediction identity, tests and replay validation, original-file preservation","remaining":"Original ChatGPT web silver generation, historical blinding and freeze timing are not independently reproducible/authenticated","outputs":"ablation_freeze_record.json; test_report.txt; reproducibility_manifest.json"}]
    d.put_csv("reviewer_concern_coverage.csv",coverage)
    d.put("reviewer_concern_coverage.md","# Reviewer-concern coverage\n\n"+"\n".join(f"- **{r['reviewer_concern']}** — addressed: {r['addressed']}. Remaining: {r['remaining']}. Sources: `{r['outputs']}`.\n" for r in coverage))
    d.put("README.md","# Additional diagnostics — researcher only\n\n"
        "All authoritative outputs below `outputs/camera_ready/additional_diagnostics/` are separate from the frozen v1.2 evaluation. Use the existing Python3.12 environment; runtime is stdlib-only. No dependency installation is needed.\n\n"
        "```powershell\npython -B scripts/camera_ready/additional_diagnostics.py --prepare --audit\npython -B scripts/camera_ready/run_lexical_ablation.py --freeze\npython -B scripts/camera_ready/run_lexical_ablation.py --config A --stage dev\npython -B scripts/camera_ready/run_lexical_ablation.py --config B --stage dev\npython -B scripts/camera_ready/run_lexical_ablation.py --config C --stage dev\npython -B scripts/camera_ready/run_lexical_ablation.py --config A --stage test\npython -B scripts/camera_ready/run_lexical_ablation.py --config B --stage test\npython -B scripts/camera_ready/run_lexical_ablation.py --config C --stage test\npython -B scripts/camera_ready/test_additional_diagnostics.py\npython -B scripts/camera_ready/additional_diagnostics.py --finalize\n```\n\n"
        "DEV and TEST are separate gated stages. On existing completed stages commands verify frozen hashes and skip writes; existing differing artifacts cause failure. A passes before B/C training, and A TEST identity passes before B/C TEST. B and C DEV may run concurrently in separate processes, then their TEST stages after A. To reproduce training afresh, use an isolated full repository copy containing original inputs and the frozen code, with an empty `additional_diagnostics` directory; never clear/overwrite these records or frozen source outputs in the working repository. The fixed freeze covers code, original modules/config and all three input hashes; test executions check prediction byte-repeatability from saved checkpoints. Full B/C retraining twice is not claimed; deterministic small-corpus training and full-A historical checkpoint reproduction are tested.\n\n"
        "Reports: `annotation_discrepancy_report.md`, `frozen_mapping_decomposition_report.md`, `training_regularity_and_unseen_audit.md`, `ablation_interpretation_report.md`, `reviewer_concern_coverage.md`. Data: POS/mapping CSVs, `ablation_results.csv`, `training_label_regularity.csv`. Models: `models/{A,B,C}/`; predictions: `predictions/{A,B,C}/`; counts: `metrics/{A,B,C}_{dev,test}.csv`; stage gates: `dev_validation/`, `test_validation/`; logs: `training_logs/`, `training_records/`. Frozen design: `ablation_configurations.json`, `ablation_freeze_record.json`, `frozen_source/`. Validation: `test_report.txt`, `test_validation_report.json`, `reproducibility_manifest.json`; original preservation: `preservation_manifest.json`; execution environment/commands: `environment.json`. All inputs, checkpoints and outputs have SHA-256 records.\n\n"
        "Do not distribute human/silver labels, trace examples, predictions or internal diagnostics to annotators. No new human-reference scoring, mapping fitting, inference, manuscript editing or supplementary packaging occurred. Historical v1.1 and prospective v1.2 remain separate. Exact replication of original web-assisted annotation and historical timing/blinding is not established.\n")
    source_records=[d.record(d.ROOT/"scripts/camera_ready"/s) for s in d.SCRIPTS]
    artifacts=[d.record(p) for p in sorted(d.OUT.rglob("*")) if p.is_file() and p.name!="reproducibility_manifest.json"]
    d.put_json("reproducibility_manifest.json",{"source_scripts":source_records,"artifacts":artifacts,
        "original_files_verified_unchanged":len(preservation["preexisting_files"]),"completed_all_six_stages":True,"test_report":test,
        "configuration_A_exact_historical_reproduction":True,"new_human_reference_evaluation":False})
    d.preserve()
    print("All additional diagnostics complete; authoritative originals preserved",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--config",choices=list("ABC"))
    parser.add_argument("--stage",choices=("dev","test"))
    args=parser.parse_args()
    if args.freeze:
        freeze()
    if args.config:
        d.require(args.stage is not None,"Provide --stage")
        stage(args.config,args.stage)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
