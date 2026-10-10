"""Read-only audits and descriptive diagnostics separate from frozen v1.2.

Only writes under additional_diagnostics; existing outputs must agree byte for
byte on repetition. No human-reference evaluation of new models is permitted.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs/camera_ready/additional_diagnostics"
V12 = ROOT/"outputs/camera_ready/v12_integration"
CLEAN = ROOT/"outputs/final_clean_reproducibility_run"
SPLITS = {s: ROOT/f"data/processed/panyo_original_adp_docsplit_{s}.conllu" for s in ("train","dev","test")}
SCRIPTS = ("additional_diagnostics.py","run_lexical_ablation.py","test_additional_diagnostics.py")
sys.path.insert(0,str(ROOT/"scripts"))
import perceptron_tagger as original
from train_baselines import load_tagged_sents, N_EPOCHS, SEED
from read_conllu import iter_sentences, get_misc_lang
from diagnostic_ud_mapping import apply_map, BASE_MAP_1, BASE_MAP_2


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def put(name,data):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(data,str):
        data=data.encode("utf-8")
    if path.exists():
        require(path.read_bytes()==data,f"Refusing to overwrite a differing existing artifact: {path}")
    else:
        path.write_bytes(data)


def put_json(name,value):
    put(name,json_bytes(value))


def put_csv(name,data,fields=None):
    fields=fields or list(data[0])
    f=io.StringIO(newline="")
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
    writer.writeheader()
    writer.writerows(data)
    put(name,f.getvalue())


def record(path):
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path)}


def preserve():
    baseline=read_json(OUT/"preservation_manifest.json")
    changed=[name for name,h in baseline["preexisting_files"].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    require(not changed,f"Frozen/preexisting files changed or missing: {changed}")
    return baseline


def prepare():
    if (OUT/"preservation_manifest.json").exists():
        preserve()
        return
    files={}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative=path.relative_to(ROOT)
        if any(x in relative.parts for x in (".git","__pycache__",".pytest_cache")) or path.is_relative_to(OUT):
            continue
        if relative.as_posix() in {"scripts/camera_ready/"+s for s in SCRIPTS}:
            continue
        files[relative.as_posix()]=sha(path)
    put_json("preservation_manifest.json",{"git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "initial_git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "preexisting_files":files,"exclusions":"Only .git/caches and this task's three new scripts/output directory"})
    # Verify the completed v1.2 outputs against their successful repeatability record.
    repeat=read_json(V12/"repeatability_validation.json")
    require(not repeat["mismatches"] and repeat["unit_regression_tests_passed"],"v1.2 completion verification failed")
    for name,h in repeat["output_sha256"].items():
        require(sha(V12/name)==h,f"Completed v1.2 artifact differs: {name}")
    env=read_json(CLEAN/"environment/environment.json")
    for name,h in env["reused_script_checksums_sha256"].items():
        require(sha(ROOT/"scripts"/name)==h,f"Original implementation changed: {name}")
    put_json("environment.json",{"python":sys.version,"platform":platform.platform(),"original_epochs":N_EPOCHS,"original_seed":SEED,
        "stdlib_execution":True,"existing_environment":record(CLEAN/"environment/environment.json"),
        "commands":["python -B scripts/camera_ready/additional_diagnostics.py --prepare --audit",
                    "python -B scripts/camera_ready/run_lexical_ablation.py --freeze",
                    *[f"python -B scripts/camera_ready/run_lexical_ablation.py --config {c} --stage dev" for c in "ABC"],
                    *[f"python -B scripts/camera_ready/run_lexical_ablation.py --config {c} --stage test" for c in "ABC"],
                    "python -B scripts/camera_ready/test_additional_diagnostics.py",
                    "python -B scripts/camera_ready/additional_diagnostics.py --finalize"],
        "timing_status":"Additional post-review descriptive diagnostics, not preregistered under the v1.2 protocol; frozen human results already inspected."})
    print(f"Preserved {len(files)} preexisting file hashes",flush=True)


def sentence_meta(sentence):
    return {c[2:].split(" = ",1)[0]:c[2:].split(" = ",1)[1] for c in sentence["comments"] if " = " in c}


def partition_inventory():
    result={}
    seen_docs,seen_sids=set(),set()
    for split,path in SPLITS.items():
        sids,docs,tokens=set(),set(),0
        for sent in iter_sentences(path):
            meta=sentence_meta(sent)
            sid,doc=meta["sent_id"],meta["doc_id"]
            require(meta["split"]==split and sid not in sids,"Invalid/duplicate frozen split identity")
            sids.add(sid)
            docs.add(doc)
            tokens+=sum(t["is_real"] for t in sent["tokens"])
        require(not (sids&seen_sids or docs&seen_docs),"Train/dev/test sentence or document leakage")
        seen_sids|=sids
        seen_docs|=docs
        result[split]={**record(path),"sentences":len(sids),"documents":len(docs),"real_tokens":tokens}
    return result


def training_audit():
    tagged=load_tagged_sents(SPLITS["train"])
    by_form=collections.defaultdict(collections.Counter)
    by_pair=collections.defaultdict(collections.Counter)
    langs=collections.defaultdict(collections.Counter)
    for sent in tagged:
        for form,lang,tag in sent:
            by_form[form.lower()][tag]+=1
            by_pair[form.lower(),lang][tag]+=1
            langs[lang][tag]+=1
    detail=[]
    for form,cnt in sorted(by_form.items()):
        n=sum(cnt.values())
        detail.append({"lowercase_form":form,"train_tokens":n,"distinct_silver_labels":len(cnt),
            "majority_count":max(cnt.values()),"majority_tags":"|".join(sorted(t for t,v in cnt.items() if v==max(cnt.values()))),
            "label_counts_json":json.dumps(dict(sorted(cnt.items())),ensure_ascii=False,sort_keys=True),
            "conditional_entropy_bits":-sum(v/n*math.log2(v/n) for v in cnt.values())})
    n=sum(sum(c.values()) for c in by_form.values())
    summary={"training_tokens":n,"lowercase_types":len(by_form),"single_label_types":sum(len(c)==1 for c in by_form.values()),
        "tokens_of_single_label_types":sum(sum(c.values()) for c in by_form.values() if len(c)==1),
        "singleton_types":sum(sum(c.values())==1 for c in by_form.values()),
        "empirical_form_majority_correct":sum(max(c.values()) for c in by_form.values()),
        "empirical_form_lang_majority_correct":sum(max(c.values()) for c in by_pair.values()),
        "form_conditional_entropy_bits":sum(r["train_tokens"]*r["conditional_entropy_bits"] for r in detail)/n,
        "label_inventory":sorted({t for c in by_form.values() for t in c}),
        "definition":"FORM.lower(), Python Unicode lowercase; no NFKC, stripping, lemma, affix matching or test vocabulary fitting"}
    put_csv("training_label_regularity.csv",detail)
    put_json("training_label_regularity_summary.json",summary)
    put_csv("training_labels_by_language.csv",[{"language":lang,"silver_upos":tag,"train_tokens":count} for lang,c in sorted(langs.items()) for tag,count in sorted(c.items())])
    # Existing, authoritative denominators/counts are reused, not rerun as experiments.
    existing=CLEAN/"baselines/diagnostics/seen_unseen_accuracy.csv"
    old=[r for r in rows(existing) if r["model"] in ("perceptron_joint","perceptron_pipeline_panyo")]
    partitions=partition_inventory()
    for split in ("dev","test"):
        vocab=set(by_form)
        observed=collections.Counter("seen" if f.lower() in vocab else "unseen" for s in load_tagged_sents(SPLITS[split]) for f,l,t in s)
        for r in old:
            if r["split"]==split:
                require(int(r["total"])==observed[r["condition"]],"Existing unseen vocabulary partition differs")
    put_csv("existing_unseen_authoritative_counts.csv",old)
    put_json("partition_and_existing_unseen_audit.json",{"partitions":partitions,"seen_unseen_source":record(existing),
        "definition":summary["definition"],"seen_definition":"lowercase FORM present anywhere in frozen Pangyo TRAIN; language-independent, also for routed models",
        "vocabulary_size":len(by_form),"archived_source_script":record(ROOT/"scripts/diagnostic_seen_unseen.py"),
        "limitations":["The original diagnostic builds form+language and form+language+gold sets but scores only lowercased form membership.",
          "Its unseen-error example list has one global 500-row cap across models/splits, despite a comment saying per model/split; examples are incomplete and are not used here.",
          "Training consistency is in-sample silver-label regularity, not evidence of correct human labels or a reproducible annotation rule.",
          "Unseen whole forms can still share prefixes/suffixes, shape, language, contexts and tags; near-perfect unseen silver accuracy does not establish human-valid generalization."]})
    report="# Training-label regularity and existing unseen-token audit\n\n"
    report+=f"Frozen TRAIN: {n:,} original tokens in {len(tagged):,} sentences; {len(by_form):,} lowercase form types. {summary['single_label_types']:,} types have one observed silver label, covering {summary['tokens_of_single_label_types']:,} training tokens. {summary['singleton_types']:,} types occur once. Empirical per-form majority matches {summary['empirical_form_majority_correct']:,}/{n:,}; per-form+language majority matches {summary['empirical_form_lang_majority_correct']:,}/{n:,}. Frequency-weighted label entropy given form is {summary['form_conditional_entropy_bits']:.8f} bits. These are training-only descriptive regularities, not held-out baseline scores. Singleton consistency is tautological.\n\n"
    report+="| Model | Split | Condition | Correct / tokens | Accuracy |\n|:--|:--|:--|--:|--:|\n"
    for r in old:
        report+=f"| {r['model']} | {r['split']} | {r['condition']} | {r['correct']}/{r['total']} | {100*int(r['correct'])/int(r['total']):.6f}% |\n"
    report+="\nSeen means exact Python-lowercased FORM membership in frozen Pangyo TRAIN, independent of Lang; original exported tokens remain unchanged. Unseen is its complement. No dev/test word is added to that vocabulary. Original document/sentence partitions are disjoint and their hashes are recorded in `partition_and_existing_unseen_audit.json`. Training feature extraction itself lowercases current and neighbor identities and uses up to three-character affixes.\n\nExisting source: `outputs/final_clean_reproducibility_run/baselines/diagnostics/seen_unseen_accuracy.csv`; definitions: `scripts/diagnostic_seen_unseen.py`. No authoritative unseen experiment needs rerunning. Its global 500-example cap is narrower than its comment implies; archived example lists are not exhaustive. Counts above are authoritative, and lowercase partition denominators have been independently verified. Full-feature A will independently check the preserved predictions and exact counts.\n\nDirect whole-word identity, affixes and shape are distinct information channels. High training form-label consistency and high unseen silver accuracy are compatible with corpus/annotation regularities, but do not establish deterministic annotation-rule reproduction or agreement with human UPOS. Routed models retain supplied language identifiers; their contribution is not isolated by these requested ablations.\n"
    put("training_regularity_and_unseen_audit.md",report)
    print("Training regularity and existing unseen audit complete",flush=True)


def rule_effect(gold,raw,mapped):
    require(len(gold)==len(raw)==len(mapped),"Mapping token counts differ")
    affected=[i for i in range(len(gold)) if raw[i]!=mapped[i]]
    corrected=sum(raw[i]!=gold[i] and mapped[i]==gold[i] for i in affected)
    harmed=sum(raw[i]==gold[i] and mapped[i]!=gold[i] for i in affected)
    still=sum(raw[i]!=gold[i] and mapped[i]!=gold[i] for i in affected)
    require(len(affected)==corrected+harmed+still,"Mapping categories not exhaustive")
    return {"affected_predictions":len(affected),"incorrect_to_correct":corrected,"correct_to_incorrect":harmed,
        "affected_still_incorrect":still,"net_correct_tokens":corrected-harmed,
        "full_reference_denominator":len(gold),"net_accuracy_change":(corrected-harmed)/len(gold) if gold else 0.}


def annotation_and_mappings():
    aligned=rows(V12/"aligned_v12_predictions.csv")
    require(len(aligned)==3186 and all(r["reference_status"]=="RESOLVED" for r in aligned),"Unexpected frozen v1.2 reference")
    total=len(aligned)
    disagreements=[r for r in aligned if r["canonical_silver"]!=r["human_upos_v12"]]
    nerr=len(disagreements)
    matrix=collections.Counter((r["human_upos_v12"],r["canonical_silver"]) for r in aligned)
    preserved=rows(V12/"v12_confusion_matrices.csv")
    for r in preserved:
        if r["system"]=="canonical_silver":
            require(matrix[r["human_upos"],r["predicted_upos"]]==int(r["tokens"]),"Frozen confusion counts differ")
    confusion=[{"human_upos":h,"silver_upos":s,"tokens":v,"disagreement":h!=s,
        "share_of_all_tokens":v/total,"share_of_disagreements":v/nerr if h!=s else 0.} for (h,s),v in sorted(matrix.items())]
    put_csv("silver_human_pos_confusions.csv",confusion)
    by_pos=[]
    for side,field in (("human","human_upos_v12"),("silver","canonical_silver")):
        sup=collections.Counter(r[field] for r in aligned)
        errs=collections.Counter(r[field] for r in disagreements)
        for tag in sorted(set(sup)):
            by_pos.append({"side":side,"upos":tag,"tokens":sup[tag],"disagreements":errs[tag],
                "error_rate_within_category":errs[tag]/sup[tag],"share_of_total_disagreements":errs[tag]/nerr})
    put_csv("silver_human_disagreement_by_pos.csv",by_pos)
    stratum=[]
    for s in map(str,range(1,6)):
        eligible=[r for r in aligned if r["stratum"]==s]
        errors=[r for r in disagreements if r["stratum"]==s]
        stratum.append({"stratum":s,"tokens":len(eligible),"disagreements":len(errors),"disagreement_rate":len(errors)/len(eligible),
            "share_of_total_disagreements":len(errors)/nerr,"silver_noun_disagreements":sum(r["canonical_silver"]=="NOUN" for r in errors)})
    put_csv("silver_human_disagreement_by_stratum.csv",stratum)
    refs={(r["sample_id"],r["token_ordinal"]):r for r in rows(V12/"inputs/PANGYO_V12_FINAL_HUMAN_REFERENCE_3186_TOKENS.csv")}
    traces=[]
    for r in disagreements:
        ref=refs[r["sample_id"],r["token_ordinal"]]
        traces.append({**{f:r[f] for f in ("batch","stratum","sample_id","source_document_id","source_sentence_id","sentence","token_ordinal","original_token_id","token")},
            "human_upos_v12":r["human_upos_v12"],"canonical_silver":r["canonical_silver"],
            "human_reference_provenance":ref["upos_provenance"],"human_reference_case_id":ref["upos_case_id"]})
    put_csv("silver_human_disagreement_trace.csv",traces)
    # Deterministic examples, not selected for apparent severity or rule fit.
    largest=sorted((r for r in confusion if r["disagreement"]),key=lambda r:(-r["tokens"],r["human_upos"],r["silver_upos"]))[:10]
    examples=[]
    for category in largest:
        candidates=[r for r in traces if (r["human_upos_v12"],r["canonical_silver"])==(category["human_upos"],category["silver_upos"])]
        examples+=sorted(candidates,key=lambda r:(r["sample_id"],int(r["token_ordinal"])))[:3]
    put_csv("traceable_disagreement_examples.csv",examples)
    noun=sum(r["canonical_silver"]=="NOUN" for r in disagreements)
    summary={"reference_tokens":total,"disagreements":nerr,"silver_noun_disagreements":noun,
        "silver_noun_share_of_disagreements":noun/nerr,"sources":[record(V12/"aligned_v12_predictions.csv"),record(V12/"v12_confusion_matrices.csv")],
        "example_rule":"first three sample_id/token_ordinal rows in each of ten largest confusion categories; retrospective descriptive examples"}
    put_json("annotation_discrepancy_summary.json",summary)
    report=f"# v1.2 annotation discrepancy report\n\nReuses the completed frozen evaluation, not a new benchmark. {nerr}/{total} tokens disagree; {noun}/{nerr} ({100*noun/nerr:.4f}%) disagreements involve silver NOUN (human is not NOUN). Direction is human → silver throughout. The human reference is an adjudicated scheme-specific reference; disagreement is not independent proof of linguistic error. Historical v1.1 is untouched and separate.\n\n"
    report+="| Human → silver | Count | Share of disagreements |\n|:--|--:|--:|\n"
    for r in largest:
        report+=f"| {r['human_upos']} → {r['silver_upos']} | {r['tokens']} | {100*r['share_of_disagreements']:.4f}% |\n"
    report+="\n| Length stratum | Disagreements / tokens | Rate | Share of disagreements |\n|:--|--:|--:|--:|\n"
    for r in stratum:
        report+=f"| {r['stratum']} | {r['disagreements']}/{r['tokens']} | {100*r['disagreement_rate']:.4f}% | {100*r['share_of_total_disagreements']:.4f}% |\n"
    report+="\nTraceable examples (one displayed per largest category; up to three retained per category):\n\n"
    for category in largest:
        r=next(r for r in examples if (r["human_upos_v12"],r["canonical_silver"])==(category["human_upos"],category["silver_upos"]))
        report+=f"- `{r['sample_id']}`, `{r['source_sentence_id']}`, ordinal {r['token_ordinal']}, original ID {r['original_token_id']}: **{r['token']}**; human {r['human_upos_v12']} → silver {r['canonical_silver']}. Original context: {r['sentence']}\n"
    report+="\nFull category denominators are in `silver_human_disagreement_by_pos.csv`; confusion contributions in `silver_human_pos_confusions.csv`; all source rows/decision provenance in `silver_human_disagreement_trace.csv`. Examples were selected by a recorded stable ordering, not fitted into new mapping rules. Equal sentence strata do not provide population weighting.\n"
    put("annotation_discrepancy_report.md",report)
    # Original supplied Lang is used only to certify the existing scheme3 guards.
    wanted={r["source_sentence_id"] for r in aligned}
    language={}
    for sent in iter_sentences(SPLITS["test"]):
        sid=sentence_meta(sent)["sent_id"]
        if sid in wanted:
            for ordinal,t in enumerate((t for t in sent["tokens"] if t["is_real"]),1):
                language[sid,str(ordinal)]=get_misc_lang(t["fields"][9])
    gold=[r["human_upos_v12"] for r in aligned]
    raw=[r["ud_transfer_no_mapping"] for r in aligned]
    all_effects,scheme_effects,rule_traces=[],[],[]
    maps={"conservative":1,"panyo_coarse":2,"ko_oriented":3}
    for name,scheme in maps.items():
        mapped=[r["ud_transfer_"+name] for r in aligned]
        groups=collections.defaultdict(list)
        if scheme in (1,2):
            rules={("ALL",s,t) for s,t in (BASE_MAP_1 if scheme==1 else BASE_MAP_2).items()}
        else:
            rules={("KO",s,t) for s,t in BASE_MAP_2.items()}|{("EN",s,t) for s,t in {"SYM":"PUNCT","X":"NOUN","INTJ":"NOUN","ADJ":"NOUN"}.items()}|{("OTHER",s,t) for s,t in {"SYM":"PUNCT","X":"NOUN"}.items()}
        for i,r in enumerate(aligned):
            lang=language[r["source_sentence_id"],r["token_ordinal"]]
            require(mapped[i]==apply_map(raw[i],lang,scheme,True),"Frozen mapped output/rule mismatch")
            matches=[rule for rule in rules if rule[0] in ("ALL",lang) and rule[1]==raw[i]]
            require(len(matches)<=1,"Overlapping mapping rules; interaction accounting needed")
            if matches:
                rule=matches[0]
                groups[rule].append(i)
                rule_traces.append({"mapping":name,"guard":rule[0],"from":rule[1],"to":rule[2],
                    **{f:r[f] for f in ("sample_id","source_sentence_id","token_ordinal","original_token_id","token")},
                    "human_upos_v12":gold[i],"raw_prediction":raw[i],"mapped_prediction":mapped[i],
                    "raw_correct":raw[i]==gold[i],"mapped_correct":mapped[i]==gold[i]})
        for guard,source,target in sorted(rules):
            idx=groups[guard,source,target]
            effect=rule_effect([gold[i] for i in idx],[raw[i] for i in idx],[mapped[i] for i in idx])
            effect["full_reference_denominator"]=total
            effect["net_accuracy_change"]=effect["net_correct_tokens"]/total
            all_effects.append({"mapping":name,"language_guard":guard,"source_tag":source,"target_tag":target,**effect})
        aggregate=rule_effect(gold,raw,mapped)
        relevant=[r for r in all_effects if r["mapping"]==name]
        for field in ("affected_predictions","incorrect_to_correct","correct_to_incorrect","affected_still_incorrect","net_correct_tokens"):
            require(sum(r[field] for r in relevant)==aggregate[field],"Rule effects do not add to mapping total")
        require(aggregate["net_correct_tokens"]==sum(p==g for p,g in zip(mapped,gold))-sum(p==g for p,g in zip(raw,gold)),"Incorrect net mapping delta")
        scheme_effects.append({"mapping":name,**aggregate,"raw_correct":sum(p==g for p,g in zip(raw,gold)),
            "mapped_correct":sum(p==g for p,g in zip(mapped,gold)),"unaffected_incorrect":sum(p!=g and p==m for g,p,m in zip(gold,raw,mapped)),
            "within_mapping_overlap_tokens":0,"interaction_residual_correct_tokens":0})
    put_csv("frozen_mapping_rule_effects.csv",all_effects)
    put_csv("frozen_mapping_scheme_effects.csv",scheme_effects)
    put_csv("frozen_mapping_affected_token_trace.csv",rule_traces)
    report="# Frozen mapping-rule decomposition (v1.2 descriptive)\n\n"
    report+="Effects compare each preserved mapped UD stream to the same preserved raw UD stream on all 3,186 tokens. No rule was fitted, added, optimized or selected using these outcomes. Scheme3 uses the already frozen EN_ADJ flag=true from corrected DEV selection. Conservative leaves ADJ unchanged. `affected_still_incorrect` means the predicted label changed but both raw and mapped labels are incorrect; it does not mean an unchanged label. `unaffected_incorrect` counts incorrect labels not changed by any rule.\n\n| Mapping | Affected | Corrected | Harmed | Still incorrect | Net correct |\n|:--|--:|--:|--:|--:|--:|\n"
    for r in scheme_effects:
        report+=f"| {r['mapping']} | {r['affected_predictions']} | {r['incorrect_to_correct']} | {r['correct_to_incorrect']} | {r['affected_still_incorrect']} | {r['net_correct_tokens']} |\n"
    report+="\nWithin each scheme, rules are guarded replacements applied once to the original prediction. Source tags and language guards are mutually exclusive. Verified overlap=0; each token changes under at most one rule. Thus rule effects add exactly to the whole scheme's net correct-token change and there is no sequential interaction term. This statement concerns these fixed streams and implementation, not a general interaction claim. Across schemes, shared tokens/rules are dependent and their effects must not be summed as independent evidence. The different scheme3 guards matter (e.g. EN PART remains PART); rule CSVs retain them. All rule and aggregate changes use denominator 3,186.\n\n`frozen_mapping_rule_effects.csv` includes zero-affected applicable rules; `frozen_mapping_scheme_effects.csv` provides net and unaffected counts; `frozen_mapping_affected_token_trace.csv` provides exact source tokens. These are post-outcome descriptive comparisons, without significance tests or model selection.\n"
    put("frozen_mapping_decomposition_report.md",report)
    print("v1.2 discrepancy and frozen mapping decomposition complete",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.audit:
        preserve()
        training_audit()
        annotation_and_mappings()
        preserve()
    if args.finalize:
        import run_lexical_ablation
        run_lexical_ablation.finalize()


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
