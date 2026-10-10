"""Summarize existing unseen/POS/Lang tables; no prediction evaluation."""
import sys
from pathlib import Path
import additional_diagnostics as d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d.preserve()
    diagnostics=d.CLEAN/"baselines/diagnostics"
    sources={name:diagnostics/name for name in ("seen_unseen_accuracy.csv","seen_unseen_by_pos.csv","seen_unseen_by_lang.csv")}
    summary=d.rows(sources["seen_unseen_accuracy.csv"])
    detail=[]
    for dimension,filename,group in (("POS","seen_unseen_by_pos.csv","gold_upos"),("Lang","seen_unseen_by_lang.csv","lang")):
        for row in d.rows(sources[filename]):
            if row["model"] not in ("perceptron_joint","perceptron_pipeline_panyo"):
                continue
            aggregate=next(r for r in summary if all(r[k]==row[k] for k in ("model","split","condition")))
            detail.append({"dimension":dimension,"model":row["model"],"split":row["split"],"condition":row["condition"],
                "group":row[group],"tokens":row["total"],"correct":row["correct"],"errors":row["error_count"],
                "accuracy":int(row["correct"])/int(row["total"]),"condition_total":aggregate["total"],
                "share_of_condition_tokens":int(row["total"])/int(aggregate["total"])})
        for aggregate in summary:
            if aggregate["model"] in ("perceptron_joint","perceptron_pipeline_panyo"):
                subset=[r for r in detail if r["dimension"]==dimension and all(r[k]==aggregate[k] for k in ("model","split","condition"))]
                d.require(sum(int(r["tokens"]) for r in subset)==int(aggregate["total"]),"Archived unseen subgroup denominators differ")
                d.require(sum(int(r["correct"]) for r in subset)==int(aggregate["correct"]),"Archived unseen subgroup correct counts differ")
    d.put_csv("existing_unseen_composition.csv",detail)
    report="# Existing unseen-token composition audit\n\n"
    report+="This reads authoritative saved POS/language subgroup tables; no existing prediction evaluation is rerun. The tables use frozen silver UPOS and supplied Lang. Unseen remains FORM.lower() absence from Pangyo TRAIN, not novel morphemes, contexts, language patterns or human categories. Both architectures share the same eligibility partition.\n\n| Dimension | Group | Unseen TEST tokens | Share | Joint correct |\n|:--|:--|--:|--:|--:|\n"
    for row in detail:
        if row["model"]=="perceptron_joint" and row["split"]=="test" and row["condition"]=="unseen":
            report+=f"| {row['dimension']} | {row['group']} | {row['tokens']} | {100*row['share_of_condition_tokens']:.4f}% | {row['correct']} |\n"
    report+="\nThe 9,857 unseen TEST tokens have only six supported silver POS categories and are dominated by NOUN (8,066 tokens) and KO Lang (7,296). All 20 joint unseen errors fall in silver NOUN and EN; the routed model has 19 such errors. These are composition observations, not evidence that unseen tokens are semantically easy or human-valid. One unseen PUNCT token is excluded from the non-PUNCT subgroup; therefore unseen non-PUNCT denominator is 9,856. Both POS and language subgroup counts/correct counts sum exactly to archived totals for seen/unseen in DEV and TEST.\n\nSources: `outputs/final_clean_reproducibility_run/baselines/diagnostics/{seen_unseen_accuracy,seen_unseen_by_pos,seen_unseen_by_lang}.csv`. Full denominators/composition and source checksums are in `existing_unseen_composition.csv` and `existing_unseen_composition_provenance.json`.\n"
    d.put("existing_unseen_composition_audit.md",report)
    d.put_json("existing_unseen_composition_provenance.json",{"sources":[d.record(p) for p in sources.values()],"source_code":d.record(Path(__file__)),
        "archived_counts_reconciled":True,"no_new_predictions_evaluated":True,"command":"python -B scripts/camera_ready/audit_existing_unseen_composition.py"})
    d.preserve()
    print("Existing unseen subgroup counts reconciled to saved totals",flush=True)


if __name__=="__main__":
    main()
