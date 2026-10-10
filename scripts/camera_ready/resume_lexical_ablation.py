"""Resume interrupted A DEV without overwriting completed frozen artifacts.

Execution-only recovery: original frozen feature/train/predict functions are
reused unchanged. Existing complete checkpoints must match historical A bytes.
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import additional_diagnostics as d
import run_lexical_ablation as a
from read_conllu import read_sentences,write_sentences


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a.verify_freeze()
    addendum=d.read_json(d.OUT/"ablation_execution_recovery_addendum.json")
    d.require(d.sha(Path(__file__))==addendum["recovery_code"]["sha256"],"Recovery code checksum differs")
    d.preserve()
    d.require(not (d.OUT/"dev_validation/A.json").exists(),"A already completed; recovery unnecessary")
    train=d.load_tagged_sents(d.SPLITS["train"])
    models,records={},[]
    with a.feature_configuration("A"):
        for name,original_name in a.MODEL_NAMES.items():
            path=d.OUT/f"models/A/{original_name}.json"
            prior=path.exists()
            elapsed=None
            if prior:
                d.require(d.sha(path)==d.sha(d.CLEAN/f"baselines/models/{original_name}.json"),"Interrupted saved checkpoint is not exact A")
                model=d.original.AveragedPerceptron.load(path)
                print(f"Verified and retained completed A/{name}",flush=True)
            else:
                print(f"Training missing A/{name}, original five epochs/seed42",flush=True)
                start=time.perf_counter()
                model=d.original.AveragedPerceptron()
                log=io.StringIO()
                with contextlib.redirect_stdout(log):
                    model.train(train,n_epochs=d.N_EPOCHS,seed=d.SEED,lang_filter=None if name=="joint" else name)
                    model.save(path)
                d.put(f"training_logs/A_{name}.txt",log.getvalue())
                elapsed=time.perf_counter()-start
            d.require(d.sha(path)==d.sha(d.CLEAN/f"baselines/models/{original_name}.json"),"Recovered A checkpoint not identical")
            models[name]=model
            records.append({"model":name,"training_seconds":elapsed,"completed_before_interruption":prior,
                "feature_count":len(model.weights),"epochs":d.N_EPOCHS,"seed":d.SEED,"steps":sum(map(len,train))*d.N_EPOCHS,
                "model_file":d.record(path),"exact_original_checkpoint_match":True,
                "interruption_note":"Earlier completed checkpoint retained; prior timing remains in session output" if prior else "Retrained only missing complete model from frozen TRAIN"})
            print(f"Completed A/{name}, {len(model.weights)} features",flush=True)
        d.put_json("training_records/A.json",records)
        source=read_sentences(d.SPLITS["dev"])
        vocab={f.lower() for sent in train for f,l,t in sent}
        predictors={"joint":a.PerceptronPredictor(models["joint"]),"routed":a.PipelinePanyoPredictor(models["KO"],models["EN"],models["OTHER"],"NOUN")}
        results,outputs=[],[]
        for architecture,predictor in predictors.items():
            predicted=predictor.predict_sents(source)
            a.verify_prediction_source(source,predicted)
            path=d.OUT/f"predictions/A/{architecture}_dev_predictions.conllu"
            d.require(not path.exists(),"Recovery refuses existing prediction overwrite")
            write_sentences(predicted,path)
            d.require(read_sentences(path)==predicted,"Recovery prediction reopen mismatch")
            d.require(d.sha(path)==d.sha(a.original_prediction_file(architecture,"dev")),"Recovered A DEV stream not byte identical")
            outputs.append(d.record(path))
            results+=a.metric_rows("A",architecture,"dev",predicted,vocab)
        a.verify_a_counts(results,"dev")
        d.put_csv("metrics/A_dev.csv",results)
        outputs.append(d.record(d.OUT/"metrics/A_dev.csv"))
        outputs += [d.record(d.OUT/f"models/A/{name}.json") for name in a.MODEL_NAMES.values()]
        a.verify_freeze()
        d.preserve()
        d.put_json("dev_validation/A.json",{"configuration":"A","split":"dev","original_source_unchanged":True,
            "source_token_identity_checked":True,"predictions_reopened_verified":True,"reference":"frozen silver only",
            "original_A_checkpoint_and_prediction_identity":True,"original_A_counts_match":True,
            "frozen_config_sha256":a.verify_freeze()["config"]["sha256"],"execution_recovery":d.record(d.OUT/"ablation_execution_recovery_addendum.json"),"outputs":outputs})
        print("PASS: all four A checkpoints and both DEV streams/counts exactly reproduced",flush=True)


if __name__=="__main__":
    main()
