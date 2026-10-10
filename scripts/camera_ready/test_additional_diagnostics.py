"""Feature, determinism, mapping, partition and full artifact replay tests."""
from __future__ import annotations
import argparse
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

import additional_diagnostics as d
import run_lexical_ablation as a
from read_conllu import read_sentences, write_sentences


class RegressionTests(unittest.TestCase):
    def test_A_feature_identity(self):
        contexts=[("Cats","EN","__BOS__","walk","__BOS__","EN","__START__","__START__"),
                  ("한국어","KO","is",".","EN","OTHER","VERB","NOUN"),
                  ("a1!","MIXED","동작","__EOS__","KO","__EOS__","ADP","PUNCT"),
                  (".","OTHER","Hi",".","EN","OTHER","PROPN","VERB")]
        for context in contexts:
            self.assertEqual(a.features("A",*context),a.ORIGINAL_FEATURES(*context))
            with a.feature_configuration("A"):
                self.assertIs(d.original.get_features,a.ORIGINAL_FEATURES)

    def test_B_removes_only_three_identity_features(self):
        args=("Cats","EN","the","sleep","EN","EN","DET","__START__")
        full=a.ORIGINAL_FEATURES(*args)
        selected=a.features("B",*args)
        self.assertEqual(len(full),20)
        self.assertEqual(len(selected),17)
        self.assertEqual([f for f in full if f not in selected],["w=cats","p1=the","n1=sleep"])
        self.assertIn("p1.tag=DET",selected)
        self.assertIn("p1.tag+lang=DET+EN",selected)
        self.assertIn("pre3=cat",selected)
        self.assertIn("suf3=ats",selected)

    def test_C_retains_exact_original_nonlexical_features_and_order(self):
        selected=a.features("C","Cats","EN","the","sleep","EN","EN","DET","__START__")
        self.assertEqual(selected,["bias","lang=EN","shape=Xx","hgl=False","lat=True","dig=False", "p1.lang=EN","n1.lang=EN","p1.tag=DET","p2.tag=__START__","p1.tag+lang=DET+EN"])

    def test_original_training_and_small_seed_determinism_all_configs_routes(self):
        data=[[('cat','EN','NOUN'),('는','KO','ADP'),('.','OTHER','PUNCT')],
              [('big','EN','ADJ'),('집','KO','NOUN')],[('go','EN','VERB'),('집','KO','NOUN')]]
        # A vs untouched original, then exact two-run weights/predictions for
        # all three configurations and all four original update conditions.
        original=d.original.AveragedPerceptron()
        with contextlib.redirect_stdout(io.StringIO()):
            original.train(data,n_epochs=5,seed=42)
        for config in "ABC":
            for lang_filter in (None,"KO","EN","OTHER"):
                replicas=[]
                with a.feature_configuration(config),contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(2):
                        m=d.original.AveragedPerceptron()
                        m.train(data,n_epochs=5,seed=42,lang_filter=lang_filter)
                        replicas.append((m.tags,m.weights,m.predict([(f,l) for f,l,t in data[0]])))
                self.assertEqual(replicas[0],replicas[1])
                if config=="A" and lang_filter is None:
                    self.assertEqual(replicas[0][:2],(original.tags,original.weights))

    def test_feature_patch_always_restores_original(self):
        with self.assertRaises(RuntimeError):
            with a.feature_configuration("C"):
                raise RuntimeError("fixture")
        self.assertIs(d.original.get_features,a.ORIGINAL_FEATURES)

    def test_unseen_lowercase_partition_and_nonpunct(self):
        fields=lambda form,gold,pred,lang:["1",form,"_",gold,"_","_","0","root","_",f"Lang={lang}|PredUPOS={pred}"]
        tokens=[fields("CAT","NOUN","PUNCT","KO"),fields("cats","NOUN","NOUN","EN"),fields(".","PUNCT","PUNCT","OTHER")]
        sentences=[{"comments":[],"tokens":[{"is_real":True,"fields":f} for f in tokens]}]
        results=a.metric_rows("B","joint","fixture",sentences,{"cat","."})
        lookup={(r["condition"],r["punctuation"]):r for r in results}
        self.assertEqual((lookup["seen","included"]["eligible_tokens"],lookup["unseen","included"]["eligible_tokens"]),(2,1))
        self.assertEqual((lookup["seen","nonpunct"]["eligible_tokens"],lookup["seen","nonpunct"]["correct"]),(1,0))
        self.assertEqual(lookup["ALL","nonpunct"]["eligible_tokens"],2)
        self.assertNotIn("CAT",{"cat","."})

    def test_mapping_accounting_corrected_harmed_still_incorrect(self):
        effect=d.rule_effect(["ADP","PART","VERB","NOUN"],["PART","PART","X","NOUN"],["ADP","ADP","NOUN","NOUN"])
        self.assertEqual((effect["affected_predictions"],effect["incorrect_to_correct"],effect["correct_to_incorrect"],effect["affected_still_incorrect"],effect["net_correct_tokens"]),(3,1,1,1,0))

    def test_nonchained_frozen_maps_and_conservative_adjectives(self):
        self.assertEqual(d.apply_map("ADJ","KO",1,True),"ADJ")
        self.assertEqual(d.apply_map("ADJ","KO",2,True),"NOUN")
        self.assertEqual(d.apply_map("PART","EN",3,True),"PART")
        self.assertEqual(d.apply_map("PART","KO",3,True),"ADP")

    def test_prediction_source_mutation_detected(self):
        source={"comments":["# sent_id = fixture"],"tokens":[{"is_real":True,"fields":["1","cat","cat","NOUN","_","_","0","root","_","Lang=EN"]}]}
        import copy
        for index in (0,1,3):
            changed=copy.deepcopy(source)
            changed["tokens"][0]["fields"][index]="ALTERED"
            with self.assertRaises(ValueError):
                a.verify_prediction_source([source],[changed])

    def test_partition_hashes_and_document_split_preserved(self):
        expected=d.read_json(d.OUT/"partition_and_existing_unseen_audit.json")["partitions"]
        self.assertEqual(d.partition_inventory(),expected)
        self.assertEqual([expected[s]["real_tokens"] for s in ("train","dev","test")],[788802,98566,98529])

    def test_all_preexisting_files_unchanged(self):
        self.assertGreater(len(d.preserve()["preexisting_files"]),800)


class IntegrationTests(unittest.TestCase):
    def test_frozen_code_config_and_splits(self):
        a.verify_freeze()
        for name in d.SCRIPTS:
            self.assertEqual((d.OUT/"frozen_source"/name).read_bytes(),(d.ROOT/"scripts/camera_ready"/name).read_bytes())

    def test_exact_full_A_checkpoints_and_all_predictions(self):
        for name in a.MODEL_NAMES.values():
            self.assertEqual(d.sha(d.OUT/f"models/A/{name}.json"),d.sha(d.CLEAN/f"baselines/models/{name}.json"))
        for split in ("dev","test"):
            for architecture in a.ARCHITECTURES:
                self.assertEqual(d.sha(d.OUT/f"predictions/A/{architecture}_{split}_predictions.conllu"),d.sha(a.original_prediction_file(architecture,split)))
            a.verify_a_counts(d.rows(d.OUT/f"metrics/A_{split}.csv"),split)

    def test_all_B_C_checkpoint_features_match_removal(self):
        for config in "BC":
            removed=a.IDENTITY if config=="B" else a.IDENTITY+a.AFFIX
            for name in a.MODEL_NAMES.values():
                model=d.original.AveragedPerceptron.load(d.OUT/f"models/{config}/{name}.json")
                self.assertFalse(any(f.startswith(removed) for f in model.weights))
                self.assertEqual(model.tags,d.original.AveragedPerceptron.load(d.CLEAN/f"baselines/models/{name}.json").tags)

    def test_completed_stage_checksums_and_denominators(self):
        for split,total in (("dev",98566),("test",98529)):
            common={}
            for config in "ABC":
                validated=d.read_json(d.OUT/f"{split}_validation/{config}.json")
                for entry in validated["outputs"]:
                    self.assertEqual(d.sha(d.ROOT/entry["path"]),entry["sha256"])
                metrics=d.rows(d.OUT/f"metrics/{config}_{split}.csv")
                for architecture in a.ARCHITECTURES:
                    subset=[r for r in metrics if r["architecture"]==architecture]
                    all_row=next(r for r in subset if r["condition"]=="ALL" and r["punctuation"]=="included")
                    self.assertEqual(int(all_row["eligible_tokens"]),total)
                    for punct in ("included","nonpunct"):
                        group={r["condition"]:r for r in subset if r["punctuation"]==punct}
                        self.assertEqual(sum(int(group[c]["eligible_tokens"]) for c in ("seen","unseen")),int(group["ALL"]["eligible_tokens"]))
                        self.assertEqual(sum(int(group[c]["correct"]) for c in ("seen","unseen")),int(group["ALL"]["correct"]))
                        key=(architecture,punct)
                        counts=tuple(int(group[c]["eligible_tokens"]) for c in ("ALL","seen","unseen"))
                        self.assertEqual(counts,common.setdefault(key,counts))

    def test_actual_rule_effects_are_additive_with_zero_interaction(self):
        rules=d.rows(d.OUT/"frozen_mapping_rule_effects.csv")
        for scheme in d.rows(d.OUT/"frozen_mapping_scheme_effects.csv"):
            applicable=[r for r in rules if r["mapping"]==scheme["mapping"]]
            for field in ("affected_predictions","incorrect_to_correct","correct_to_incorrect","affected_still_incorrect","net_correct_tokens"):
                self.assertEqual(sum(int(r[field]) for r in applicable),int(scheme[field]))
            self.assertEqual(int(scheme["within_mapping_overlap_tokens"]),0)
            self.assertEqual(int(scheme["interaction_residual_correct_tokens"]),0)

    def test_full_prediction_replay_byte_identical_for_each_config_split_and_architecture(self):
        self.replayed=[]
        with tempfile.TemporaryDirectory(dir=d.OUT) as tmp:
            for config in "ABC":
                models={name:d.original.AveragedPerceptron.load(d.OUT/f"models/{config}/{filename}.json") for name,filename in a.MODEL_NAMES.items()}
                with a.feature_configuration(config):
                    predictors={"joint":a.PerceptronPredictor(models["joint"]),"routed":a.PipelinePanyoPredictor(models["KO"],models["EN"],models["OTHER"],"NOUN")}
                    for split in ("dev","test"):
                        source=read_sentences(d.SPLITS[split])
                        for architecture,predictor in predictors.items():
                            predicted=predictor.predict_sents(source)
                            a.verify_prediction_source(source,predicted)
                            path=Path(tmp)/f"{config}_{architecture}_{split}.conllu"
                            write_sentences(predicted,path)
                            existing=d.OUT/f"predictions/{config}/{architecture}_{split}_predictions.conllu"
                            self.assertEqual(d.sha(path),d.sha(existing))
                            self.replayed.append(d.record(existing))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--unit-only",action="store_true")
    args=parser.parse_args()
    log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RegressionTests)
    if not args.unit_only:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    stable=re.sub(r"Ran (\d+) tests in [\d.]+s",r"Ran \1 tests (elapsed time omitted from stable report)",log.getvalue())
    print(log.getvalue(),flush=True)
    if args.unit_only:
        d.put("unit_test_report.txt",stable)
    else:
        preserved=d.preserve()
        report={"passed":result.wasSuccessful(),"tests_run":result.testsRun,"full_prediction_streams_replayed_byte_identically":12 if result.wasSuccessful() else 0,
            "full_A_original_checkpoint_matches":4 if result.wasSuccessful() else 0,"full_A_original_prediction_matches":4 if result.wasSuccessful() else 0,
            "preexisting_files_verified_unchanged":len(preserved["preexisting_files"]),"small_corpus_training_repeatability":"all configs × all four update filters, two runs each",
            "full_B_C_training_repeated":False,"command":"python -B scripts/camera_ready/test_additional_diagnostics.py"}
        d.put_json("test_validation_report.json",report)
        d.put("test_report.txt",stable+"\n"+d.json_bytes(report).decode("utf-8"))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
