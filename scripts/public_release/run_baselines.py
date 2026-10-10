"""Explicit-path adapter for the original five baseline implementations.

Use fresh output directories. No changes to features, epochs, seed or decoding.
This is the silver benchmark, not human-reference evaluation.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_baselines as train
import predict_baselines as predict
import evaluate_baselines as evaluate


def fresh(path):
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory must be new or empty; never use frozen result directories")
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_training(train_path, gum_path, kaist_path, output):
    for p in (train_path, gum_path, kaist_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    train.PANYO_TRAIN, train.GUM_TRAIN, train.KAIST_TRAIN = train_path, gum_path, kaist_path
    train.MODELS, train.RESULTS = output / "models", output / "results"
    train._log_lines.clear()


def configure_prediction(dev, test, models, output):
    for p in (dev, test):
        if not p.is_file():
            raise FileNotFoundError(p)
    names = ("majority_global", "majority_lang", "perceptron_joint",
             "perceptron_pipeline_panyo_ko", "perceptron_pipeline_panyo_en",
             "perceptron_pipeline_panyo_other", "perceptron_pipeline_ud_en_gum",
             "perceptron_pipeline_ud_ko_kaist")
    for name in names:
        if not (models / (name + ".json")).is_file():
            raise FileNotFoundError("Missing checkpoint: " + name)
    predict.SPLITS = {"dev": dev, "test": test}
    predict.MODELS, predict.RESULTS = models, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    for name in ("train", "gum", "kaist", "output"):
        t.add_argument("--"+name, required=True, type=Path)
    r = sub.add_parser("predict")
    for name in ("dev", "test", "models", "output"):
        r.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    if a.action == "train":
        # Validate inputs before creating anything.
        for path in (a.train, a.gum, a.kaist):
            if not path.is_file():
                p.error("Missing input: " + str(path))
        configure_training(a.train, a.gum, a.kaist, fresh(a.output))
        train.main()
    else:
        # Checkpoint/input validation does not write files.
        configure_prediction(a.dev, a.test, a.models, a.output)
        predict.RESULTS = fresh(a.output)
        predict.main()
        evaluate.RESULTS = predict.RESULTS
        evaluate.main()


if __name__ == "__main__":
    main()
