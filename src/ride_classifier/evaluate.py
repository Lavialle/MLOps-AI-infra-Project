"""Evaluation gate on the frozen test split: macro F1 above a floor AND above the majority baseline, or fail.

    python -m ride_classifier.evaluate            # evaluates the version produced by ride_classifier.train
    python -m ride_classifier.evaluate --no-promote

The model is loaded back FROM THE REGISTRY (what will be served), not reused from memory.
If the gate passes, the version gets the "staging" alias in the Model Registry.
Exit code 1 when the gate fails: the CI stops there.
"""

from __future__ import annotations

import argparse
import json
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from ride_classifier.config import RIDE_ID, ROOT, TARGET, load_params
from ride_classifier.model import INPUT_COLUMNS, confusion, majority_baseline, ride_scores

REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    params = load_params()
    train_report = json.loads((REPORTS / "train.json").read_text())
    name, version = train_report["model_name"], train_report["model_version"]

    test = pd.read_parquet(ROOT / params["data"]["test_path"])
    train_labels = pd.read_parquet(ROOT / params["data"]["train_path"], columns=[RIDE_ID, TARGET])
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    pred = model.predict(test[INPUT_COLUMNS])

    gate = params["gate"]
    result = {
        "model_name": name,
        "model_version": version,
        **ride_scores(test, pred, "test"),
        **majority_baseline(test, train_labels.drop_duplicates(RIDE_ID)[TARGET], "test"),
        **{f"gate_{k}": v for k, v in gate.items()},
    }
    passed = (
        result["test_macro_f1"] >= gate["min_macro_f1"] and result["test_macro_f1"] > result["test_baseline_macro_f1"]
    )
    result["passed"] = passed
    (REPORTS / "eval.json").write_text(json.dumps(result, indent=2))

    client = MlflowClient()
    client.set_model_version_tag(name, version, "gate", "passed" if passed else "failed")
    with mlflow.start_run(run_id=train_report["run_id"]):
        mlflow.log_metrics({k: v for k, v in result.items() if k.startswith("test_")})
        mlflow.log_dict(confusion(test, pred), "test_confusion_matrix.json")

    summary = (
        f"macro F1 {result['test_macro_f1']:.3f} (baseline {result['test_baseline_macro_f1']:.3f}), "
        f"balanced accuracy {result['test_balanced_accuracy']:.3f}"
    )
    if not passed:
        print(f"GATE FAILED: {summary} (min {gate['min_macro_f1']})")
        return 1

    if not args.no_promote:
        client.set_registered_model_alias(name, "staging", version)
    print(f"GATE PASSED: {summary} -> {name} v{version} @staging")
    return 0


if __name__ == "__main__":
    sys.exit(main())
