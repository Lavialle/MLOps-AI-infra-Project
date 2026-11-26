"""Train the 1D-CNN, log everything to MLflow and register a new version.

    python -m ride_classifier.train

Promotion (the "staging" alias) is NOT done here: that is the job of the evaluation
gate (ride_classifier.evaluate), as in the CI/CD pipeline.

Class imbalance (292 Safe vs 4 Reckless training rides): the loss is weighted by
inverse class frequency, and the epoch kept is the one with the best ride-level
macro F1 on the validation split (early stopping).
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ride_classifier.config import CLASSES, MODEL_NAME, RIDE_ID, ROOT, TARGET, load_params
from ride_classifier.features import ChannelScaler, aggregate_by_ride, make_windows
from ride_classifier.model import (
    RideClassifier,
    build_net,
    confusion,
    majority_baseline,
    predict_windows,
    ride_scores,
    synthetic_ride,
)

REPORTS = ROOT / "reports"
EXPERIMENT = "ride-classifier"


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    """'balanced' = n_samples / (n_classes * count_c), as sklearn's class_weight='balanced'."""
    if mode != "balanced":
        return torch.ones(len(CLASSES))
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = np.where(counts > 0, len(y) / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def fit(
    train: pd.DataFrame, val: pd.DataFrame, p_feat: dict, p_train: dict, on_epoch=None, device: str = "cpu"
) -> tuple[RideClassifier, dict]:
    """Train on `train`, early-stop on ride-level macro F1 of `val`. Returns the wrapper (net loaded) and best scores.

    MLflow-free on purpose: the tests call it directly on fake data.
    """
    torch.manual_seed(p_train["random_state"])
    ws, stride = p_feat["window_size"], p_feat["window_stride"]

    scaler = ChannelScaler.fit(train)  # train only: no leakage from val/test
    X_tr, y_tr, _ = make_windows(train, scaler, ws, stride)
    X_val, _, val_ids = make_windows(val, scaler, ws, stride)

    net = build_net(p_train["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr, p_train["class_weighting"]).to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=p_train["learning_rate"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=p_train["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(p_train["random_state"]),
    )

    best = {"val_macro_f1": -1.0}
    best_state, waited = None, 0
    for epoch in range(1, p_train["epochs"] + 1):
        net.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        proba = aggregate_by_ride(predict_windows(net, X_val), val_ids)
        pred = proba.assign(prediction=[CLASSES[i] for i in proba.to_numpy().argmax(axis=1)]).reset_index()
        scores = {"train_loss": total / len(X_tr), **ride_scores(val, pred, "val")}
        if on_epoch:
            on_epoch(epoch, scores)

        if scores["val_macro_f1"] > best["val_macro_f1"]:
            best, best_state, waited = {**scores, "best_epoch": epoch}, copy.deepcopy(net.state_dict()), 0
        else:
            waited += 1
            if waited >= p_train["patience"]:
                break

    net.load_state_dict(best_state)
    model = RideClassifier(scaler.to_dict(), ws, stride, p_train["dropout"])
    model.net = net.cpu().eval()  # the served artifact is always CPU
    return model, best


def main() -> None:
    params = load_params()
    p_data, p_feat, p_train = params["data"], params["features"], params["train"]

    train_path, val_path = ROOT / p_data["train_path"], ROOT / p_data["val_path"]
    train, val = pd.read_parquet(train_path), pd.read_parquet(val_path)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        # The three things to version: code, data, model
        mlflow.set_tags(
            {"git_commit": git_commit(), "data_md5": file_md5(train_path), "data_path": p_data["train_path"]}
        )
        for section in ("data", "features", "train"):
            mlflow.log_params({f"{section}.{k}": v for k, v in params[section].items()})

        model, best = fit(train, val, p_feat, p_train, on_epoch=lambda e, s: mlflow.log_metrics(s, step=e))

        # Re-run the full wrapper (raw rows in, rides out) to log the real serving behaviour
        val_pred = model.predict(None, val)
        metrics = {
            **ride_scores(val, val_pred, "val"),
            **majority_baseline(val, train.drop_duplicates(RIDE_ID)[TARGET], "val"),
            "best_epoch": best["best_epoch"],
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict(confusion(val, val_pred), "val_confusion_matrix.json")
        mlflow.log_dict(model.scaler, "channel_scaler.json")

        # Synthetic example: the input example is published with the model (registry, Docker image), real rides are not
        sample = synthetic_ride()
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights.pt"
            torch.save(model.net.state_dict(), weights)
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=model,
                artifacts={"weights": str(weights)},
                signature=infer_signature(sample, model.predict(None, sample)),
                input_example=sample,
                registered_model_name=MODEL_NAME,
            )

        result = {
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "model_version": str(model_info.registered_model_version),
            **metrics,
        }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
