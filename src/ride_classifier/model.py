"""1D-CNN on IMU windows + an MLflow pyfunc wrapper holding preprocessing AND model.

Anti-pattern avoided (same as the labs): preprocessing in one place, a model that
expects already-transformed inputs in another. The logged MLflow model takes RAW
IMU rows (ride_id, time_index, ax..rz) and returns one row per ride:

    ride_id | prediction | Safe | Safe overall | Bad weather | Tandem | Reckless
                      (mean window probabilities)

Gap filling, channel scaling (statistics fitted on train), windowing, the network
and the per-ride aggregation all travel inside the artifact, so the API cannot drift
from training.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn

from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, TARGET, TIME
from ride_classifier.data import fill_gaps
from ride_classifier.features import ChannelScaler, aggregate_by_ride, make_windows

INPUT_COLUMNS = [RIDE_ID, TIME, *CHANNELS]


def synthetic_ride(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A fake ride with the real schema (gravity on ay, small noise). Used as the MLflow input example,
    which is published with the model: it must never contain real course data."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, [0.3, 0.3, 0.3, 2.0, 5.0, 2.0], (n, len(CHANNELS))).round(4)
    values[:, 1] -= 9.4
    ride = pd.DataFrame(values, columns=CHANNELS)
    ride.insert(0, TIME, np.arange(n, dtype="int64") * 100)
    ride.insert(0, RIDE_ID, "synthetic-ride")
    return ride


def build_net(dropout: float = 0.3) -> nn.Module:
    """Small 1D-CNN: (batch, 6, window) -> (batch, 5) logits. Global average pooling makes it window-size agnostic."""

    def block(c_in: int, c_out: int, k: int) -> list[nn.Module]:
        return [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU()]

    return nn.Sequential(
        *block(len(CHANNELS), 32, 7),
        nn.MaxPool1d(2),
        *block(32, 64, 5),
        nn.MaxPool1d(2),
        *block(64, 128, 3),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, len(CLASSES)),
    )


@torch.no_grad()
def predict_windows(net: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Softmax probabilities (n_windows, 5)."""
    net.eval()
    device = next(net.parameters()).device
    out = [
        torch.softmax(net(torch.from_numpy(X[i : i + batch_size]).to(device)), dim=1).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(out).numpy()


class RideClassifier(mlflow.pyfunc.PythonModel):
    """Everything needed to go from raw IMU rows to a ride label. Weights are an MLflow artifact."""

    def __init__(self, scaler: dict, window_size: int, window_stride: int, dropout: float):
        self.scaler = scaler
        self.window_size = window_size
        self.window_stride = window_stride
        self.dropout = dropout
        self.net: nn.Module | None = None

    def __getstate__(self) -> dict:
        # The network is saved as a weights artifact (state_dict), not pickled with the wrapper
        return {**self.__dict__, "net": None}

    def load_context(self, context) -> None:
        self.net = build_net(self.dropout)
        self.net.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True))
        self.net.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        df = fill_gaps(model_input[INPUT_COLUMNS].drop(columns=TARGET, errors="ignore"))
        X, _, ride_ids = make_windows(df, ChannelScaler.from_dict(self.scaler), self.window_size, self.window_stride)
        proba = aggregate_by_ride(predict_windows(self.net, X), ride_ids)
        proba.insert(0, "prediction", [CLASSES[i] for i in proba.to_numpy().argmax(axis=1)])
        return proba.reset_index()


# --- Metrics (shared by train and evaluate) -----------------------------------------------


def ride_scores(truth: pd.DataFrame, pred: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Ride-level metrics. truth: ride_id + label, pred: ride_id + prediction (RideClassifier.predict output).

    Macro F1 is the headline metric: accuracy is meaningless here (always "Safe" = 83 %).
    """
    merged = truth.drop_duplicates(RIDE_ID)[[RIDE_ID, TARGET]].merge(pred[[RIDE_ID, "prediction"]], on=RIDE_ID)
    y, y_hat = merged[TARGET], merged["prediction"]
    scores = {
        f"{prefix}_macro_f1": f1_score(y, y_hat, labels=CLASSES, average="macro", zero_division=0),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y, y_hat),
        f"{prefix}_accuracy": accuracy_score(y, y_hat),
    }
    recalls = recall_score(y, y_hat, labels=CLASSES, average=None, zero_division=0)
    for label, r in zip(CLASSES, recalls, strict=True):
        scores[f"{prefix}_recall_{label.lower().replace(' ', '_')}"] = float(r)
    return {k: float(v) for k, v in scores.items()}


def majority_baseline(truth: pd.DataFrame, train_labels: pd.Series, prefix: str) -> dict[str, float]:
    """Scores of a model that always predicts the most frequent training class: the bar to beat."""
    rides = truth.drop_duplicates(RIDE_ID)
    majority = train_labels.mode().iloc[0]
    pred = pd.DataFrame({RIDE_ID: rides[RIDE_ID], "prediction": majority})
    return {f"{k.replace(prefix, prefix + '_baseline', 1)}": v for k, v in ride_scores(rides, pred, prefix).items()}


def confusion(truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    merged = truth.drop_duplicates(RIDE_ID)[[RIDE_ID, TARGET]].merge(pred[[RIDE_ID, "prediction"]], on=RIDE_ID)
    cm = confusion_matrix(merged[TARGET], merged["prediction"], labels=CLASSES)
    return {"labels": CLASSES, "rows_true_cols_pred": cm.tolist()}
