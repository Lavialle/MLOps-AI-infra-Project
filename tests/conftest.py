"""Test fixtures.

Tests never read the real dataset (DVC remote not reachable from CI, 30 MB): they use a
small fake file with the SAME schema (ride_id, time_index, ax..rz + RIDE_ID, LABEL)
and an easy class signal (each class shakes the accelerometer at a different amplitude).
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from ride_classifier.config import CHANNELS, CLASSES
from ride_classifier.data import load_raw
from ride_classifier.train import fit

RIDES_PER_CLASS = 6


def make_fake_rides(rides_per_class: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for k, label in enumerate(CLASSES):
        for i in range(rides_per_class):
            ride_id = f"ride-{k}-{i:03d}"
            n = int(rng.integers(300, 600))
            values = rng.normal(0, 0.1 + 0.5 * k, (n, len(CHANNELS)))
            values[:, 1] -= 9.4  # gravity on ay, as in the real data
            values[rng.choice(n, 3, replace=False)] = np.nan  # sensor dropouts: whole rows, ~1 % as in GBG500
            ride = pd.DataFrame(values, columns=CHANNELS)
            ride.insert(0, "time_index", np.arange(n) * 100)
            ride.insert(0, "ride_id", ride_id)
            rows.append(ride)
            labels.append({"RIDE_ID": ride_id, "LABEL": label})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(labels)


@pytest.fixture(scope="session")
def raw_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("raw")
    rides, labels = make_fake_rides(RIDES_PER_CLASS, seed=1)
    rides.to_parquet(folder / "rides.parquet", index=False)
    labels.to_csv(folder / "labels.csv", index=False)
    return folder / "rides.parquet", folder / "labels.csv"


@pytest.fixture(scope="session")
def small_data(raw_files) -> pd.DataFrame:
    return load_raw(*raw_files)


# Small and fast: fake rides are 300-600 samples long
FEAT_PARAMS = {"window_size": 128, "window_stride": 64}
TRAIN_PARAMS = {
    "epochs": 15,
    "batch_size": 32,
    "learning_rate": 0.003,
    "dropout": 0.1,
    "class_weighting": "balanced",
    "patience": 15,
    "random_state": 0,
}


@pytest.fixture(scope="session")
def trained(small_data):
    """(RideClassifier, best val scores) trained on fake rides, validated on other fake rides."""
    val_rides, val_labels = make_fake_rides(3, seed=2)
    val = val_rides.merge(val_labels.rename(columns={"RIDE_ID": "ride_id", "LABEL": "label"}), on="ride_id")
    return fit(small_data, val.fillna(0.0), FEAT_PARAMS, TRAIN_PARAMS)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, trained):
    """The trained wrapper saved as an MLflow pyfunc model, as train.py logs it (no server needed)."""
    model, _ = trained
    folder = tmp_path_factory.mktemp("model")
    weights = folder / "weights.pt"
    torch.save(model.net.state_dict(), weights)
    path = folder / "model"
    mlflow.pyfunc.save_model(str(path), python_model=model, artifacts={"weights": str(weights)})
    return path
