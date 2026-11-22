"""Turn variable-length rides into fixed-size windows the model can consume.

Same rule as in the labs: the preprocessing used at training time must be the one
used at serving time. The `ChannelScaler` statistics are fitted on the TRAIN split
only and saved with the model (step 2), and the API calls these same functions.

A ride of n samples (~1,800-4,800 at 10 Hz) becomes k windows of shape
(channels=6, window_size). The model predicts each window and the ride-level
prediction is the mean of its window probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, TARGET, TIME


@dataclass
class ChannelScaler:
    """Per-channel standardisation (z-score), serialisable to JSON."""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> ChannelScaler:
        std = df[CHANNELS].std().replace(0, 1.0)  # a constant channel must not divide by zero
        return cls(mean=df[CHANNELS].mean().tolist(), std=std.tolist())

    def transform(self, values: np.ndarray) -> np.ndarray:
        """values: (n_samples, 6) -> standardised float32 array of the same shape."""
        return ((values - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> ChannelScaler:
        return cls(mean=list(d["mean"]), std=list(d["std"]))


def ride_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Cut one ride (n_samples, 6) into windows (k, 6, window_size).

    A ride shorter than one window is padded by repeating its last sample, so the API
    can still answer for a short recording.
    """
    n = len(values)
    if n == 0:
        raise ValueError("empty ride")
    if n < window_size:
        values = np.pad(values, ((0, window_size - n), (0, 0)), mode="edge")
        n = window_size
    starts = range(0, n - window_size + 1, stride)
    return np.stack([values[s : s + window_size].T for s in starts])


def make_windows(
    df: pd.DataFrame, scaler: ChannelScaler, window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows for every ride of a split.

    Returns X (n_windows, 6, window_size), y (n_windows,) class indices (-1 when the frame
    has no label, e.g. at serving time) and the ride id of each window (to aggregate per ride).
    """
    xs, ys, ids = [], [], []
    for ride_id, ride in df.sort_values([RIDE_ID, TIME], kind="stable").groupby(RIDE_ID, sort=False):
        windows = ride_windows(scaler.transform(ride[CHANNELS].to_numpy()), window_size, stride)
        label = CLASSES.index(ride[TARGET].iloc[0]) if TARGET in ride else -1
        xs.append(windows)
        ys.append(np.full(len(windows), label))
        ids.append(np.full(len(windows), ride_id, dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def aggregate_by_ride(window_proba: np.ndarray, window_ride_ids: np.ndarray) -> pd.DataFrame:
    """Mean of the window probabilities per ride -> one row per ride, one column per class."""
    proba = pd.DataFrame(window_proba, columns=CLASSES)
    proba[RIDE_ID] = window_ride_ids
    return proba.groupby(RIDE_ID, sort=False)[CLASSES].mean()
