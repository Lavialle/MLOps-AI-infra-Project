"""Data ingestion for the GBG500 rides (IMU time series + one label per ride).

    python -m ride_classifier.data      # data/raw -> data/processed/{train,val,test}.parquet

The raw files are versioned with DVC (`dvc pull` downloads them from the S3 remote).

The split is made BY RIDE, never by row: the windows of one ride are strongly
correlated, so a ride split across train and test would leak. It is STRATIFIED
per class and keeps at least one ride of every class in each split. With only
6 "Reckless" rides, a plain random split would often leave val or test without any.

    train   ~70 %   fit the model and the channel scaler
    val     ~15 %   early stopping / model selection
    test    ~15 %   frozen set used by the evaluation gate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, ROOT, TARGET, TIME, load_params

SPLITS = ("train", "val", "test")


def load_raw(rides_path, labels_path, max_missing_share: float = 0.05) -> pd.DataFrame:
    """Read the IMU rows and the labels, validate, fill sensor gaps and join (one label column per row)."""
    rides = pd.read_parquet(rides_path)
    labels = pd.read_csv(labels_path).rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET})
    validate(rides, labels, max_missing_share)
    rides = fill_gaps(rides)
    return rides.merge(labels, on=RIDE_ID, how="inner")


def validate(rides: pd.DataFrame, labels: pd.DataFrame, max_missing_share: float = 0.05) -> None:
    """Fail fast on a broken input file instead of training on it."""
    missing = {RIDE_ID, TIME, *CHANNELS} - set(rides.columns)
    if missing:
        raise ValueError(f"rides: missing columns {sorted(missing)}")
    # GBG500: 0.7 % of the rows are sensor dropouts (all 6 channels NaN at once), <= 1.6 % per ride.
    # A few gaps are interpolated (fill_gaps); a ride that is mostly gaps is rejected.
    missing_share = rides[CHANNELS].isna().any(axis=1).groupby(rides[RIDE_ID]).mean()
    too_sparse = missing_share[missing_share > max_missing_share]
    if len(too_sparse):
        raise ValueError(f"rides: {len(too_sparse)} rides have more than {max_missing_share:.0%} missing samples")

    unknown = set(labels[TARGET]) - set(CLASSES)
    if unknown:
        raise ValueError(f"labels: unknown classes {sorted(unknown)}")
    if labels[RIDE_ID].duplicated().any():
        raise ValueError("labels: duplicated ride ids")

    ride_ids, label_ids = set(rides[RIDE_ID]), set(labels[RIDE_ID])
    if ride_ids != label_ids:
        raise ValueError(
            f"rides and labels do not match: {len(ride_ids - label_ids)} rides without label, "
            f"{len(label_ids - ride_ids)} labels without ride"
        )


def fill_gaps(rides: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation of missing samples inside each ride (edges: nearest valid sample)."""
    rides = rides.sort_values([RIDE_ID, TIME], kind="stable").reset_index(drop=True)
    rides[CHANNELS] = rides.groupby(RIDE_ID)[CHANNELS].transform(lambda s: s.interpolate(limit_direction="both"))
    return rides


def stratified_ride_split(
    labels: pd.DataFrame, val_share: float, test_share: float, random_state: int
) -> dict[str, list[str]]:
    """Split ride ids per class, with at least one ride of each class in val and test."""
    rng = np.random.default_rng(random_state)
    out: dict[str, list[str]] = {name: [] for name in SPLITS}
    for label, group in labels.groupby(TARGET):
        ids = rng.permutation(group[RIDE_ID].to_numpy())
        if len(ids) < 3:
            raise ValueError(f"class {label!r} has {len(ids)} rides: need at least 3 (train, val, test)")
        n_test = max(1, round(len(ids) * test_share))
        n_val = max(1, round(len(ids) * val_share))
        out["test"] += ids[:n_test].tolist()
        out["val"] += ids[n_test : n_test + n_val].tolist()
        out["train"] += ids[n_test + n_val :].tolist()
    return out


def split(df: pd.DataFrame, val_share: float, test_share: float, random_state: int) -> dict[str, pd.DataFrame]:
    labels = df[[RIDE_ID, TARGET]].drop_duplicates()
    ids = stratified_ride_split(labels, val_share, test_share, random_state)
    return {name: df[df[RIDE_ID].isin(ids[name])].reset_index(drop=True) for name in SPLITS}


def main() -> None:
    p = load_params()["data"]
    df = load_raw(ROOT / p["raw_rides_path"], ROOT / p["raw_labels_path"], p["max_missing_share"])
    parts = split(df, p["val_share"], p["test_share"], p["random_state"])
    for name, part in parts.items():
        path = ROOT / p[f"{name}_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        counts = part.drop_duplicates(RIDE_ID)[TARGET].value_counts().reindex(CLASSES, fill_value=0)
        counts = {label: int(n) for label, n in counts.items()}
        print(f"{name:5s} {part[RIDE_ID].nunique():4d} rides {len(part):8d} rows  {counts} -> {p[f'{name}_path']}")


if __name__ == "__main__":
    main()
