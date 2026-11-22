import numpy as np
import pandas as pd
import pytest
from conftest import RIDES_PER_CLASS, make_fake_rides

from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, TARGET
from ride_classifier.data import fill_gaps, split, stratified_ride_split, validate
from ride_classifier.features import ChannelScaler, aggregate_by_ride, make_windows, ride_windows


# --- validation ------------------------------------------------------------------
def test_load_raw_joins_one_label_per_row(small_data):
    assert small_data[TARGET].notna().all()
    assert small_data[RIDE_ID].nunique() == RIDES_PER_CLASS * len(CLASSES)


def test_load_raw_fills_sensor_dropouts(small_data):
    assert not small_data[CHANNELS].isna().any().any()


def test_fill_gaps_interpolates_inside_a_ride_only():
    rides = pd.DataFrame(
        {
            RIDE_ID: ["a", "a", "a", "b", "b"],
            "time_index": [0, 100, 200, 0, 100],
            **{c: [1.0, np.nan, 3.0, np.nan, 10.0] for c in CHANNELS},
        }
    )
    out = fill_gaps(rides)
    assert out[CHANNELS[0]].tolist() == [2.0 - 1, 2.0, 3.0, 10.0, 10.0]  # b's edge uses b's own value, not a's


def test_validate_rejects_ride_with_too_many_gaps():
    rides, labels = make_fake_rides(3, seed=0)
    labels = labels.rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET})
    first = rides[RIDE_ID] == rides[RIDE_ID].iloc[0]
    rides.loc[first, CHANNELS] = np.nan
    with pytest.raises(ValueError, match="missing samples"):
        validate(rides, labels, max_missing_share=0.05)


def test_validate_rejects_missing_channel():
    rides, labels = make_fake_rides(3, seed=0)
    labels = labels.rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET})
    with pytest.raises(ValueError, match="missing columns"):
        validate(rides.drop(columns="rz"), labels)


def test_validate_rejects_unknown_class():
    rides, labels = make_fake_rides(3, seed=0)
    labels = labels.rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET})
    labels.loc[0, TARGET] = "Drunk"
    with pytest.raises(ValueError, match="unknown classes"):
        validate(rides, labels)


def test_validate_rejects_ride_without_label():
    rides, labels = make_fake_rides(3, seed=0)
    labels = labels.rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET})
    with pytest.raises(ValueError, match="do not match"):
        validate(rides, labels.iloc[1:])


# --- split -----------------------------------------------------------------------
def test_split_is_by_ride_and_covers_every_ride(small_data):
    parts = split(small_data, val_share=0.15, test_share=0.15, random_state=0)
    ids = {name: set(part[RIDE_ID]) for name, part in parts.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert ids["train"] | ids["val"] | ids["test"] == set(small_data[RIDE_ID])


def test_split_keeps_every_class_in_every_split(small_data):
    for part in split(small_data, val_share=0.15, test_share=0.15, random_state=0).values():
        assert set(part[TARGET]) == set(CLASSES)


def test_split_rare_class_like_reckless():
    # 6 rides, as "Reckless" in GBG500: 4 train / 1 val / 1 test
    labels = pd.DataFrame({RIDE_ID: [f"r{i}" for i in range(6)], TARGET: "Reckless"})
    ids = stratified_ride_split(labels, val_share=0.15, test_share=0.15, random_state=0)
    assert [len(ids[s]) for s in ("train", "val", "test")] == [4, 1, 1]


def test_split_is_reproducible(small_data):
    a = split(small_data, 0.15, 0.15, random_state=42)["test"][RIDE_ID].unique()
    b = split(small_data, 0.15, 0.15, random_state=42)["test"][RIDE_ID].unique()
    assert list(a) == list(b)


# --- features --------------------------------------------------------------------
def test_scaler_standardises_train_data(small_data):
    scaler = ChannelScaler.fit(small_data)
    scaled = scaler.transform(small_data[CHANNELS].to_numpy())
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-2)
    assert ChannelScaler.from_dict(scaler.to_dict()) == scaler


def test_ride_windows_shape():
    windows = ride_windows(np.zeros((1000, 6)), window_size=256, stride=128)
    assert windows.shape == (6, 6, 256)  # starts 0,128,...,640


def test_short_ride_is_padded_to_one_window():
    windows = ride_windows(np.ones((100, 6)), window_size=256, stride=128)
    assert windows.shape == (1, 6, 256)
    assert (windows == 1).all()


def test_make_windows_labels_and_ride_ids(small_data):
    scaler = ChannelScaler.fit(small_data)
    X, y, ids = make_windows(small_data, scaler, window_size=128, stride=64)
    assert X.shape[1:] == (6, 128) and X.dtype == np.float32
    assert len(X) == len(y) == len(ids)
    assert set(y) == set(range(len(CLASSES)))
    assert set(ids) == set(small_data[RIDE_ID])


def test_make_windows_without_labels_for_serving(small_data):
    one_ride = small_data[small_data[RIDE_ID] == small_data[RIDE_ID].iloc[0]].drop(columns=TARGET)
    _, y, _ = make_windows(one_ride, ChannelScaler.fit(small_data), window_size=128, stride=64)
    assert (y == -1).all()


def test_aggregate_by_ride_returns_probabilities():
    proba = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
    out = aggregate_by_ride(proba, np.array(["a", "a", "b"]))
    assert list(out.columns) == CLASSES
    assert out.loc["a", "Safe"] == 0.5 and out.loc["b", "Reckless"] == 1.0
    np.testing.assert_allclose(out.sum(axis=1), 1)


# --- incident simulation (monitoring) ---------------------------------------------
def test_unit_bug_divides_accelerometer_by_g_only():
    from ride_classifier.simulate import G, apply_incident

    rides, _ = make_fake_rides(1, seed=0)
    ride = rides[rides[RIDE_ID] == rides[RIDE_ID].iloc[0]]
    out = apply_incident(ride, unit_bug=True, dropouts=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out["ay"].dropna(), ride["ay"].dropna() / G)
    pd.testing.assert_series_equal(out["rx"], ride["rx"])


def test_dropouts_blank_whole_samples():
    from ride_classifier.simulate import apply_incident

    rides, _ = make_fake_rides(1, seed=0)
    ride = rides[rides[RIDE_ID] == rides[RIDE_ID].iloc[0]].fillna(0.0)
    out = apply_incident(ride, unit_bug=False, dropouts=0.3, rng=np.random.default_rng(0))
    missing = out[CHANNELS].isna()
    assert 0.2 < missing.any(axis=1).mean() < 0.4
    assert (missing.all(axis=1) == missing.any(axis=1)).all()  # all 6 channels at once, like real dropouts
