import mlflow
import numpy as np
import pandas as pd
import torch
from conftest import make_fake_rides

from ride_classifier.config import CLASSES, RIDE_ID, TARGET
from ride_classifier.model import INPUT_COLUMNS, build_net, majority_baseline, ride_scores
from ride_classifier.train import class_weights


def fresh_rides(n_per_class: int = 3, seed: int = 99) -> pd.DataFrame:
    rides, labels = make_fake_rides(n_per_class, seed=seed)
    return rides.merge(labels.rename(columns={"RIDE_ID": RIDE_ID, "LABEL": TARGET}), on=RIDE_ID)


# --- network & weights -----------------------------------------------------------
def test_net_outputs_one_logit_per_class_for_any_window_size():
    net = build_net().eval()
    for window in (64, 256):
        assert net(torch.zeros(4, 6, window)).shape == (4, len(CLASSES))


def test_balanced_class_weights_favour_rare_classes():
    y = np.array([0] * 90 + [4] * 10)
    w = class_weights(y, "balanced")
    assert w[4] > w[0] > 0
    assert w[1] == 0  # absent class
    assert torch.equal(class_weights(y, "none"), torch.ones(len(CLASSES)))


# --- training ----------------------------------------------------------------------
def test_training_beats_majority_baseline_on_fresh_data(trained):
    model, best = trained
    holdout = fresh_rides()
    scores = ride_scores(holdout, model.predict(None, holdout), "holdout")
    baseline = majority_baseline(holdout, holdout[TARGET], "holdout")
    assert scores["holdout_macro_f1"] > baseline["holdout_baseline_macro_f1"] + 0.3
    assert best["best_epoch"] >= 1


# --- MLflow artifact (what the API will load) ----------------------------------------
def test_saved_model_takes_raw_rows_and_returns_one_row_per_ride(model_dir):
    model = mlflow.pyfunc.load_model(str(model_dir))
    rides = fresh_rides(1, seed=5)
    out = model.predict(rides[INPUT_COLUMNS])
    assert list(out.columns) == [RIDE_ID, "prediction", *CLASSES]
    assert set(out[RIDE_ID]) == set(rides[RIDE_ID])
    assert set(out["prediction"]) <= set(CLASSES)
    np.testing.assert_allclose(out[CLASSES].sum(axis=1), 1, atol=1e-5)


def test_saved_model_handles_short_ride_with_gaps(model_dir):
    model = mlflow.pyfunc.load_model(str(model_dir))
    ride = fresh_rides(1, seed=6)
    ride = ride[ride[RIDE_ID] == ride[RIDE_ID].iloc[0]].head(50).copy()  # shorter than one window
    ride.loc[ride.index[10], "ax"] = np.nan
    out = model.predict(ride[INPUT_COLUMNS])
    assert len(out) == 1


def test_saved_model_matches_in_memory_model(trained, model_dir):
    model, _ = trained
    rides = fresh_rides(1, seed=7)[INPUT_COLUMNS]
    np.testing.assert_allclose(
        mlflow.pyfunc.load_model(str(model_dir)).predict(rides)[CLASSES].to_numpy(),
        model.predict(None, rides)[CLASSES].to_numpy(),
        atol=1e-5,
    )


# --- metrics -----------------------------------------------------------------------
def test_majority_baseline_macro_f1_is_low_on_imbalanced_rides():
    truth = pd.DataFrame({RIDE_ID: [f"r{i}" for i in range(10)], TARGET: ["Safe"] * 8 + ["Reckless"] * 2})
    scores = majority_baseline(truth, truth[TARGET], "t")
    assert scores["t_baseline_accuracy"] == 0.8
    assert scores["t_baseline_macro_f1"] < 0.2  # 0.89 F1 on Safe, 0 on the 4 other classes


# --- what gets published -----------------------------------------------------------
def test_synthetic_ride_has_the_input_schema():
    from ride_classifier.model import synthetic_ride

    ride = synthetic_ride()
    assert list(ride.columns) == INPUT_COLUMNS and len(ride) == 300
    assert ride[RIDE_ID].unique().tolist() == ["synthetic-ride"]


def test_export_strips_input_example(model_dir, tmp_path):
    import shutil

    from ride_classifier.export_model import strip_input_example
    from ride_classifier.model import synthetic_ride

    path = tmp_path / "m"
    shutil.copytree(model_dir, path)
    model = mlflow.pyfunc.load_model(str(path))
    # re-save with an input example, as train.py does
    with_example = tmp_path / "with_example"
    mlflow.pyfunc.save_model(
        str(with_example),
        python_model=model.unwrap_python_model(),
        artifacts={"weights": str(path / "artifacts" / "weights.pt")},
        input_example=synthetic_ride(),
    )
    assert (with_example / "input_example.json").exists()
    strip_input_example(with_example)
    assert not list(with_example.glob("*input_example*"))
    assert "saved_input_example_info" not in (with_example / "MLmodel").read_text()
    assert len(mlflow.pyfunc.load_model(str(with_example)).predict(synthetic_ride())) == 1
