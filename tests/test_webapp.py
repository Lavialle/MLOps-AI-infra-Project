"""Web app: the API client against the real API app (TestClient), and a smoke test of the Streamlit page."""

import importlib
import io

import pytest
from conftest import make_fake_rides
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from ride_classifier.client import ApiClient, read_rides
from ride_classifier.config import CLASSES, ROOT


@pytest.fixture
def api(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_URI", str(model_dir))
    monkeypatch.delenv("PREDICTION_LOG_PATH", raising=False)
    import ride_classifier.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield ApiClient("http://testserver", http=c)


# --- read_rides --------------------------------------------------------------------
def test_read_rides_parquet_keeps_ride_ids():
    rides, _ = make_fake_rides(1, seed=8)
    buf = io.BytesIO()
    rides.to_parquet(buf, index=False)
    out = read_rides("rides.parquet", buf.getvalue())
    assert set(out["ride_id"]) == set(rides["ride_id"]) and len(out) == len(rides)


def test_read_rides_csv_without_ride_id_uses_filename():
    rides, _ = make_fake_rides(1, seed=8)
    one = rides[rides["ride_id"] == rides["ride_id"].iloc[0]].drop(columns="ride_id")
    out = read_rides("my_ride.csv", one.to_csv(index=False).encode())
    assert out["ride_id"].unique().tolist() == ["my_ride"]


@pytest.mark.parametrize(
    "name,content,match", [("a.txt", b"x", "parquet or .csv"), ("a.csv", b"time_index,ax\n0,1\n", "Missing")]
)
def test_read_rides_rejects_bad_files(name, content, match):
    with pytest.raises(ValueError, match=match):
        read_rides(name, content)


# --- ApiClient <-> API ---------------------------------------------------------------
def test_client_health(api):
    assert api.health()["model_loaded"]


def test_client_predicts_every_ride(api):
    rides, _ = make_fake_rides(1, seed=9)
    out = api.predict(rides)
    assert set(out["ride_id"]) == set(rides["ride_id"])
    assert set(out["prediction"]) <= set(CLASSES)
    assert {*CLASSES, "model_version"} <= set(out.columns)


def test_client_surfaces_api_validation_errors(api):
    rides, _ = make_fake_rides(1, seed=9)
    with pytest.raises(RuntimeError, match="422"):
        api.predict(rides.head(5))  # shorter than the 10-sample minimum


def test_client_feedback(api):
    api.feedback("ride-1", "Safe", "Tandem")


# --- Streamlit page ----------------------------------------------------------------
def test_app_starts_and_reports_unreachable_api(monkeypatch, tmp_path):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.setenv("EXAMPLES_PATH", str(tmp_path / "none.parquet"))
    at = AppTest.from_file(str(ROOT / "webapp/app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("unreachable" in e.value for e in at.error)
