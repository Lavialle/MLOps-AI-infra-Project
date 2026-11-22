import importlib
import io
import json

import pytest
from conftest import make_fake_rides
from fastapi.testclient import TestClient

from ride_classifier.config import CHANNELS, CLASSES


def ride_payload(n: int = 300, ride_id: str = "ride-1") -> dict:
    rides, _ = make_fake_rides(1, seed=3)
    ride = rides[rides["ride_id"] == rides["ride_id"].iloc[0]].head(n)
    samples = json.loads(ride[["time_index", *CHANNELS]].to_json(orient="records"))  # NaN -> null
    return {"ride_id": ride_id, "samples": samples}


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_URI", str(model_dir))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    import ride_classifier.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        c.log_path = tmp_path / "predictions.jsonl"
        yield c


# --- health & docs ---------------------------------------------------------------
def test_ready_when_model_loaded(client):
    assert client.get("/ready").status_code == 200


def test_health_exposes_model_version(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_health_is_degraded_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "nope"))
    import ride_classifier.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=ride_payload()).status_code == 503


def test_swagger_example_is_a_valid_request(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Ride"]
    r = client.post("/predict", json=schema["examples"][0])
    assert r.status_code == 200, r.text


# --- /predict ----------------------------------------------------------------------
def test_predict(client):
    r = client.post("/predict", json=ride_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["ride_id"] == "ride-1" and body["prediction"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["n_samples"] == 300 and body["model_version"] == "test"


def test_predict_accepts_sensor_dropouts(client):
    payload = ride_payload()
    payload["samples"][5] = {**payload["samples"][5], "ax": None, "ay": None}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["samples"][0].update(ax=500),  # out of physical range
        lambda p: p["samples"][0].update(rz="abc"),  # not a number
        lambda p: p["samples"][0].pop("ry"),  # missing channel
        lambda p: p["samples"][0].update(speed=3),  # unknown field
        lambda p: p.update(samples=p["samples"][:5]),  # too short
        lambda p: p["samples"][1].update(time_index=0),  # time not increasing
        lambda p: [s.update(ax=None) for s in p["samples"][:200]],  # mostly missing
    ],
)
def test_invalid_input_is_rejected(client, change):
    payload = ride_payload()
    change(payload)
    assert client.post("/predict", json=payload).status_code == 422


# --- /predict/file -----------------------------------------------------------------
def test_predict_file_parquet_with_several_rides(client):
    rides, _ = make_fake_rides(1, seed=4)  # 5 rides, one per class
    buf = io.BytesIO()
    rides.to_parquet(buf, index=False)
    r = client.post("/predict/file", files={"file": ("rides.parquet", buf.getvalue())})
    assert r.status_code == 200
    assert {p["ride_id"] for p in r.json()} == set(rides["ride_id"])


def test_predict_file_csv_without_ride_id(client):
    rides, _ = make_fake_rides(1, seed=4)
    one = rides[rides["ride_id"] == rides["ride_id"].iloc[0]].drop(columns="ride_id")
    r = client.post("/predict/file", files={"file": ("my_ride.csv", one.to_csv(index=False).encode())})
    assert r.status_code == 200
    assert r.json()[0]["ride_id"] == "my_ride"


@pytest.mark.parametrize(
    "filename,content,status",
    [
        ("ride.txt", b"hello", 415),
        ("ride.csv", b"time_index,ax\n0,1\n", 422),  # missing channels
        ("ride.parquet", b"not a parquet file", 422),
    ],
)
def test_predict_file_rejects_bad_files(client, filename, content, status):
    assert client.post("/predict/file", files={"file": (filename, content)}).status_code == status


# --- logs, metrics, feedback ---------------------------------------------------------
def test_prediction_is_logged(client):
    client.post("/predict", json=ride_payload())
    line = json.loads(client.log_path.read_text().splitlines()[-1])
    assert line["event"] == "prediction" and line["ride_id"] == "ride-1"
    assert {"prediction", "model_version", "mean_ax", "missing_share"} <= line.keys()


def test_metrics_exposed(client):
    client.post("/predict", json=ride_payload())
    text = client.get("/metrics/").text
    assert "api_request_duration_seconds" in text and "prediction_total" in text and "input_channel_mean" in text


def test_feedback(client):
    r = client.post("/feedback", json={"ride_id": "ride-1", "predicted": "Safe", "actual": "Reckless"})
    assert r.status_code == 204
    assert 'feedback_total{actual="Reckless",correct="false"}' in client.get("/metrics/").text


def test_feedback_rejects_unknown_label(client):
    r = client.post("/feedback", json={"ride_id": "ride-1", "predicted": "Safe", "actual": "Drunk"})
    assert r.status_code == 422
