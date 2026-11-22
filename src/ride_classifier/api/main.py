"""Ride classification API.

    MODEL_URI=model uvicorn ride_classifier.api.main:app --port 8000
    MODEL_URI=models:/ride-classifier@staging uvicorn ride_classifier.api.main:app   # straight from the Registry

Endpoints: POST /predict (JSON ride), POST /predict/file (parquet/CSV, one or more rides),
POST /feedback, GET /health (liveness), GET /ready (readiness), GET /metrics (Prometheus). Swagger UI: /docs.

The model is the MLflow pyfunc logged by ride_classifier.train: it takes RAW IMU rows
and does gap filling, scaling, windowing and per-ride aggregation itself.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from prometheus_client import make_asgi_app

from ride_classifier.api import metrics as m
from ride_classifier.api.schemas import MAX_MISSING_SHARE, MAX_SAMPLES, Feedback, Health, Ride, RidePrediction
from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, TIME
from ride_classifier.model import INPUT_COLUMNS

MODEL_URI = os.getenv("MODEL_URI", "model")
LOG_PATH = os.getenv("PREDICTION_LOG_PATH")  # e.g. logs/predictions.jsonl, input of the drift analysis
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_RIDES_PER_FILE = 100

logger = logging.getLogger("ride_classifier.api")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False

state: dict = {"model": None, "version": "none"}


def read_model_meta(uri: str) -> dict:
    meta = Path(uri) / "model_meta.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def log_event(event: dict) -> None:
    line = json.dumps(event, default=str)
    logger.info(line)
    if LOG_PATH:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        meta = read_model_meta(MODEL_URI)
        state["version"] = str(meta.get("version", os.getenv("MODEL_VERSION", MODEL_URI)))
        m.MODEL_INFO.labels(version=state["version"]).set(1)
    except Exception as exc:  # the API still starts, /health reports "degraded"
        logger.error(json.dumps({"event": "model_load_failed", "model_uri": MODEL_URI, "error": str(exc)}))
    yield


app = FastAPI(
    title="Ride classification API",
    version="0.1.0",
    description=(
        "Classifies a bike ride from its IMU time series (accelerometer `ax, ay, az`, gyroscope `rx, ry, rz`, "
        f"10 Hz) into one of {', '.join(CLASSES)}."
    ),
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if not path.startswith("/metrics"):
            m.REQUESTS.labels(endpoint=path, status=str(status)).inc()
            m.LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)


def predict_rides(df: pd.DataFrame) -> list[RidePrediction]:
    """Rows of one or more rides -> one prediction per ride (+ metrics and prediction log)."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    out = state["model"].predict(df[INPUT_COLUMNS])
    elapsed_ms = (time.perf_counter() - start) * 1000

    results = []
    for row in out.itertuples(index=False):
        ride = df[df[RIDE_ID] == row.ride_id]
        proba = {label: float(p) for label, p in zip(CLASSES, row[2:], strict=True)}
        means = ride[CHANNELS].mean()
        missing = float(ride[CHANNELS].isna().any(axis=1).mean())

        m.INPUT_DURATION.observe((ride[TIME].max() - ride[TIME].min()) / 1000)
        for channel in CHANNELS:
            m.INPUT_CHANNEL_MEAN.labels(channel=channel).observe(means[channel])
        m.INPUT_MISSING_SHARE.observe(missing)
        m.PREDICTIONS.labels(label=row.prediction).inc()
        m.PREDICTION_CONFIDENCE.observe(proba[row.prediction])

        log_event(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "prediction",
                "ride_id": row.ride_id,
                "n_samples": len(ride),
                "missing_share": round(missing, 4),
                **{f"mean_{c}": round(float(means[c]), 4) for c in CHANNELS},
                "prediction": row.prediction,
                **{f"proba_{k}": round(v, 4) for k, v in proba.items()},
                "model_version": state["version"],
                "inference_ms": round(elapsed_ms, 2),
            }
        )
        results.append(
            RidePrediction(
                ride_id=row.ride_id,
                prediction=row.prediction,
                probabilities=proba,
                n_samples=len(ride),
                model_version=state["version"],
            )
        )
    return results


@app.get("/health", response_model=Health)
def health() -> Health:
    loaded = state["model"] is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded, model_version=state["version"])


@app.get("/ready", summary="Readiness probe: 503 until the model is loaded")
def ready() -> dict:
    """Kubernetes readinessProbe. /health stays 200 when degraded (liveness: restarting would not help),
    /ready returns 503 so that a pod without a model never receives traffic."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.post("/predict", response_model=RidePrediction, summary="Classify one ride sent as JSON")
def predict(ride: Ride) -> RidePrediction:
    df = pd.DataFrame([s.model_dump() for s in ride.samples]).astype({c: float for c in CHANNELS})
    df.insert(0, RIDE_ID, ride.ride_id)
    return predict_rides(df)[0]


@app.post(
    "/predict/file",
    response_model=list[RidePrediction],
    summary="Classify the rides of a parquet or CSV file",
    description=(
        "Columns: `time_index, ax, ay, az, rx, ry, rz` and optionally `ride_id` (several rides per file). "
        "Same format as GBG500.parquet."
    ),
)
async def predict_file(file: Annotated[UploadFile, File()]) -> list[RidePrediction]:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // 1024**2} MB")
    name = (file.filename or "").lower()
    try:
        if name.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        elif name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=415, detail="Expected a .parquet or .csv file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unreadable file: {exc}") from exc

    if RIDE_ID not in df:
        df.insert(0, RIDE_ID, Path(name).stem or "ride")
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {sorted(missing)}")
    df = df[INPUT_COLUMNS].astype({RIDE_ID: str, TIME: "int64", **{c: float for c in CHANNELS}})
    sizes = df.groupby(RIDE_ID).size()
    if len(sizes) > MAX_RIDES_PER_FILE:
        raise HTTPException(status_code=422, detail=f"More than {MAX_RIDES_PER_FILE} rides in one file")
    if sizes.min() < 10 or sizes.max() > MAX_SAMPLES:
        raise HTTPException(status_code=422, detail=f"Each ride needs between 10 and {MAX_SAMPLES} samples")
    if df[CHANNELS].isna().any(axis=1).groupby(df[RIDE_ID]).mean().max() > MAX_MISSING_SHARE:
        raise HTTPException(status_code=422, detail="A ride has more than half of its samples missing")
    if df[TIME].lt(0).any() or not df.groupby(RIDE_ID)[TIME].apply(lambda t: t.is_monotonic_increasing).all():
        raise HTTPException(status_code=422, detail="time_index must be >= 0 and increasing within each ride")
    return predict_rides(df)


@app.post("/feedback", status_code=204, summary="Report the true label of a ride already predicted")
def feedback(fb: Feedback) -> None:
    m.FEEDBACK.labels(actual=fb.actual, correct=str(fb.predicted == fb.actual).lower()).inc()
    log_event({"ts": datetime.now(UTC).isoformat(), "event": "feedback", **fb.model_dump()})
