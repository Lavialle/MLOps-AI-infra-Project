"""Thin HTTP client for the prediction API, used by the web app (webapp/app.py).

The web app never loads the model: it only talks to the API, like any other client.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

from ride_classifier.config import CHANNELS, RIDE_ID, TIME


def read_rides(filename: str, content: bytes) -> pd.DataFrame:
    """Parquet/CSV bytes -> rows sorted by ride and time. A file without ride_id is one ride named after the file."""
    name = filename.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        raise ValueError("Expected a .parquet or .csv file")
    missing = {TIME, *CHANNELS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if RIDE_ID not in df:
        df.insert(0, RIDE_ID, Path(filename).stem)
    return df.astype({RIDE_ID: str}).sort_values([RIDE_ID, TIME], kind="stable").reset_index(drop=True)


class ApiClient:
    """`http` can be any httpx.Client, e.g. FastAPI's TestClient in the tests."""

    def __init__(self, base_url: str, timeout: float = 60.0, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def health(self) -> dict:
        r = self.http.get("/health")
        r.raise_for_status()
        return r.json()

    def predict(self, rides: pd.DataFrame) -> pd.DataFrame:
        """All rides of the frame in one /predict/file call -> one row per ride (prediction + probabilities)."""
        buf = io.BytesIO()
        rides[[RIDE_ID, TIME, *CHANNELS]].to_parquet(buf, index=False)
        r = self.http.post("/predict/file", files={"file": ("rides.parquet", buf.getvalue())})
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        rows = [
            {
                RIDE_ID: p["ride_id"],
                "prediction": p["prediction"],
                **p["probabilities"],
                "model_version": p["model_version"],
            }
            for p in r.json()
        ]
        return pd.DataFrame(rows)

    def feedback(self, ride_id: str, predicted: str, actual: str) -> None:
        r = self.http.post("/feedback", json={"ride_id": ride_id, "predicted": predicted, "actual": actual})
        r.raise_for_status()
