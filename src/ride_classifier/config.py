"""Central configuration: params.yaml + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

MODEL_NAME = os.getenv("MODEL_NAME", "ride-classifier")

RIDE_ID = "ride_id"
TIME = "time_index"
CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz"]  # accelerometer (m/s²) + gyroscope
TARGET = "label"
CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]


@lru_cache
def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
