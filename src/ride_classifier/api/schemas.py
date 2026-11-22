"""API contract: Pydantic rejects any out-of-domain input with a 422, before it reaches the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ride_classifier.config import CLASSES

Label = Literal["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
assert list(Label.__args__) == CLASSES  # one source of truth for the classes

# Physical bounds, far outside what GBG500 contains (|acc| <= 19 m/s², |gyro| <= 300):
# a value beyond is a client bug (wrong unit, corrupted file), not a ride.
ACC = {"ge": -80, "le": 80, "description": "Accelerometer, m/s² (gravity included). null = sensor dropout"}
GYRO = {"ge": -2000, "le": 2000, "description": "Gyroscope. null = sensor dropout"}
MAX_SAMPLES = 20_000  # 33 min at 10 Hz; the longest GBG500 ride has 4,767 samples
MAX_MISSING_SHARE = 0.5

# Swagger example: 4 s of a phone at rest on the handlebar (gravity on ay), slightly shaking
EXAMPLE_SAMPLES = [
    {
        "time_index": i * 100,
        "ax": round(-2.5 + 0.05 * (i % 3), 3),
        "ay": round(-9.38 - 0.02 * (i % 4), 3),
        "az": round(-1.05 + 0.03 * (i % 5), 3),
        "rx": round(0.04 * ((i % 3) - 1), 3),
        "ry": 0.02,
        "rz": round(-0.15 + 0.05 * (i % 2), 3),
    }
    for i in range(40)
]


class Sample(BaseModel):
    """One IMU reading. Sensor dropouts (null) are interpolated by the model, as in training."""

    model_config = ConfigDict(extra="forbid")

    time_index: int = Field(ge=0, description="Milliseconds since the start of the ride (10 Hz: step of 100)")
    ax: float | None = Field(**ACC)
    ay: float | None = Field(**ACC)
    az: float | None = Field(**ACC)
    rx: float | None = Field(**GYRO)
    ry: float | None = Field(**GYRO)
    rz: float | None = Field(**GYRO)


class Ride(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"ride_id": "demo-ride", "samples": EXAMPLE_SAMPLES}]},
    )

    ride_id: str = Field(default="ride", min_length=1, max_length=128)
    samples: list[Sample] = Field(
        min_length=10,
        max_length=MAX_SAMPLES,
        description="IMU readings at 10 Hz. Rides shorter than one window (25.6 s) are padded: less reliable.",
    )

    @model_validator(mode="after")
    def time_is_increasing(self) -> Ride:
        times = [s.time_index for s in self.samples]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("time_index must be strictly increasing")
        n_missing = sum(any(v is None for v in (s.ax, s.ay, s.az, s.rx, s.ry, s.rz)) for s in self.samples)
        if n_missing > MAX_MISSING_SHARE * len(times):
            raise ValueError("more than half of the samples have missing values")
        return self


class RidePrediction(BaseModel):
    ride_id: str
    prediction: Label
    probabilities: dict[Label, float]
    n_samples: int
    model_version: str


class Feedback(BaseModel):
    """True label received later (user report, manual review): the "outcome" layer."""

    model_config = ConfigDict(extra="forbid")

    ride_id: str
    predicted: Label
    actual: Label


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
