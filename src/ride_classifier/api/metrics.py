"""Prometheus metrics, one family per monitoring layer."""

from prometheus_client import Counter, Gauge, Histogram

# Layer 1: infrastructure
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)

# Layer 2: input distribution
INPUT_DURATION = Histogram(
    "input_ride_duration_seconds",
    "Duration of incoming rides",
    buckets=(10, 30, 60, 120, 180, 240, 300, 400, 600, 1200),
)
# Mean of each channel over a ride: a unit bug (e.g. g instead of m/s²) or a mounted-differently phone shifts these
INPUT_CHANNEL_MEAN = Histogram(
    "input_channel_mean",
    "Per-ride mean of each IMU channel",
    ["channel"],
    buckets=(-15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15),
)
INPUT_MISSING_SHARE = Histogram(
    "input_missing_share",
    "Share of samples with a sensor dropout",
    buckets=(0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5),
)

# Layer 3: prediction distribution
PREDICTIONS = Counter("prediction_total", "Rides predicted per class", ["label"])
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Probability of the predicted class",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# Layer 4: business outcome (labels received later)
FEEDBACK = Counter("feedback_total", "True labels received", ["actual", "correct"])

MODEL_INFO = Gauge("model_info", "Model version being served", ["version"])
