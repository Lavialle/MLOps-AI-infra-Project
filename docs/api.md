# Prediction API

FastAPI service in [`src/ride_classifier/api/`](../src/ride_classifier/api/):

- `main.py`: the endpoints
- `schemas.py`: the Pydantic contract
- `metrics.py`: the Prometheus metrics

It serves the MLflow pyfunc model, which does gap filling, scaling, windowing and per-ride aggregation itself.
The API therefore only validates and forwards raw IMU rows.

```bash
make export      # @staging model -> ./model
make api         # http://localhost:8000, Swagger UI at http://localhost:8000/docs
# or straight from the registry:
MODEL_URI=models:/ride-classifier@staging uv run uvicorn ride_classifier.api.main:app --port 8000
```

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | – | `status` (`ok` / `degraded` if the model failed to load), `model_loaded`, `model_version` |
| POST | `/predict` | JSON ride: `ride_id`, `samples[]` of `{time_index, ax, ay, az, rx, ry, rz}` | one `RidePrediction` |
| POST | `/predict/file` | multipart upload of a `.parquet` or `.csv` file (GBG500 format, `ride_id` optional, several rides allowed) | list of `RidePrediction` |
| POST | `/feedback` | `{ride_id, predicted, actual}` (true label received later) | 204 |
| GET | `/metrics` | – | Prometheus metrics |
| GET | `/docs` | – | Swagger UI, with a ready-to-run example |

`RidePrediction`:

```json
{
  "ride_id": "demo-ride",
  "prediction": "Safe",
  "probabilities": {"Safe": 0.71, "Safe overall": 0.12, "Bad weather": 0.09, "Tandem": 0.05, "Reckless": 0.03},
  "n_samples": 2349,
  "model_version": "1"
}
```

Example call:

```bash
curl -X POST http://localhost:8000/predict/file -F "file=@data/processed/test.parquet"
```

## Input validation (422 before reaching the model)

| Rule | Why |
|------|-----|
| `ax, ay, az` in [-80, 80] m/s², `rx, ry, rz` in [-2000, 2000] | GBG500 stays within ±19 and ±300. A value beyond means a wrong unit or a corrupted file |
| `null` allowed on channels, but at most 50 % of a ride | sensor dropouts are interpolated, as in training (0.7 % of GBG500 rows) |
| 10 to 20,000 samples per ride, `time_index` ≥ 0 and increasing | a ride shorter than one window (256 samples) is padded, so its prediction is less reliable |
| no unknown fields (`extra="forbid"`) | catches client typos |
| `/predict/file`: `.parquet` / `.csv` only, ≤ 50 MB, ≤ 100 rides | protects the service |

## Observability

Each prediction is logged as one JSON line on stdout and, if `PREDICTION_LOG_PATH` is set, to a file.
A line holds the ride id, sample count, missing share, per-channel means, prediction, probabilities, model version and latency.
These logs are the input for drift analysis.

Prometheus metrics (`/metrics`) are grouped in the same 4 layers as in the labs:

| Layer | Metrics |
|-------|---------|
| Infrastructure | `api_requests_total{endpoint,status}`, `api_request_duration_seconds{endpoint}` |
| Inputs | `input_ride_duration_seconds`, `input_channel_mean{channel}`, `input_missing_share` |
| Predictions | `prediction_total{label}`, `prediction_confidence` |
| Outcome | `feedback_total{actual,correct}` |
| Model | `model_info{version}` |

## Performance

`make load` runs Locust with 20 users for 1 minute, posting real test rides (~3,000 samples each).
It was run on a laptop with one uvicorn process and model v1:

| Endpoint | Requests | Failures | Median | P95 | Throughput |
|----------|----------|----------|--------|-----|------------|
| `/predict` | 938 | 0 | 78 ms | 340 ms | 16.5 rides/s |
| `/health` | 58 | 0 | 2 ms | 26 ms | – |

The whole test split (75 rides, 5.3 MB parquet) goes through `/predict/file` in about 1.1 s.
