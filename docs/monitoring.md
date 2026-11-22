# Monitoring (Prometheus + Grafana)

The API exposes Prometheus metrics on `/metrics`. Prometheus scrapes them every 5–10 s and evaluates the alerts,
and Grafana shows a **4-layer dashboard** (same structure as the course labs). Both run in the two environments
from the **same config files** in [`infra/monitoring/`](../infra/monitoring/):

| | Dev (docker compose) | Prod (kind cluster) |
|---|---|---|
| Start | `make up` or `make monitoring` | deployed with the rest (`make k8s-deploy`, CD pipeline) |
| Grafana | http://localhost:3000 | http://localhost:3030 (`make monitoring-forward` on a cluster created before step 9) |
| Prometheus | http://localhost:9090 | http://localhost:9091 |
| API discovery | static targets (`prometheus-compose.yml`) | **Kubernetes pod discovery**: every pod annotated `prometheus.io/scrape: "true"` (`prometheus-k8s.yml`), one series per replica |

Grafana opens directly on the provisioned dashboard. Anonymous users are read-only viewers; the admin is `admin` / `admin`
in dev, and the password comes from the `GRAFANA_ADMIN_PASSWORD` Secret in prod.

## Dashboard and alerts

| Layer | Panels | Alerts (`alerts.yml`) |
|-------|--------|-----------------------|
| 1 · Infrastructure | instances up, model version served, `/predict` latency P50/P95, requests/s by status | `ApiDown`, `LatencyP95High` (> 1 s), `ErrorRateHigh` (> 5 % 5xx) |
| 2 · Inputs | median per-ride mean of each IMU channel, median ride duration, P90 sensor-dropout share | `InputGravityShift` (median `ay` > -5 m/s²), `SensorDropoutsHigh` (P90 > 5 %) |
| 3 · Predictions | share of rides per predicted class, median confidence | `PredictedRecklessShareHigh` (> 15 %), `PredictionConfidenceLow` (< 0.4) |
| 4 · Outcome | accuracy on rides with a true label (`POST /feedback`), labels per class correct/wrong | `FeedbackAccuracyLow` (< 50 %, ≥ 10 labels/h) |

Thresholds come from the baselines measured on the test split with model v1: 69 % Safe, 5 % Reckless,
median confidence 0.54, accuracy 69 %. They also come from the load test (P95 340 ms). Each alert links to a
section of [runbooks.md](runbooks.md).

## Simulate traffic and incidents

```bash
make simulate               # 50 real test rides + their true labels -> fills all 4 rows
make simulate-unit-bug      # accelerometer in g instead of m/s²: ay jumps from -9.4 to about -1 -> InputGravityShift
make simulate-dropouts      # 20 % of the samples lost -> SensorDropoutsHigh
# against dev: make simulate API_URL=http://localhost:8000
```

The input and prediction alerts use 10–15 min windows plus a `for:` duration, so a few odd rides never page
anyone. An incident has to last about 10–15 minutes before it fires.
