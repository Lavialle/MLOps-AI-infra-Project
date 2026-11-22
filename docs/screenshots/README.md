# Screenshots to capture (used by the main README)

Save each one in this folder under the exact file name below (PNG, about 1600 px wide).

| File | What to show | Where |
|------|--------------|-------|
| `mlflow.png` | Experiment `ride-classifier`: run list with `val_macro_f1`, or a run's metric charts (per epoch); ideally also the Models tab with `ride-classifier` and the `@staging` alias | Prod MLflow http://localhost:5050 |
| `swagger.png` | Swagger UI with `POST /predict` expanded and a 200 response (use *Try it out* with the built-in example) | http://localhost:8080/docs |
| `webapp.png` | *Example rides* → *Classify* → a ride selected: signal charts, prediction and probability bars | http://localhost:8081 (prod) or http://localhost:8501 (dev, needs `data/processed/test.parquet` for the examples) |
| `grafana.png` | Dashboard *Ride classifier · 4-layer monitoring* after `make simulate` (all four rows filled) | `make monitoring-forward`, then http://localhost:3030 |
| `cd-run.png` | A green **CD (build, push, deploy to Kubernetes)** run: the job graph or the step list including "End-to-end tests against prod" | GitHub → Actions |
| `k8s-pods.png` | Terminal with `make k8s-status` (all pods Running) | terminal |

Optional extras for the slides: `retrain-run.png` (green *Retrain & redeploy* run), `prometheus-alerts.png`
(Prometheus → Alerts after `make simulate-unit-bug` for ~15 min), `dockerhub.png` (the image tags).
