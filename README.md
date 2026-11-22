# Ride Classifier: MLOps & AI Infrastructure Project

End-to-end MLOps pipeline that classifies bike rides into five categories:
**Safe**, **Safe overall**, **Bad weather**, **Tandem** and **Reckless**.
The input is each ride's IMU time series (accelerometer `ax, ay, az` and gyroscope `rx, ry, rz`, sampled at 10 Hz).

> Albert School: MLOps & Model Deployment. Status: 🚧 work in progress.

## Team

| Name | Main area |
|------|-----------|
| TBD  | Data & preprocessing |
| TBD  | Model & MLflow |
| TBD  | API & web app |
| TBD  | Docker & Kubernetes |
| TBD  | CI/CD & monitoring |

## Architecture

*(Diagram to come.)*

| Component | Dev (docker compose) | Prod (Kubernetes) |
|-----------|----------------------|-------------------|
| Object storage (S3) | SeaweedFS | SeaweedFS |
| Data versioning & pipeline | DVC (S3 remote) | – |
| Experiment tracking & registry | MLflow + Postgres | MLflow + Postgres |
| Prediction API | FastAPI | FastAPI |
| Web app | Streamlit | Streamlit |
| Monitoring | Prometheus + Grafana | Prometheus + Grafana |

**Why SeaweedFS and not MinIO.** The brief asks for "AWS S3 (Minio)". MinIO no longer publishes its Docker images:
`minio/minio` is gone from Docker Hub and `quay.io/minio/minio` refuses anonymous pulls. We therefore use SeaweedFS,
the S3-compatible store used in the course labs. All code talks to it through the standard S3 API (boto3, DVC, MLflow),
so switching to MinIO or AWS S3 only means changing the endpoint and credentials.

## Repository layout

```
src/ride_classifier/   # one module per pipeline step (python -m ride_classifier.<step>), api/
webapp/                # Streamlit app for testing the model
params.yaml            # parameters (tracked by DVC, logged to MLflow)
docker/                # Dockerfiles of the services (MLflow server)
infra/                 # service configs (S3, Prometheus, Grafana)
k8s/                   # Kubernetes manifests (prod)
tests/                 # tests
docs/                  # extra documentation
.github/workflows/     # CI/CD
```

## Quickstart (dev)

Prerequisites: [uv](https://docs.astral.sh/uv/) and Docker Desktop.

```bash
make setup     # Python env (3.14) + .env
make up        # S3 (SeaweedFS), Postgres, MLflow
make test
```

| Service | URL |
|---------|-----|
| MLflow | http://localhost:5000 |
| S3 file browser | http://localhost:8888/buckets/ |

Run `make help` to list every command.

## Pipeline

`make pipeline` runs the DVC DAG `split → train → evaluate` (see [docs/pipeline.md](docs/pipeline.md)).
Each step logs to MLflow, and the evaluation gate promotes the model to `@staging` only if it beats the baseline.

## API

FastAPI with Swagger UI at `/docs`. It accepts a ride as JSON (`POST /predict`) or as a parquet/CSV file
(`POST /predict/file`, several rides at once), and also exposes `/health`, `/feedback` and Prometheus `/metrics`.
See [docs/api.md](docs/api.md).

```bash
make export && make api      # http://localhost:8000/docs
```

## Data

| File | Content |
|------|---------|
| `GBG500.parquet` | 500 Gothenburg rides, 1.53M IMU rows (`ride_id, time_index, ax..rz`) |
| `GBG500_labels.csv` | One label per ride: Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6 |
| `SRV3490.parquet` | 3,490 rides in 8 cities with a `topple` (fall) flag (optional / auxiliary) |

The data is versioned with DVC and stored in the S3 `dvc` bucket (see [docs/data.md](docs/data.md)).

## Results

1D-CNN on 25.6 s IMU windows, class-weighted loss, early stopping on validation macro F1 (see [docs/model.md](docs/model.md)).
Scores are ride-level. The baseline always predicts "Safe".

| Split (75 rides) | Macro F1 | Balanced accuracy | Accuracy | Baseline macro F1 | Baseline accuracy |
|------------------|----------|-------------------|----------|-------------------|-------------------|
| Validation (best epoch 17) | **0.567** | 0.565 | 0.747 | 0.181 | 0.827 |
| Test (evaluation gate) | **0.317** | 0.458 | 0.693 | 0.181 | 0.827 |

The model beats the baseline on macro F1 and balanced accuracy. The baseline's higher accuracy only reflects the
83 % share of Safe rides. Val and test each hold a single Reckless ride and 3 Tandem / Bad weather rides, so the
scores vary a lot between epochs and splits. The next improvement would be cross-validation or data augmentation.

*(Screenshots of MLflow to come.)*

## Docker images

*(To come: Docker Hub URLs.)*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
