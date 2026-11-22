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

## Data

| File | Content |
|------|---------|
| `GBG500.parquet` | 500 Gothenburg rides, 1.53M IMU rows (`ride_id, time_index, ax..rz`) |
| `GBG500_labels.csv` | One label per ride: Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6 |
| `SRV3490.parquet` | 3,490 rides in 8 cities with a `topple` (fall) flag (optional / auxiliary) |

The data is versioned with DVC and stored in the S3 `dvc` bucket (see [docs/data.md](docs/data.md)).

## Results

*(To come: metrics vs. baseline, confusion matrix, screenshots.)*

## Docker images

*(To come: Docker Hub URLs.)*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
