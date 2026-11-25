# Docker images

| Image | Dockerfile | Contents | Port |
|-------|------------|----------|------|
| `<DOCKERHUB_USER>/ride-classifier-api` | [`Dockerfile`](../Dockerfile) | core dependencies (FastAPI, MLflow, CPU-only torch), the package, **the model `@staging` baked in** | 8000 |
| `<DOCKERHUB_USER>/ride-classifier-webapp` | [`docker/webapp/Dockerfile`](../docker/webapp/Dockerfile) | `webapp` dependency group only (Streamlit, plotly, httpx), the API client | 8501 |
| `ride-classifier-mlflow` (local only) | [`docker/mlflow/Dockerfile`](../docker/mlflow/Dockerfile) | MLflow server + Postgres and S3 drivers | 5000 |

Each image is tagged with the git commit (`:<short sha>`) and `:latest`.

## Build, run, push

```bash
# DOCKERHUB_USER=<your Docker Hub account> in .env
make export         # model @staging -> ./model
make image          # API image (runs `make export` itself)
make image-webapp   # web app image
make app            # docker compose --profile app up -d: API on :8000, web app on :8501
docker login
make push           # builds both images for linux/amd64 + linux/arm64 and pushes :<sha> and :latest
```

`make image` / `make image-webapp` build for the local architecture only (enough for `make app` and `kind load`).
`make push` rebuilds with `docker buildx` for both architectures, so the Docker Hub images run on Apple silicon and
on x86 machines alike. The amd64 half is emulated on an arm64 Mac (QEMU, bundled with Docker Desktop), which
makes the first push slower.

## Design choices

- **The model is inside the API image**, as in the labs. A tagged image is a frozen, reproducible model. The
  Kubernetes pod never needs to reach MLflow, and rolling back the model means rolling back the image tag.
  `/health` reports the model version from `model/model_meta.json`.
- **Only what each image needs.** `pyproject.toml` separates the core dependencies from the `dev` group (DVC,
  pytest, Locust) and the `webapp` group. The API image installs the core only (`uv sync --no-default-groups`).
  The web app image installs only the `webapp` group, with no torch and no MLflow, so it is much smaller. Versions
  come from `uv.lock` in both cases.
- **CPU-only torch** on Linux (`[tool.uv.sources]` in `pyproject.toml`) avoids about 2 GB of unused CUDA libraries.
- **Same base as the labs:** `python:3.14-slim`, `uv` copied from its official image, dependency layer before
  the code layer (cache), non-root `appuser`, `HEALTHCHECK` on `/health` (API) and `/_stcore/health` (web app).
- **Configuration through environment variables:** `MODEL_URI` and `PREDICTION_LOG_PATH` for the API, `API_URL` for
  the web app. The same images run in docker compose (dev) and Kubernetes (prod).
