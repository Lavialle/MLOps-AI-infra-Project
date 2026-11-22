#   uv run python -m ride_classifier.export_model   # Registry @staging -> ./model   (make export)
#   docker build -t ride-classifier-api:dev .        # (make image)
#
# The model is IN the image: a tagged image is a frozen model, and the Kubernetes pod
# never needs to reach the MLflow server.

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Core dependencies only (no dev / webapp groups); torch comes from the CPU-only index (pyproject.toml)
COPY pyproject.toml uv.lock .python-version ./
# Cache mount: uv's download cache (~1.4 GB with torch) stays on the build machine, not in the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

COPY README.md params.yaml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable \
    && rm -rf build src/*.egg-info

COPY model ./model

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/model
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "ride_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
