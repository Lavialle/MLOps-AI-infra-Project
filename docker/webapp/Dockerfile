#   docker build -f docker/webapp/Dockerfile -t ride-classifier-webapp:dev .   (make image-webapp)
#   docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 ride-classifier-webapp:dev
#
# The web app only calls the API: no model, no torch, no mlflow in this image (webapp group only).

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Exact versions of the webapp group from uv.lock, without the project's core dependencies
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --only-group webapp --no-hashes --no-emit-project -o requirements.txt \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -r requirements.txt

# The app imports ride_classifier.client and ride_classifier.config only (no model code is executed)
COPY params.yaml ./
COPY src/ride_classifier/__init__.py src/ride_classifier/config.py src/ride_classifier/client.py ./src/ride_classifier/
COPY webapp ./webapp

RUN useradd --create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    API_URL=http://api:8000 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
