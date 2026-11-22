.DEFAULT_GOAL := help

# .env (cp .env.example .env) provides MLFLOW_TRACKING_URI etc. to every command below
-include .env
export

help: ## Command list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Services ----------------------------------------------------------------
setup: ## Installs the Python environment (uv sync), once per clone
	uv sync
	@test -f .env || cp .env.example .env

up: ## Starts S3 and MLflow
	docker compose up -d --build

down: ## Stops services (volumes are kept)
	docker compose down

# --- Model lifecycle -----------------------------------------------------------
pull: ## Downloads the raw data versioned with DVC (needs `make up`)
	uv run dvc pull

data: ## Validates raw data and splits it by ride (data/processed/{train,val,test})
	uv run python -m ride_classifier.data

train: ## Trains and saves a version to MLflow
	uv run python -m ride_classifier.train

evaluate: ## Gate macro F1 on the test split, promote to @staging if OK
	uv run python -m ride_classifier.evaluate

pipeline: ## Pipeline DVC (split -> train -> evaluate)
	uv run dvc repro

dag: ## Prints the pipeline DAG (dvc.yaml)
	uv run dvc dag

export: ## Downloads model @staging in ./model
	uv run python -m ride_classifier.export_model

# --- Serving -------------------------------------------------------------------
api: ## Local API (model ./model, run `make export` first), logs in logs/predictions.jsonl
	mkdir -p logs
	PREDICTION_LOG_PATH=logs/predictions.jsonl MODEL_URI=model uv run uvicorn ride_classifier.api.main:app --reload --host 0.0.0.0 --port 8000

webapp: ## Streamlit web app on http://localhost:8501 (needs `make api` running)
	API_URL=http://localhost:8000 uv run streamlit run webapp/app.py --server.headless true

load: ## Load test 1 min, 20 users (results in reports/load_*)
	mkdir -p reports
	uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 1m --csv reports/load

# --- Code quality ---------------------------------------------------------------
lint: ## Ruff
	uv run ruff check --fix src tests && uv run ruff format --check src tests

test: ## Tests
	uv run pytest -q

.PHONY: help setup up down pull data train evaluate pipeline dag export api webapp load lint test
