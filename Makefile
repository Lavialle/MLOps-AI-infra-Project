.DEFAULT_GOAL := help

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

pipeline: ## Pipeline DVC (split -> train -> evaluate)
	uv run dvc repro

# --- Code quality ---------------------------------------------------------------
lint: ## Ruff
	uv run ruff check --fix src tests && uv run ruff format --check src tests

test: ## Tests
	uv run pytest -q

.PHONY: help setup up down pull data pipeline lint test
