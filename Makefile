.DEFAULT_GOAL := help

# .env (cp .env.example .env) provides MLFLOW_TRACKING_URI etc. to every command below
-include .env
export

DOCKERHUB_USER ?= local
VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE_API := $(DOCKERHUB_USER)/ride-classifier-api
IMAGE_WEBAPP := $(DOCKERHUB_USER)/ride-classifier-webapp

help: ## Command list
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Services ----------------------------------------------------------------
setup: ## Installs the Python environment (uv sync) and the ruff pre-commit hook, once per clone
	uv sync
	uv run pre-commit install
	@test -f .env || cp .env.example .env

up: ## Starts S3, MLflow, Prometheus, Grafana
	docker compose up -d --build

monitoring: ## Starts Prometheus and Grafana only (dev): http://localhost:9090, http://localhost:3000
	docker compose up -d prometheus grafana

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

cv: ## 5-fold cross-validation by ride (measurement only, no promotion) -> reports/cv.json
	uv run python -m ride_classifier.cross_validate

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

# --- Docker images (step 6) ----------------------------------------------------
image: export ## Builds the API image with model @staging: $(IMAGE_API):$(VERSION) + :latest
	docker build -t $(IMAGE_API):$(VERSION) -t $(IMAGE_API):latest .

image-webapp: ## Builds the web app image: $(IMAGE_WEBAPP):$(VERSION) + :latest
	docker build -f docker/webapp/Dockerfile -t $(IMAGE_WEBAPP):$(VERSION) -t $(IMAGE_WEBAPP):latest .

app: ## Runs API + web app containers with the dev stack (http://localhost:8000/docs, http://localhost:8501)
	docker compose --profile app up -d

PLATFORMS ?= linux/amd64,linux/arm64
BUILDER := ride-classifier-multiarch

# Multi-platform builds need a docker-container builder; the result goes straight to Docker Hub (not loaded locally)
push: ## Builds both images for amd64 + arm64 and pushes them to Docker Hub (docker login + DOCKERHUB_USER in .env)
	docker buildx inspect $(BUILDER) > /dev/null 2>&1 || docker buildx create --name $(BUILDER) --driver docker-container
	docker buildx build --builder $(BUILDER) --platform $(PLATFORMS) --push \
	  -t $(IMAGE_API):$(VERSION) -t $(IMAGE_API):latest .
	docker buildx build --builder $(BUILDER) --platform $(PLATFORMS) --push -f docker/webapp/Dockerfile \
	  -t $(IMAGE_WEBAPP):$(VERSION) -t $(IMAGE_WEBAPP):latest .

# --- Kubernetes "prod" (step 7) ------------------------------------------------
CLUSTER := ride-classifier
K8S_NS := ride-classifier
KUBECTL := kubectl --context kind-$(CLUSTER)   # never deploy to another cluster by accident
TAG ?= latest                                    # image tag to deploy (the CD workflow passes the git sha)
PROD_MLFLOW := http://localhost:5050

kind-up: ## Creates the kind cluster (ports: API 8080, web app 8081, MLflow 5050)
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --name $(CLUSTER) --config k8s/kind-config.yaml

k8s-secrets: ## Creates k8s/secrets/ (git-ignored) from k8s/secrets.example if missing
	mkdir -p k8s/secrets
	cp -n k8s/secrets.example/secrets.env k8s/secrets.example/s3.json k8s/secrets/ || true

kind-load: ## Loads local images $(TAG) into kind (faster than pulling 2.5 GB from Docker Hub)
	kind load docker-image ride-classifier-mlflow:3.18.0 $(IMAGE_API):$(TAG) $(IMAGE_WEBAPP):$(TAG) --name $(CLUSTER)

k8s-deploy: k8s-secrets ## Deploys everything with images $(TAG) and waits for the rollouts
	$(KUBECTL) kustomize k8s/ \
	  | sed -e 's#$(IMAGE_API):latest#$(IMAGE_API):$(TAG)#' -e 's#$(IMAGE_WEBAPP):latest#$(IMAGE_WEBAPP):$(TAG)#' \
	  | $(KUBECTL) apply -f -
	$(KUBECTL) -n $(K8S_NS) rollout status deploy/s3 deploy/postgres deploy/mlflow --timeout=300s
	$(KUBECTL) -n $(K8S_NS) rollout status deploy/ride-classifier-api deploy/ride-classifier-webapp --timeout=300s

k8s-status: ## Pods, services, volumes of the prod namespace
	$(KUBECTL) -n $(K8S_NS) get pods,svc,pvc -o wide

e2e: ## End-to-end tests against prod (Kubernetes)
	E2E_API_URL=http://localhost:8080 E2E_WEBAPP_URL=http://localhost:8081 uv run pytest -v tests/test_e2e.py

s3-forward: ## Exposes prod S3 on localhost:8333 (DVC remote) until Ctrl-C. Stop the compose stack first
	$(KUBECTL) -n $(K8S_NS) port-forward svc/s3 8333:8333

kind-down: ## Drops the cluster (all prod data is lost). To free RAM, use `make pause` instead
	kind delete cluster --name $(CLUSTER)

# --- Laptop mode -------------------------------------------------------------------
pause: ## Frees RAM when not working: stops tunnels, runner, dev stack and the cluster (all data KEPT)
	-pkill -f "port-forward svc/"
	-pkill -f Runner.Listener
	-docker compose down
	-docker stop $(CLUSTER)-control-plane
	@echo "Paused. Now quit Docker Desktop to free its memory (resume with: make resume)."

resume: ## Restarts the paused cluster (start Docker Desktop first) and the GitHub runner if installed
	@docker info > /dev/null 2>&1 || { echo "Docker is not running: start Docker Desktop first."; exit 1; }
	docker start $(CLUSTER)-control-plane
	$(KUBECTL) -n $(K8S_NS) rollout status deploy/ride-classifier-api deploy/mlflow --timeout=300s
	@if [ -x "$$HOME/actions-runner/run.sh" ] && ! pgrep -f Runner.Listener > /dev/null; then \
	  cd "$$HOME/actions-runner" && nohup ./run.sh > runner.log 2>&1 & echo "GitHub runner started."; fi
	@echo "Prod is back: API http://localhost:8080/docs, web app http://localhost:8081, MLflow http://localhost:5050"

# --- Monitoring (step 9) ----------------------------------------------------------
API_URL ?= http://localhost:8080

monitoring-forward: ## Prod Grafana on http://localhost:3030 and Prometheus on http://localhost:9091 until Ctrl-C
	$(KUBECTL) -n $(K8S_NS) port-forward svc/grafana 3030:3000 & \
	$(KUBECTL) -n $(K8S_NS) port-forward svc/prometheus 9091:9090; kill %1

simulate: ## Replays 50 test rides (+ feedback) against $(API_URL) to feed the dashboard
	uv run python -m ride_classifier.simulate --api $(API_URL) --n 50

simulate-unit-bug: ## Incident: accelerometer sent in g instead of m/s² (alerts InputGravityShift, accuracy)
	uv run python -m ride_classifier.simulate --api $(API_URL) --n 50 --unit-bug

simulate-dropouts: ## Incident: 20 % of the samples lost by the sensor (alert SensorDropoutsHigh)
	uv run python -m ride_classifier.simulate --api $(API_URL) --n 50 --dropouts 0.2

# --- AI Infrastructure (ai-infrastructure/) -----------------------------------------
infra-env: ## Records hardware, versions and the model/test reference (ai-infrastructure/benchmarks/environment.*)
	uv run python -m ride_classifier.environment

benchmark: ## Inference benchmark of ./model (threads 4, 1, 8; batches 1/4/16/75 rides) -> benchmarks/raw/*.csv
	for t in 4 1 8; do uv run python -m ride_classifier.benchmark run --threads $$t; done
	uv run python -m ride_classifier.benchmark summarize

onnx: ## Step 3: export ./model to ONNX (fp32 + int8), benchmark both (threads 4, 1), serving memory per runtime
	uv run python -m ride_classifier.optimize export
	for t in 4 1; do for r in onnxruntime onnxruntime-int8; do \
	  uv run python -m ride_classifier.benchmark run --runtime $$r --threads $$t --experiment onnx-inference \
	    --out ai-infrastructure/optimization/raw/onnx-inference.csv; done; done
	uv run python -m ride_classifier.optimize memory --threads 1
	uv run python -m ride_classifier.benchmark summarize --raw ai-infrastructure/optimization/raw/onnx-inference.csv \
	  --out ai-infrastructure/optimization/onnx-results.csv

train-benchmark: ## Training benchmark on the laptop CPU (3 epochs x 3 repeats) -> benchmarks/raw/training.csv
	uv run python -m ride_classifier.train_benchmark
	uv run python -m ride_classifier.benchmark summarize --raw ai-infrastructure/benchmarks/raw/training.csv \
	  --out ai-infrastructure/benchmarks/training-results.csv

colab-bundle: ## colab_bundle.zip for the Colab GPU notebook: train/test splits + model v1 (private, git-ignored)
	zip -q -r colab_bundle.zip data/processed/train.parquet data/processed/test.parquet model
	@ls -lh colab_bundle.zip | awk '{print $$NF, $$5}'

infra-diagrams: ## Renders ai-infrastructure/architecture/*.dot to PNG (needs Graphviz: brew install graphviz)
	for f in ai-infrastructure/architecture/*.dot; do dot -Tpng -Gdpi=150 "$$f" -o "$${f%.dot}.png"; done

# --- Code quality ---------------------------------------------------------------
lint: ## Ruff
	uv run ruff check --fix . && uv run ruff format --check .

test: ## Tests
	uv run pytest -q

.PHONY: help setup up monitoring down pull data train evaluate pipeline dag export api webapp load image image-webapp
.PHONY: app push kind-up k8s-secrets kind-load k8s-deploy k8s-status e2e s3-forward kind-down
.PHONY: pause resume monitoring-forward simulate simulate-unit-bug simulate-dropouts lint test
.PHONY: infra-env benchmark onnx train-benchmark colab-bundle infra-diagrams
