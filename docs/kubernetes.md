# Production environment (Kubernetes)

"Prod" runs on a local **kind** cluster, as in the course labs. Every service lives in the namespace
`ride-classifier` and is described by the manifests in [`k8s/`](../k8s/), assembled with kustomize
(`kubectl apply -k k8s/`).

```
                         kind cluster "ride-classifier" — namespace ride-classifier
  localhost:8081  ─▶  webapp (1 pod) ──http://ride-classifier-api──▶ api (2 pods, model baked in)  ◀─ localhost:8080
  localhost:5050  ─▶  mlflow (1 pod) ──▶ postgres (PVC 1 Gi)
                                     └─▶ s3 / SeaweedFS (PVC 5 Gi) ◀── s3-init Job (buckets mlflow, dvc)
```

| Component | Kind | Replicas | Exposed | Probes | Requests → limits |
|-----------|------|----------|---------|--------|-------------------|
| `ride-classifier-api` | Deployment + NodePort | 2 | http://localhost:8080 (`/docs`) | readiness `/ready`, liveness `/health` | 250m / 640Mi → 1 CPU / 1Gi |
| `ride-classifier-webapp` | Deployment + NodePort | 1 | http://localhost:8081 | `/_stcore/health` | 50m / 192Mi → 500m / 512Mi |
| `mlflow` | Deployment + NodePort | 1 | http://localhost:5050 | `/health` | 100m / 384Mi → 1 CPU / 1Gi |
| `postgres` | Deployment + ClusterIP + PVC | 1 | internal | `pg_isready` | 50m / 64Mi → 500m / 256Mi |
| `s3` (SeaweedFS) | Deployment + ClusterIP + PVC | 1 | internal | TCP 8333 | 50m / 128Mi → 500m / 512Mi |
| `s3-init` | Job | – | – | – | creates the `mlflow` and `dvc` buckets (idempotent, retried until S3 is up) |

## Deploy

```bash
make kind-up      # create the cluster (once)
make image image-webapp   # or pull nicogil7/ride-classifier-{api,webapp} from Docker Hub
make kind-load    # copy the images into the kind node
make k8s-deploy   # secrets from k8s/secrets.example (first time) + kubectl apply -k k8s/ + wait for rollouts
make k8s-status   # pods, services, volumes
make kind-down    # delete the cluster
```

Kubernetes has its own MLflow, separate from the docker compose dev MLflow. The CD pipeline (step 8)
trains against it and promotes models there.

## Design choices

- **Zero-downtime rollouts for the API.** It uses `maxUnavailable: 0, maxSurge: 1`: a new pod must pass its
  readiness probe before an old one is removed. A new model therefore ships with
  `kustomize edit set image nicogil7/ride-classifier-api=…:<sha>` followed by `kubectl apply -k k8s/`, and a
  rollback is `kubectl rollout undo deploy/ride-classifier-api`.
- **Readiness differs from liveness.** `/ready` returns 503 until the model is loaded, so a pod without a model
  never receives traffic. `/health` (liveness) stays 200 even when the model is missing, because restarting the
  container would not fix a missing model. The liveness probe starts later and tolerates more failures, since a
  restart is expensive.
- **Resources come from measurements.** The API uses about 550 MiB with the model loaded (`docker stats`) and
  serves 16 rides/s at a p95 of 340 ms per process (Locust). Its memory request is 640Mi and its limit 1Gi:
  going over the memory limit gets the pod OOMKilled, while going over the CPU limit only throttles it.
- **Credentials live in a Secret, never in git.** kustomize's `secretGenerator` reads `k8s/secrets/`, which is
  git-ignored and created from `k8s/secrets.example/` by `make k8s-secrets`. Non-secret settings (endpoints,
  bucket names) live in a ConfigMap.
- **Stateful services get PersistentVolumeClaims** with the `Recreate` strategy, so a ReadWriteOnce volume
  never has two writers.
- **Prometheus annotations** (`prometheus.io/scrape`) on the API pods prepare step 9.
- **Images.** The API and web app use the images published on Docker Hub (`nicogil7/...`). `kind load` copies
  them into the node to avoid pulling 2.5 GB again. The MLflow server image is built locally from `docker/mlflow`.
