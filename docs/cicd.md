# CI/CD (GitHub Actions)

| Workflow | Runner | Trigger | Steps |
|----------|--------|---------|-------|
| [`ci.yml`](../.github/workflows/ci.yml) | GitHub-hosted (ubuntu) | every push and pull request | `uv sync --locked` → ruff lint + format check → pytest (unit + integration; e2e skipped) |
| [`cd.yml`](../.github/workflows/cd.yml) | **self-hosted** (machine running the kind cluster) | push to `main` (code changes), manual, or called by `retrain.yml` | export `@staging` from prod MLflow → build API + web app images tagged with the commit → push to Docker Hub → `kind load` → rolling update → **end-to-end tests on prod** → automatic `rollout undo` if they fail → clean up old images |
| [`retrain.yml`](../.github/workflows/retrain.yml) | **self-hosted** | manual (with a reason) or every Monday | `dvc pull` from prod S3 → split → train (prod MLflow) → evaluation gate → `@staging` → calls `cd.yml` |

```
push / PR ─────────▶ CI (GitHub cloud): lint + tests
merge to main ─────▶ CD (self-hosted): model @staging → images :<sha> → Docker Hub → kind → e2e tests (→ rollback)
Run "Retrain" ─────▶ data (DVC) → train → gate ──passed──▶ @staging ──▶ CD
                                           └─failed──▶ stop: production untouched
```

## Why a self-hosted runner

The production cluster is a local kind cluster, so GitHub's cloud runners cannot reach it. A runner installed on
the machine that hosts the cluster can: it reaches prod MLflow on `:5050`, the API on `:8080`, and `kubectl`
through the `kind-ride-classifier` context.

**Security.** The repository is public, so `cd.yml` and `retrain.yml` never run on `pull_request`. Code from a
fork can therefore never run on the self-hosted machine. Keep the default setting *Settings → Actions → General →
Require approval for all outside collaborators*.

## Setting up the runner (once)

Do this on the Mac that runs the cluster. Keep the runner **outside** iCloud-synced folders such as Desktop or Documents.

1. GitHub → repository **Settings → Actions → Runners → New self-hosted runner** → macOS, ARM64.
2. Run the commands GitHub shows, in `~/actions-runner`:
   ```bash
   mkdir -p ~/actions-runner && cd ~/actions-runner
   # curl -o actions-runner-osx-arm64-<version>.tar.gz -L <url shown by GitHub>
   # tar xzf ./actions-runner-osx-arm64-<version>.tar.gz
   ./config.sh --url https://github.com/<owner>/MLOps-AI-infra-Project --token <token shown by GitHub>
   ```
   Accept the default labels (`self-hosted`, `macOS`, `ARM64`).
3. Start it from a terminal where `uv`, `docker`, `kind` and `kubectl` are on the `PATH`:
   ```bash
   ./run.sh                     # foreground, or:
   ./svc.sh install && ./svc.sh start   # as a background service
   ```
   The runner should show as **Idle** in Settings → Actions → Runners.

## Secrets and variables (Settings → Secrets and variables → Actions)

| Name | Kind | Required | Used for |
|------|------|----------|----------|
| `DOCKERHUB_TOKEN` | secret | recommended | `docker login` before pushing (Docker Hub → Account settings → Personal access tokens, scope *Read & Write*). Without it, the runner uses the machine's existing Docker login |
| `DOCKERHUB_USER` | variable | no (default `nicogil7`) | image namespace |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | secrets | no (default: dev values from `k8s/secrets.example`) | DVC access to prod S3 |

## First run

1. Prod cluster up: `make kind-up image image-webapp kind-load k8s-deploy`.
2. The data must be in prod S3 (once): `make s3-forward` in one terminal, then `uv run dvc push` in another.
3. **Actions → Retrain & redeploy → Run workflow.** It trains, gates and promotes a model in prod MLflow, then deploys it.
4. After that, every merge to `main` redeploys automatically with the current `@staging` model.
