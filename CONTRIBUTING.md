# Contributing

## Workflow

1. Never push to `main` directly. It is protected: changes land through pull requests only.
2. Create a branch from an up-to-date `main`:
   ```bash
   git switch main && git pull
   git switch -c <type>/<short-description>
   ```
   Branch types: `feat/`, `fix/`, `chore/`, `docs/`, `test/`, `ci/`.
3. Commit small, focused changes using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(api): add /predict endpoint`, `fix(data): drop rides shorter than one window`.
4. Run `make lint test` before pushing. `make setup` also installs a pre-commit hook that runs ruff (lint + format)
   on every commit, with the same version as CI. To run it on the whole repo: `uv run pre-commit run --all-files`.
5. Open a PR using the template. CI must be green and one teammate must approve it before merging.

## Code conventions (same as the course labs)

- One module per pipeline step in `src/ride_classifier/` (`data.py`, `features.py`, `train.py`, `evaluate.py`, ...),
  run with `python -m ride_classifier.<module>` and exposed as a `make` target.
- Parameters live in `params.yaml` (tracked by DVC, logged to MLflow). Constants and env variables are read in `config.py`.
- Data is versioned with DVC (S3 remote), never committed to git.
- Tests go in `tests/`. Shared fixtures go in `tests/conftest.py`, and tests use small fake data with the real schema.
