"""Download a Model Registry version into ./model, to bake it into the Docker image.

    python -m ride_classifier.export_model                 # staging alias
    python -m ride_classifier.export_model --alias champion

The API image contains the model: a tagged image is a frozen model, and the
Kubernetes pod never needs to reach the MLflow server.

The image is published (Docker Hub): the MLflow input example is removed from the
export, so no data sample ever leaves the private registry, whatever the model version.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

from ride_classifier.config import MODEL_NAME, ROOT


def strip_input_example(model_dir: Path) -> None:
    """Remove the input example files and their reference in MLmodel (not needed to load or serve the model)."""
    mlmodel = model_dir / "MLmodel"
    spec = yaml.safe_load(mlmodel.read_text())
    info = spec.pop("saved_input_example_info", None) or {}
    for name in (info.get("artifact_path"), info.get("serving_input_path"), "input_example.json"):
        if name:
            (model_dir / name).unlink(missing_ok=True)
    (model_dir / "serving_input_example.json").unlink(missing_ok=True)
    mlmodel.write_text(yaml.safe_dump(spec, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="staging")
    parser.add_argument("--out", default=str(ROOT / "model"))
    args = parser.parse_args()

    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, args.alias)
    shutil.rmtree(args.out, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{mv.version}", dst_path=args.out)
    strip_input_example(Path(args.out))
    meta = {"name": MODEL_NAME, "version": mv.version, "alias": args.alias, "run_id": mv.run_id}
    with open(f"{args.out}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{MODEL_NAME} v{mv.version} (@{args.alias}) -> {args.out}")


if __name__ == "__main__":
    main()
