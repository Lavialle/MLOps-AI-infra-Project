"""ONNX optimization: same predictions as PyTorch, and an ONNX-only serving process without torch."""

import json
import subprocess
import sys
from argparse import Namespace

import mlflow
import numpy as np
import pandas as pd
import pytest

from ride_classifier.benchmark import Pipeline, make_pipeline, parity, pytorch_runner
from ride_classifier.model import INPUT_COLUMNS
from ride_classifier.optimize import export, onnx_runner


@pytest.fixture(scope="module")
def onnx_dir(model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx")
    export(model_dir, out)
    return out


def test_export_writes_graphs_and_preprocessing(onnx_dir):
    info = json.loads((onnx_dir / "preprocessing.json").read_text())
    assert {"model.onnx", "model.int8.onnx"} == set(info["files"])
    assert info["scaler"]["mean"] and info["window_size"] > 0
    assert info["files"]["model.int8.onnx"]["bytes"] < info["files"]["model.onnx"]["bytes"]


def test_onnx_fp32_matches_pytorch(model_dir, onnx_dir, small_data):
    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    served = wrapper.predict(None, small_data[INPUT_COLUMNS])
    ours, _ = Pipeline(wrapper, onnx_runner(onnx_dir / "model.onnx", threads=1))(small_data)
    p = parity(ours, served)
    assert p["max_abs_diff_vs_pytorch"] < 1e-4 and p["agreement_vs_pytorch"] == 1.0


def test_onnx_int8_stays_close(model_dir, onnx_dir, small_data):
    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    served = wrapper.predict(None, small_data[INPUT_COLUMNS])
    ours, _ = Pipeline(wrapper, onnx_runner(onnx_dir / "model.int8.onnx", threads=1))(small_data)
    np.testing.assert_allclose(ours[["Safe"]].sum().to_numpy(), served[["Safe"]].sum().to_numpy(), rtol=0.2)


def test_make_pipeline_picks_the_runtime(model_dir, onnx_dir):
    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    args = Namespace(runtime="onnxruntime", onnx_dir=str(onnx_dir), threads=1, device="cpu")
    X = np.zeros((3, 6, wrapper.window_size), dtype=np.float32)
    np.testing.assert_allclose(make_pipeline(args, wrapper).runner(X), pytorch_runner(wrapper.net)(X), atol=1e-5)


def test_onnx_serving_process_never_imports_torch(model_dir, onnx_dir, small_data, tmp_path):
    test_path = tmp_path / "test.parquet"
    small_data.to_parquet(test_path, index=False)
    out = subprocess.run(
        [sys.executable, "-m", "ride_classifier.optimize", "_serve", "--runtime", "onnxruntime",
         "--model", str(model_dir), "--onnx-dir", str(onnx_dir), "--test", str(test_path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[-1]  # fmt: skip
    result = json.loads(out)
    assert result["torch_imported"] is False
    assert result["rides"] == pd.read_parquet(test_path)["ride_id"].nunique()
