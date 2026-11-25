"""ONNX Runtime optimization of the classifier (AI Infrastructure, step 3).

    python -m ride_classifier.optimize export   # ./model -> model-onnx/ (fp32 + int8 graphs, preprocessing.json)
    python -m ride_classifier.optimize memory   # peak RSS of a fresh serving process, per runtime, 3 repeats

Latency, throughput and prediction quality are measured by the SAME harness as the baseline:
    python -m ride_classifier.benchmark run --runtime onnxruntime --threads 4 --experiment optimization ...

Only the network changes: the exported graph is the CNN + softmax (windows -> probabilities). Gap filling,
scaling, windowing and per-ride aggregation stay the identical Python code, and their parameters (the scaler
fitted on train, window size and stride) are written next to the graph in preprocessing.json. That is all an
ONNX-only serving process needs: no PyTorch, no MLflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from ride_classifier.config import ROOT, load_params

ONNX_DIR = ROOT / "model-onnx"
OUT_DIR = ROOT / "ai-infrastructure" / "optimization"
RUNTIMES = {"pytorch": None, "onnxruntime": "model.onnx", "onnxruntime-int8": "model.int8.onnx"}


def export(model_dir: Path, out_dir: Path, opset: int = 17) -> dict:
    """CNN + softmax -> ONNX (dynamic batch), then an INT8 dynamically quantized copy."""
    import mlflow
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from ride_classifier.environment import sha256

    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    graph = torch.nn.Sequential(wrapper.net, torch.nn.Softmax(dim=1)).eval()
    dummy = torch.zeros(2, 6, wrapper.window_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / "model.onnx", out_dir / "model.int8.onnx"
    torch.onnx.export(
        graph,
        (dummy,),
        fp32,
        input_names=["windows"],
        output_names=["probabilities"],
        dynamic_axes={"windows": {0: "n"}, "probabilities": {0: "n"}},
        opset_version=opset,
        dynamo=False,  # classic exporter: a static CNN, no data-dependent control flow
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    meta_src = model_dir / "model_meta.json"
    info = {
        "source_model": json.loads(meta_src.read_text()) if meta_src.exists() else {},
        "source_weights_sha256": sha256(model_dir / "artifacts" / "weights.pt")[:16],
        "scaler": wrapper.scaler,
        "window_size": wrapper.window_size,
        "window_stride": wrapper.window_stride,
        "opset": opset,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)[:16]} for p in (fp32, int8)},
    }
    (out_dir / "preprocessing.json").write_text(json.dumps(info, indent=2))
    return info


def ort_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def onnx_runner(path: Path, threads: int, batch_windows: int = 512):
    """Same contract as benchmark.pytorch_runner: windows (n, 6, W) float32 -> probabilities (n, 5)."""
    import numpy as np

    session = ort_session(path, threads)

    def run(X):
        return np.concatenate(
            [session.run(None, {"windows": X[i : i + batch_windows]})[0] for i in range(0, len(X), batch_windows)]
        )

    return run


# --- Memory of a serving process ------------------------------------------------------


def _serve_once(runtime: str, model_dir: Path, onnx_dir: Path, test_path: Path, threads: int) -> dict:
    """Load the model like a serving process, score the whole test split, report peak RSS.

    The ONNX path deliberately imports neither torch nor mlflow, as an ONNX-only API image would.
    """
    import pandas as pd

    from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, TIME
    from ride_classifier.data import fill_gaps
    from ride_classifier.environment import peak_rss_mb
    from ride_classifier.features import ChannelScaler, aggregate_by_ride, make_windows

    INPUT_COLUMNS = [RIDE_ID, TIME, *CHANNELS]  # not imported from model.py: that module imports torch
    test = pd.read_parquet(test_path)
    if runtime == "pytorch":
        import mlflow
        import torch

        torch.set_num_threads(threads)
        model = mlflow.pyfunc.load_model(str(model_dir))
        pred = model.predict(test[INPUT_COLUMNS])
    else:
        info = json.loads((onnx_dir / "preprocessing.json").read_text())
        run = onnx_runner(onnx_dir / RUNTIMES[runtime], threads)
        X, _, ids = make_windows(
            fill_gaps(test[INPUT_COLUMNS]), ChannelScaler.from_dict(info["scaler"]),
            info["window_size"], info["window_stride"],
        )  # fmt: skip
        proba = aggregate_by_ride(run(X), ids)
        pred = proba.assign(prediction=[CLASSES[i] for i in proba.to_numpy().argmax(axis=1)])
    return {"peak_rss_mb": peak_rss_mb(), "torch_imported": "torch" in sys.modules, "rides": len(pred)}


def memory(args) -> list[dict]:
    rows = []
    for runtime in RUNTIMES:
        values = []
        for repeat in range(1, args.repeats + 1):
            out = subprocess.run(
                [sys.executable, "-m", "ride_classifier.optimize", "_serve", "--runtime", runtime,
                 "--model", args.model, "--onnx-dir", args.onnx_dir, "--test", args.test,
                 "--threads", str(args.threads)],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[-1]  # fmt: skip
            result = json.loads(out)
            values.append(result["peak_rss_mb"])
            rows.append({"runtime": runtime, "run": repeat, **result})
        print(f"  {runtime:17s} peak RSS median {statistics.median(values):6.0f} MB "
              f"(torch imported: {result['torch_imported']})")  # fmt: skip
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    params = load_params()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", default=str(ROOT / "model"))
    e.add_argument("--out", default=str(ONNX_DIR))
    for name in ("memory", "_serve"):
        m = sub.add_parser(name)
        m.add_argument("--model", default=str(ROOT / "model"))
        m.add_argument("--onnx-dir", default=str(ONNX_DIR))
        m.add_argument("--test", default=str(ROOT / params["data"]["test_path"]))
        m.add_argument("--threads", type=int, default=1)
        m.add_argument("--repeats", type=int, default=3)
        m.add_argument("--runtime", default="pytorch", choices=list(RUNTIMES))
        m.add_argument("--out", default=str(OUT_DIR / "raw" / "serving-memory.csv"))
    args = p.parse_args()

    if args.cmd == "export":
        info = export(Path(args.model), Path(args.out))
        for name, f in info["files"].items():
            print(f"{name:18s} {f['bytes'] / 1024:7.1f} KB  sha {f['sha256']}")
    elif args.cmd == "memory":
        memory(args)
    else:
        print(
            json.dumps(_serve_once(args.runtime, Path(args.model), Path(args.onnx_dir), Path(args.test), args.threads))
        )


if __name__ == "__main__":
    main()
