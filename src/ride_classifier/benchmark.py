"""Inference benchmark of the classifier, without HTTP (AI Infrastructure, step 2).

    python -m ride_classifier.benchmark run                       # PyTorch, CPU, batches 1/4/16/75 rides
    python -m ride_classifier.benchmark run --threads 1 --repeats 5
    python -m ride_classifier.benchmark summarize                 # medians -> benchmarks/baseline-results.csv

What one "request" does is exactly what the API does for a call: the same functions as
RideClassifier.predict (checked at startup), timed in three phases:

    preprocessing   gap filling + scaling + windowing         (pandas / NumPy)
    network         CNN forward pass on all windows, softmax   (the "runner": PyTorch, or ONNX Runtime with --runtime)
    aggregation     mean of the window probabilities per ride

Protocol (ai-infrastructure/README.md): fixed model and test split (hashes from environment.json),
warm-up requests not timed, --repeats timed runs per batch size, one CSV row per run and metric,
medians reported by `summarize`. Quality is measured too: macro F1 on the whole test split.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from ride_classifier.config import CLASSES, RIDE_ID, ROOT, load_params
from ride_classifier.data import fill_gaps
from ride_classifier.environment import collect, peak_rss_mb
from ride_classifier.features import ChannelScaler, aggregate_by_ride, make_windows
from ride_classifier.model import INPUT_COLUMNS, ride_scores

OUT_DIR = ROOT / "ai-infrastructure" / "benchmarks"
RAW_COLUMNS = [
    "experiment", "kind", "runtime", "device", "threads", "batch_rides", "run",
    "metric", "value", "model_sha", "test_sha", "commit",
]  # fmt: skip

Runner = Callable[[np.ndarray], np.ndarray]  # windows (n, 6, 256) float32 -> probabilities (n, 5)


def pytorch_runner(net: torch.nn.Module, device: str = "cpu", batch_windows: int = 512) -> Runner:
    """The serving runtime of v1: eager PyTorch, inference mode, softmax."""
    net = net.to(device).eval()

    @torch.inference_mode()
    def run(X: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch_windows):
            logits = net(torch.from_numpy(X[i : i + batch_windows]).to(device))
            out.append(torch.softmax(logits, dim=1).cpu())
        if device == "cuda":
            torch.cuda.synchronize()  # GPU calls are asynchronous: time the real work, not the launch
        return torch.cat(out).numpy()

    return run


class Pipeline:
    """RideClassifier.predict split into timed phases, with a swappable network runner."""

    def __init__(self, wrapper, runner: Runner):
        self.scaler = ChannelScaler.from_dict(wrapper.scaler)
        self.window_size, self.stride = wrapper.window_size, wrapper.window_stride
        self.runner = runner

    def __call__(self, rides: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        t0 = time.perf_counter()
        X, _, ride_ids = make_windows(fill_gaps(rides[INPUT_COLUMNS]), self.scaler, self.window_size, self.stride)
        t1 = time.perf_counter()
        proba = self.runner(X)
        t2 = time.perf_counter()
        out = aggregate_by_ride(proba, ride_ids)
        out.insert(0, "prediction", [CLASSES[i] for i in out.to_numpy().argmax(axis=1)])
        t3 = time.perf_counter()
        phases = {"preprocess_s": t1 - t0, "network_s": t2 - t1, "aggregate_s": t3 - t2, "windows": len(X)}
        return out.reset_index(), phases


def requests_for(rides: list[pd.DataFrame], batch: int, min_requests: int) -> list[pd.DataFrame]:
    """Consecutive groups of `batch` rides, cycled until at least `min_requests` requests."""
    groups = [pd.concat(rides[i : i + batch]) for i in range(0, len(rides) - batch + 1, batch)]
    cycles = -(-min_requests // len(groups))  # ceil
    return groups * cycles


def benchmark(
    pipeline: Pipeline,
    test: pd.DataFrame,
    batches: list[int],
    repeats: int,
    warmup: int,
    min_requests: int,
) -> list[dict]:
    """Rows (batch_rides, run, metric, value) for every batch size and timed run."""
    rides = [ride for _, ride in test.groupby(RIDE_ID, sort=True)]
    rows = []
    for batch in batches:
        reqs = requests_for(rides, batch, min_requests)
        for req in reqs[:warmup]:
            pipeline(req)  # warm-up: allocations, caches, lazy init; never timed
        for run in range(1, repeats + 1):
            latencies, phases = [], {"preprocess_s": 0.0, "network_s": 0.0, "aggregate_s": 0.0, "windows": 0}
            for req in reqs:
                start = time.perf_counter()
                _, p = pipeline(req)
                latencies.append(time.perf_counter() - start)
                for k in phases:
                    phases[k] += p[k]
            total, n_rides = sum(latencies), batch * len(reqs)
            ms = np.array(latencies) * 1000
            metrics = {
                "latency_mean_ms": ms.mean(),
                "latency_p50_ms": np.percentile(ms, 50),
                "latency_p95_ms": np.percentile(ms, 95),
                "rides_per_s": n_rides / total,
                "windows_per_s": phases["windows"] / total,
                "preprocess_ms_per_ride": phases["preprocess_s"] * 1000 / n_rides,
                "network_ms_per_ride": phases["network_s"] * 1000 / n_rides,
                "aggregate_ms_per_ride": phases["aggregate_s"] * 1000 / n_rides,
                "preprocess_share": phases["preprocess_s"] / total,
                "network_share": phases["network_s"] / total,
                "requests": len(reqs),
            }
            rows += [{"batch_rides": batch, "run": run, "metric": k, "value": float(v)} for k, v in metrics.items()]
            print(
                f"  batch {batch:3d} run {run}: mean {metrics['latency_mean_ms']:8.1f} ms  "
                f"p95 {metrics['latency_p95_ms']:8.1f} ms  {metrics['rides_per_s']:6.1f} rides/s  "
                f"network {metrics['network_share']:.0%} of the time"
            )
    return rows


def check_equivalence(wrapper, pipeline: Pipeline, test: pd.DataFrame) -> None:
    """The benchmarked pipeline must give the serving model's exact output, or the numbers mean nothing."""
    some = test[test[RIDE_ID].isin(test[RIDE_ID].unique()[:3])]
    ours, _ = pipeline(some)
    served = wrapper.predict(None, some[INPUT_COLUMNS])
    np.testing.assert_allclose(ours[CLASSES].to_numpy(), served[CLASSES].to_numpy(), atol=1e-5)
    assert ours["prediction"].tolist() == served["prediction"].tolist()


def parity(ours: pd.DataFrame, served: pd.DataFrame) -> dict[str, float]:
    """How far an optimized runtime is from the served PyTorch model, on every test ride."""
    a = ours.set_index(RIDE_ID).loc[served[RIDE_ID]]
    return {
        "max_abs_diff_vs_pytorch": float(np.abs(a[CLASSES].to_numpy() - served[CLASSES].to_numpy()).max()),
        "agreement_vs_pytorch": float((a["prediction"].to_numpy() == served["prediction"].to_numpy()).mean()),
    }


def make_pipeline(args, wrapper) -> Pipeline:
    """Same pipeline, network runner chosen by --runtime (ONNX files from `optimize export`)."""
    if args.runtime == "pytorch":
        torch.set_num_threads(args.threads)
        return Pipeline(wrapper, pytorch_runner(wrapper.net, args.device))
    from ride_classifier.optimize import RUNTIMES, onnx_runner

    path = Path(args.onnx_dir) / RUNTIMES[args.runtime]
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python -m ride_classifier.optimize export` first")
    return Pipeline(wrapper, onnx_runner(path, args.threads))


def run(args) -> Path:
    test_path = Path(args.test)
    env = collect(Path(args.model), test_path)
    ref = env["reference"]

    t0 = time.perf_counter()
    wrapper = mlflow.pyfunc.load_model(args.model).unwrap_python_model()
    pipeline = make_pipeline(args, wrapper)
    load_s = time.perf_counter() - t0
    rss_after_load = peak_rss_mb()

    test = pd.read_parquet(test_path)
    served = wrapper.predict(None, test[INPUT_COLUMNS])  # the reference: PyTorch, as in production
    if args.runtime == "pytorch":
        check_equivalence(wrapper, pipeline, test)
    pred, _ = pipeline(test)  # quality on the whole test split, through the benchmarked pipeline
    quality = {**ride_scores(test, pred, "test"), **parity(pred, served)}

    print(f"{args.runtime} on {args.device}, {args.threads} thread(s), model v{ref['model_version']}:")
    rows = benchmark(pipeline, test, args.batches, args.repeats, args.warmup, args.min_requests)
    once = {
        "model_load_s": load_s,
        "rss_after_load_mb": rss_after_load,
        "peak_rss_mb": peak_rss_mb(),
        "macro_f1": quality["test_macro_f1"],
        "accuracy": quality["test_accuracy"],
        "max_abs_diff_vs_pytorch": quality["max_abs_diff_vs_pytorch"],
        "agreement_vs_pytorch": quality["agreement_vs_pytorch"],
    }
    rows += [{"batch_rides": 0, "run": 0, "metric": k, "value": float(v)} for k, v in once.items()]
    print(f"  load {load_s:.2f} s, peak RSS {once['peak_rss_mb']:.0f} MB, macro F1 {once['macro_f1']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if new:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "experiment": args.experiment,
                    "kind": "measured",
                    "runtime": args.runtime,
                    "device": args.device,
                    "threads": args.threads,
                    "model_sha": ref["weights_sha256"],
                    "test_sha": ref["test_sha256"],
                    "commit": env["git_commit"],
                    **r,
                }
            )
    print(f"{len(rows)} rows -> {out}")
    return out


def summarize(raw: Path, out: Path | None = None) -> pd.DataFrame:
    """Median over runs of every metric, per configuration. Refuses to mix models or test sets."""
    df = pd.read_csv(raw)
    for col in ("model_sha", "test_sha"):
        if df.groupby(["experiment", "runtime"])[col].nunique().max() > 1:
            raise SystemExit(f"{raw}: several {col} values in one experiment: results are not comparable")
    keys = ["experiment", "runtime", "device", "threads", "batch_rides", "metric"]
    summary = df.groupby(keys)["value"].agg(median="median", runs="count").reset_index()
    if out:
        summary.to_csv(out, index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark and append raw rows")
    r.add_argument("--model", default=str(ROOT / "model"))
    r.add_argument("--test", default=str(ROOT / load_params()["data"]["test_path"]))
    r.add_argument("--runtime", default="pytorch", choices=["pytorch", "onnxruntime", "onnxruntime-int8"])
    r.add_argument("--onnx-dir", default=str(ROOT / "model-onnx"), help="ONNX files from `optimize export`")
    r.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    r.add_argument("--threads", type=int, default=torch.get_num_threads())
    r.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 75], help="rides per request")
    r.add_argument("--repeats", type=int, default=5, help="timed runs per batch size")
    r.add_argument("--warmup", type=int, default=3, help="untimed requests per batch size")
    r.add_argument("--min-requests", type=int, default=20, help="timed requests per run (test rides are cycled)")
    r.add_argument("--experiment", default="baseline-inference")
    r.add_argument("--out", default=str(OUT_DIR / "raw" / "baseline-inference.csv"))
    s = sub.add_parser("summarize", help="medians per configuration")
    s.add_argument("--raw", default=str(OUT_DIR / "raw" / "baseline-inference.csv"))
    s.add_argument("--out", default=str(OUT_DIR / "baseline-results.csv"))
    args = p.parse_args()

    if args.cmd == "run":
        run(args)
    else:
        summary = summarize(Path(args.raw), Path(args.out))
        print(summary.pivot_table(index=["runtime", "threads", "batch_rides"], columns="metric", values="median"))


if __name__ == "__main__":
    main()
