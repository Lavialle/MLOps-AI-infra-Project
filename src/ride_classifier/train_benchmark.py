"""Training benchmark: time and memory of the training workload (AI Infrastructure, step 4).

    python -m ride_classifier.train_benchmark                       # CPU, threads = torch default
    python -m ride_classifier.train_benchmark --device cuda         # Colab GPU (see the notebook)
    python -m ride_classifier.train_benchmark --threads 8 --epochs 3 --repeats 3

Same training step as ride_classifier.train (CNN, class-weighted cross-entropy, Adam, batch size and
learning rate from params.yaml), but a FIXED number of epochs without early stopping, so CPU and GPU do
exactly the same work. Protocol of the course labs: warm-up steps not timed, CUDA synchronised before
and after timing, --repeats fresh models, medians reported by `benchmark summarize --raw <file>`.

Measured: data preparation time, seconds per epoch, windows per second, peak memory (process RSS on
CPU; torch.cuda.max_memory_allocated on GPU, i.e. PyTorch allocations, not total device occupancy).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ride_classifier.benchmark import OUT_DIR, RAW_COLUMNS, peak_rss_mb
from ride_classifier.config import ROOT, load_params
from ride_classifier.environment import collect
from ride_classifier.features import ChannelScaler, make_windows
from ride_classifier.model import build_net
from ride_classifier.train import class_weights


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def prepare(train_path: Path, p_feat: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Windows of the train split, as train.fit builds them. Returns X, y and the preparation time."""
    t0 = time.perf_counter()
    train = pd.read_parquet(train_path)
    scaler = ChannelScaler.fit(train)
    X, y, _ = make_windows(train, scaler, p_feat["window_size"], p_feat["window_stride"])
    return torch.from_numpy(X), torch.from_numpy(y).long(), time.perf_counter() - t0


def measure(X, y, p_train: dict, device: str, epochs: int, warmup_steps: int, seed: int) -> dict:
    """One fresh model trained for `epochs` epochs; warm-up steps on a throwaway model are not timed."""

    def new_model():
        torch.manual_seed(seed)
        net = build_net(p_train["dropout"]).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y.numpy(), p_train["class_weighting"]).to(device))
        return net, loss_fn, torch.optim.Adam(net.parameters(), lr=p_train["learning_rate"])

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=p_train["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def step(net, loss_fn, opt, xb, yb):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(xb.to(device, non_blocking=True)), yb.to(device, non_blocking=True))
        loss.backward()
        opt.step()
        return loss

    net, loss_fn, opt = new_model()  # warm-up: kernels, allocator, thread pools
    for i, (xb, yb) in enumerate(loader):
        if i >= warmup_steps:
            break
        step(net, loss_fn, opt, xb, yb)
    sync(device)

    net, loss_fn, opt = new_model()
    net.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = step(net, loss_fn, opt, xb, yb)
    sync(device)
    elapsed = time.perf_counter() - t0

    out = {
        "s_per_epoch": elapsed / epochs,
        "windows_per_s": len(X) * epochs / elapsed,
        "final_loss": float(loss.detach()),
        "peak_rss_mb": peak_rss_mb(),
    }
    if device == "cuda":
        out["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def main() -> None:
    params = load_params()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(ROOT / params["data"]["train_path"]))
    p.add_argument("--model", default=str(ROOT / "model"), help="only for the environment record")
    p.add_argument("--test", default=str(ROOT / params["data"]["test_path"]), help="only for the environment record")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    p.add_argument("--epochs", type=int, default=3, help="timed epochs per repeat")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--experiment", default="training")
    p.add_argument("--out", default=str(OUT_DIR / "raw" / "training.csv"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available: on Colab, Runtime > Change runtime type > GPU")
    env = collect(Path(args.model), Path(args.test))
    p_train = params["train"]

    X, y, prep_s = prepare(Path(args.train), params["features"])
    print(f"{len(X):,} windows {tuple(X.shape[1:])}, prepared in {prep_s:.1f} s; {args.device}, {args.threads} threads")
    rows = [{"run": 0, "metric": "data_prep_s", "value": prep_s}, {"run": 0, "metric": "windows", "value": len(X)}]
    for run in range(1, args.repeats + 1):
        m = measure(X, y, p_train, args.device, args.epochs, args.warmup_steps, seed=p_train["random_state"])
        rows += [{"run": run, "metric": k, "value": v} for k, v in m.items()]
        print(
            f"  run {run}: {m['s_per_epoch']:.2f} s/epoch, {m['windows_per_s']:,.0f} windows/s, "
            f"RSS {m['peak_rss_mb']:.0f} MB" + (f", CUDA {m['peak_cuda_mb']:.0f} MB" if "peak_cuda_mb" in m else "")
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if new:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "experiment": args.experiment, "kind": "measured", "runtime": "pytorch", "device": args.device,
                "threads": args.threads, "batch_rides": p_train["batch_size"],  # training: windows per step
                "model_sha": env["reference"]["weights_sha256"], "test_sha": env["reference"]["test_sha256"],
                "commit": env["git_commit"], **r,
            })  # fmt: skip
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
