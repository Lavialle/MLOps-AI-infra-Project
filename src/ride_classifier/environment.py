"""Record the environment and the fixed reference of an AI Infrastructure measurement.

    python -m ride_classifier.environment                          # model ./model, test split from params.yaml
    python -m ride_classifier.environment --out ai-infrastructure/benchmarks

Writes environment.json (read by the benchmark scripts, stored next to their raw results)
and environment.md (human-readable table). Run it on every machine that produces numbers
(laptop CPU, Colab GPU...): a measurement without its environment cannot be compared.

The reference is pinned by content hashes, so two results are only compared when they used
the same model weights and the same test rides.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ride_classifier.config import RIDE_ID, ROOT, load_params

PACKAGES = ["torch", "numpy", "pandas", "pyarrow", "scikit-learn", "mlflow", "fastapi", "onnx", "onnxruntime"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except OSError, subprocess.CalledProcessError:
        return None


def hardware() -> dict:
    """CPU, cores and RAM, on macOS (sysctl) or Linux (/proc, e.g. Colab)."""
    info = {
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu": platform.processor() or None,
        "ram_gb": None,
    }
    if sys.platform == "darwin":
        info["cpu"] = _sysctl("machdep.cpu.brand_string") or info["cpu"]
        info["physical_cpus"] = int(_sysctl("hw.physicalcpu") or 0) or None
        mem = _sysctl("hw.memsize")
        info["ram_gb"] = round(int(mem) / 1e9, 1) if mem else None
    elif Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_gb"] = round(int(line.split()[1]) * 1024 / 1e9, 1)
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    return info


def accelerators() -> dict:
    import torch

    info = {
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
    return info


def packages() -> dict:
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def reference(model_dir: Path, test_path: Path) -> dict:
    """The fixed model version and test data every measurement uses, pinned by hash."""
    import pandas as pd

    from ride_classifier.model import build_net

    meta_file = model_dir / "model_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    weights = model_dir / "artifacts" / "weights.pt"
    net = build_net()
    test = pd.read_parquet(test_path, columns=[RIDE_ID])
    return {
        "model_name": meta.get("name"),
        "model_version": meta.get("version"),
        "model_run_id": meta.get("run_id"),
        "model_parameters": sum(p.numel() for p in net.parameters()),
        "weights_bytes": weights.stat().st_size if weights.exists() else None,
        "weights_sha256": sha256(weights)[:16] if weights.exists() else None,
        "model_dir_bytes": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()),
        "test_path": str(test_path.relative_to(ROOT)) if test_path.is_relative_to(ROOT) else str(test_path),
        "test_sha256": sha256(test_path)[:16],
        "test_rides": int(test[RIDE_ID].nunique()),
        "test_rows": len(test),
    }


def collect(model_dir: Path, test_path: Path) -> dict:
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "hardware": hardware(),
        "accelerators": accelerators(),
        "packages": packages(),
        "reference": reference(model_dir, test_path),
    }


def to_markdown(env: dict) -> str:
    lines = [
        "# Measurement environment",
        "",
        f"Recorded {env['recorded_at']} at commit `{env['git_commit']}` by `python -m ride_classifier.environment`.",
        "",
    ]
    for section in ("hardware", "accelerators", "packages", "reference"):
        lines += [f"## {section.capitalize()}", "", "| Item | Value |", "|------|-------|"]
        lines += [f"| {k} | {'–' if v is None else v} |" for k, v in env[section].items()]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "model"), help="exported model directory (make export)")
    parser.add_argument("--test", default=str(ROOT / load_params()["data"]["test_path"]))
    parser.add_argument("--out", default=str(ROOT / "ai-infrastructure" / "benchmarks"))
    args = parser.parse_args()

    env = collect(Path(args.model), Path(args.test))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    (out / "environment.md").write_text(to_markdown(env))
    print(to_markdown(env))


if __name__ == "__main__":
    main()
