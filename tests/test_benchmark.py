"""Benchmark harness: it measures the serving pipeline (not something else) and writes comparable raw rows."""

from argparse import Namespace

import mlflow
import pandas as pd
import pytest
from conftest import make_fake_rides

from ride_classifier.benchmark import (
    Pipeline,
    check_equivalence,
    pytorch_runner,
    requests_for,
    run,
    summarize,
)


@pytest.fixture
def test_file(small_data, tmp_path):
    path = tmp_path / "test.parquet"
    small_data.to_parquet(path, index=False)
    return path


def test_requests_cycle_rides_until_enough_requests(small_data):
    rides = [r for _, r in small_data.groupby("ride_id")]
    reqs = requests_for(rides, batch=4, min_requests=20)
    assert len(reqs) >= 20
    assert all(r["ride_id"].nunique() == 4 for r in reqs)


def test_pipeline_matches_the_served_model(model_dir, small_data):
    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    pipeline = Pipeline(wrapper, pytorch_runner(wrapper.net))
    check_equivalence(wrapper, pipeline, small_data)  # raises if the benchmark drifts from serving
    _, phases = pipeline(small_data)
    assert phases["windows"] > 0 and min(phases[k] for k in ("preprocess_s", "network_s", "aggregate_s")) >= 0


def test_run_writes_raw_rows_and_summary(model_dir, test_file, tmp_path):
    out = tmp_path / "raw.csv"
    args = Namespace(
        model=str(model_dir), test=str(test_file), runtime="pytorch", device="cpu", threads=1,
        batches=[1, 4], repeats=2, warmup=1, min_requests=3, experiment="unit", out=str(out),
    )  # fmt: skip
    run(args)
    raw = pd.read_csv(out)
    assert set(raw["kind"]) == {"measured"}
    assert {"latency_p95_ms", "rides_per_s", "network_share", "macro_f1", "peak_rss_mb"} <= set(raw["metric"])
    assert raw.query("metric == 'latency_mean_ms'")["run"].nunique() == 2  # warm-up rows never recorded

    summary = summarize(out, tmp_path / "summary.csv")
    p95 = summary.query("metric == 'latency_p95_ms'")
    assert sorted(p95["batch_rides"]) == [1, 4] and (p95["runs"] == 2).all()


def test_summary_refuses_to_mix_models(model_dir, test_file, tmp_path):
    out = tmp_path / "raw.csv"
    args = Namespace(
        model=str(model_dir), test=str(test_file), runtime="pytorch", device="cpu", threads=1,
        batches=[1], repeats=1, warmup=0, min_requests=1, experiment="unit", out=str(out),
    )  # fmt: skip
    run(args)
    raw = pd.read_csv(out)
    raw.loc[raw.index[0], "model_sha"] = "another-model"
    raw.to_csv(out, index=False)
    with pytest.raises(SystemExit, match="not comparable"):
        summarize(out)


def test_pytorch_runner_leaves_the_served_model_on_its_device(model_dir):
    # On CUDA the runner must not move the served network: the CPU reference predictions would break.

    wrapper = mlflow.pyfunc.load_model(str(model_dir)).unwrap_python_model()
    pytorch_runner(wrapper.net, device="meta")  # "meta": a device available everywhere, like cuda for this purpose
    assert all(p.device.type == "cpu" for p in wrapper.net.parameters())
    rides = make_fake_rides(1, seed=3)[0]
    assert len(wrapper.predict(None, rides)) == rides["ride_id"].nunique()
