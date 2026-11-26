# AI Infrastructure contribution

This folder extends the ride-classifier MLOps project (see the [main README](../README.md)) with the AI Infrastructure
analysis. It covers the model's computing requirements, the measured inference performance, one optimization, the
distributed-training decision, and a scalable, cost-aware serving design.

> **Measure, do not claim.** Every number in this folder is labelled **measured** (we ran it, raw data committed),
> **calculated** (derived from a formula, e.g. ZeRO memory) or **estimated** (built on stated assumptions, e.g. cloud
> prices). Nothing is reported without its environment.

## Status

| Part | Folder | Suggested owner | Status |
|------|--------|-----------------|--------|
| 0. Setup: rules, reference, environment | `benchmarks/environment.*` | Nicolas Gilhodes | done |
| 1. Inventory and current architecture | [`architecture/current-system.md`](architecture/current-system.md) | Aymeric Mouttapa | done |
| 2. Baseline benchmark (inference) | [`benchmarks/README.md`](benchmarks/README.md) | Nicolas Gilhodes | done |
| 3. Optimization (ONNX Runtime) | [`optimization/README.md`](optimization/README.md) | Ange Lavialle | done: 4.5–10.6× faster, −51 % memory, identical predictions |
| 4. Training sizing and distributed-training decision | `distributed-training/` | Leon Pejic | to do |
| 5. Scaling, cost and resilience | `cost-resilience/` | Aymeric Mouttapa, Nicolas Gilhodes | to do |

Each part is developed on its own branch and merged through a reviewed pull request.

## Fixed reference

Every comparison uses the same model version and the same test data. Both are pinned by content hash in
[`benchmarks/environment.md`](benchmarks/environment.md):

| | Reference |
|-|-----------|
| Model | `ride-classifier` **v1** (PyTorch 1D-CNN, 37,477 parameters, `weights.pt` 159 KB), exported to `./model` by `make export` |
| Test data | the frozen test split `data/processed/test.parquet`: 75 rides, 230,622 rows (`make pull pipeline`) |
| CPU machine | MacBook Air, Apple M2 (8 cores), 8 GB RAM, macOS; PyTorch uses 4 threads by default |
| GPU machine | Google Colab T4 (one GPU, when available) |

When the model changes (phase B: semi-supervised pre-training), all measurements are rerun with the new
version and reported **next to** v1, never instead of it.

## Measurement rules

1. **Record the environment first:** `make infra-env` on each machine. It writes `benchmarks/environment.{json,md}`
   (hardware, versions, and the hashes of the model and test data).
2. **Warm up, then time.** Warm-up runs are never timed. On GPU, call `torch.cuda.synchronize()` before and after
   timing.
3. **Repeat:** at least 3 runs per configuration, and report the **median** (p95 for latency distributions).
4. **Hold everything else fixed:** same model hash, same test hash, same machine, same thread count, same batch
   size. Change one thing at a time.
5. **Keep the raw data.** Every run appends rows to a CSV in `benchmarks/raw/` (format below), and tables in the
   reports are computed from those files.
6. **Check quality too.** An optimization is only valid if the predictions stay the same, so every benchmark also
   reports macro F1 on the test split.
7. **Failures are results.** An installation error, an out-of-memory error, or an optimization that turns out
   slower is documented with its cause, never hidden or replaced by an invented number.

### Raw result format

One row per timed run, in `benchmarks/raw/<experiment>.csv`:

| Column | Example | Meaning |
|--------|---------|---------|
| `experiment` | `baseline-inference` | what is measured |
| `kind` | `measured` | `measured`, `calculated` or `estimated` |
| `runtime` | `pytorch` | `pytorch`, `onnxruntime`, … |
| `device` | `cpu` | `cpu`, `cuda`, `mps` |
| `threads` | `4` | CPU threads used |
| `batch_rides` | `16` | rides per call |
| `run` | `2` | repetition index (warm-up runs are not recorded) |
| `metric` | `latency_ms` | `latency_ms`, `rides_per_s`, `peak_rss_mb`, `peak_cuda_mb`, `macro_f1`, … |
| `value` | `41.7` | the measurement |
| `model_sha` | `9499b07b62e76afc` | from `environment.json`: guarantees the same model |
| `test_sha` | `ec00535a372d7a68` | guarantees the same test data |
| `commit` | `3f8aeb8` | code version |

## Layout

```
ai-infrastructure/
├── README.md                 this file: rules, reference, status
├── architecture/             current-system and proposed-system diagrams, component inventory
├── benchmarks/               environment record, benchmark procedure and results; raw/ = raw CSVs
├── optimization/             ONNX Runtime export, before/after comparison
├── distributed-training/     training measurements (CPU, Colab GPU), DDP/FSDP/ZeRO analysis
└── cost-resilience/          serving architecture, scaling test, cost estimates, rollback plan
```

The measurement code lives with the rest of the project in `src/ride_classifier/`. It runs as
`python -m ride_classifier.<module>` with a `make` target, as in the course labs. The notebooks in this folder only
present results.

## Reproduce

```bash
make setup                 # Python env
make pull pipeline export  # data, model v1 in ./model (needs the MLflow that holds v1)
make infra-env             # environment + reference record
```
