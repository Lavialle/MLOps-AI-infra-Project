# Baseline benchmark: inference without HTTP (step 2)

All values below are **measured**. They are medians of 5 timed runs, computed by `make benchmark` from
[`raw/baseline-inference.csv`](raw/baseline-inference.csv) and summarised in
[`baseline-results.csv`](baseline-results.csv).

> **Correction (25/11).** The first version of this page used runs made while Docker Desktop restarted in the
> background (it started at 15:52, mid-benchmark), although the page said "Docker stopped". All numbers below
> come from a **clean rerun on a quiet machine** (25/11, 17:16–17:27, Docker stopped, load ≈ 2 at the start).
> It was run in the same session as the ONNX runs of step 3, so the two are directly comparable. The
> contaminated numbers were 3–8 % worse. The conclusions are unchanged.

## Conditions

| Item | Value |
|------|-------|
| Model | `ride-classifier` v1, PyTorch 2.15 eager, `weights.pt` sha `9499b07b62e76afc` |
| Test data | the frozen test split: 75 rides, 230,622 rows, sha `ec00535a372d7a68`; ~23 windows per ride on average |
| Machine | MacBook Air M2 (4 performance + 4 efficiency cores), 8 GB RAM, macOS; CPU only; Docker stopped. Full record in [`environment.md`](environment.md) |
| What a request does | exactly `RideClassifier.predict`, checked at startup (`check_equivalence`): gap filling, scaling and windowing (**preprocessing**), CNN forward pass and softmax (**network**), mean per ride (**aggregation**). No HTTP |
| Protocol | 3 warm-up requests per batch size (not timed); each run = at least 20 requests, cycling over the 75 test rides; 5 runs; medians |
| Varied | `torch.set_num_threads` = 1, 4 (PyTorch's default here), 8; rides per request = 1, 4, 16, 75 |

## Results

| Threads | Rides / request | Mean latency (ms) | p95 latency (ms) | Throughput (rides/s) | Preprocessing (ms/ride) | Network (ms/ride) | Network share |
|--------:|----------------:|------------------:|-----------------:|---------------------:|------------------------:|------------------:|--------------:|
| 1 | 1 | 24.1 | 37.5 | 41.5 | 2.0 | 21.5 | 89 % |
| 1 | 4 | 94.8 | 111.6 | 42.2 | 1.0 | 22.5 | 95 % |
| 1 | 16 | 415.3 | 443.2 | 38.5 | 0.7 | 25.2 | 97 % |
| 1 | 75 | 2,116.7 | 2,148.8 | 35.4 | 0.7 | 27.5 | 98 % |
| **4** | **1** | **16.4** | **22.0** | **60.9** | 2.1 | 13.6 | 83 % |
| 4 | 4 | 40.6 | 46.1 | 98.5 | 1.0 | 8.9 | 88 % |
| **4** | **16** | 143.0 | 152.8 | **111.9** | 0.7 | 8.2 | 91 % |
| 4 | 75 | 716.8 | 739.6 | 104.6 | 0.7 | 8.9 | 93 % |
| 8 | 1 | 24.7 | 32.3 | 40.6 | 2.1 | 21.6 | 88 % |
| 8 | 4 | 52.6 | 58.6 | 76.1 | 1.0 | 11.9 | 90 % |
| 8 | 16 | 166.5 | 176.0 | 96.1 | 0.7 | 9.6 | 92 % |
| 8 | 75 | 774.2 | 893.7 | 96.9 | 0.7 | 9.6 | 93 % |

**Quality:** macro F1 **0.317**, accuracy 0.693 at every thread count, identical to the evaluation gate.

**Memory of a serving process:** **582 MB** peak RSS (median of 3 fresh processes that load the model and score
the 75 test rides, 1 thread; [`../optimization/raw/serving-memory.csv`](../optimization/raw/serving-memory.csv)).
The benchmark process itself peaks higher (≈ 670 MB), because it also runs the PyTorch reference over the whole
test split for the parity check, so it is not used as the memory figure. Model load time is 0.18–0.25 s.

### Baseline template of the brief (reference configuration: 4 threads)

| Metric | Value | Measurement conditions |
|--------|-------|------------------------|
| Model and version | `ride-classifier` v1, PyTorch 2.15 eager, MLflow pyfunc | weights sha `9499b07b62e76afc` |
| Hardware | Apple M2, 8 cores (4P + 4E), 8 GB RAM, no GPU | macOS, Docker stopped |
| Model size (MB) | 0.16 (`weights.pt`); 0.87 for the full MLflow model directory | serialized files |
| Mean latency (ms) | **16.4** (1 ride) · 143.0 (16 rides) | 4 threads, ≥ 20 requests × 5 runs |
| p95 latency (ms) | **22.0** (1 ride) · 152.8 (16 rides) | same |
| Throughput (requests/s) | **60.9** rides/s at 1 ride per request; **111.9** rides/s at 16 rides per request | 1 client, in-process, no HTTP |
| Memory usage (MB) | **582** peak RSS of a serving process | CPU RAM, fresh process, 1 thread |
| Classification quality | macro F1 **0.317**, accuracy 0.693 | the 75-ride test split; identical to the evaluation gate |

## What the numbers say

1. **The network dominates, not the preprocessing.** It takes 83–98 % of the time: 8–28 ms per ride, against
   0.7–2 ms of pandas/NumPy preprocessing. We expected the opposite before measuring. This made the inference
   runtime (step 3) the right optimization target.
2. **Batching pays.** At 4 threads, the network cost per ride drops from 13.6 ms (1 ride per call) to 8.2 ms
   (16 rides), so throughput nearly doubles: 60.9 → 111.9 rides/s. Latency per request grows. A batch API
   (`/predict/file`) serves bulk scoring better than one call per ride.
3. **More threads is not better.** 8 threads is slower than 4 at every batch size. The M2 has 4 performance and
   4 efficiency cores, so with 8 threads each step waits for the slowest. With 1 thread, throughput stays at
   35–42 rides/s whatever the batch size.
4. **For capacity planning, use 1 thread.** A production pod is limited to **1 CPU** (`k8s/api.yaml`). One thread
   in-process gives **41.5 rides/s at a p95 of 37.5 ms** for single-ride requests: the upper bound of one
   replica with PyTorch.
5. **HTTP costs as much as the model.** Through the API (Locust, 22/11, one uvicorn process), we measured
   **16.5 rides/s at a p95 of 340 ms**: 2.5× less throughput and 9× the p95 of the in-process path. The difference
   is the HTTP layer: JSON parsing and Pydantic validation of ~3,000 samples per ride, then DataFrame conversion.
   This is a lead for later, separate from the model runtime.

## Reproduce

```bash
make infra-env      # environment + hashes
make benchmark      # threads 4, 1, 8 -> raw/baseline-inference.csv, then medians -> baseline-results.csv
```

Stop Docker first: a background VM shifts every number. Each run appends rows to the raw file, so delete
`raw/baseline-inference.csv` first for a clean rerun. `summarize` refuses to mix two models or two test sets in
one experiment.
