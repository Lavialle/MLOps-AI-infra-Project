# Baseline benchmark: inference without HTTP (step 2)

All values below are **measured**. They are medians of 5 timed runs, computed by `make benchmark` from
[`raw/baseline-inference.csv`](raw/baseline-inference.csv) (225 rows per configuration), and summarised in
[`baseline-results.csv`](baseline-results.csv).

## Conditions

| Item | Value |
|------|-------|
| Model | `ride-classifier` v1, PyTorch eager, `weights.pt` sha `9499b07b62e76afc` |
| Test data | the frozen test split: 75 rides, 230,622 rows, sha `ec00535a372d7a68`; ~23 windows per ride on average |
| Machine | MacBook Air M2 (4 performance + 4 efficiency cores), 8 GB RAM, macOS; CPU only; Docker stopped, no other heavy process. Full record in [`environment.md`](environment.md) |
| What a request does | exactly `RideClassifier.predict`, checked at startup (`check_equivalence`): gap filling, scaling and windowing (**preprocessing**), CNN forward pass and softmax (**network**), mean per ride (**aggregation**). No HTTP |
| Protocol | 3 warm-up requests per batch size (not timed); each run = at least 20 requests, cycling over the 75 test rides; 5 runs; medians |
| Varied | `torch.set_num_threads` = 1, 4 (PyTorch's default here), 8; rides per request = 1, 4, 16, 75 |

## Results

| Threads | Rides / request | Mean latency (ms) | p95 latency (ms) | Throughput (rides/s) | Preprocessing (ms/ride) | Network (ms/ride) | Network share |
|--------:|----------------:|------------------:|-----------------:|---------------------:|------------------------:|------------------:|--------------:|
| 1 | 1 | 26.0 | 40.3 | 38.5 | 2.3 | 22.7 | 88 % |
| 1 | 4 | 103.4 | 127.5 | 38.7 | 1.2 | 24.2 | 94 % |
| 1 | 16 | 451.3 | 505.0 | 35.5 | 0.8 | 27.3 | 97 % |
| 1 | 75 | 2,300.5 | 2,405.9 | 32.6 | 0.8 | 29.7 | 97 % |
| **4** | **1** | **18.0** | **27.2** | **55.6** | 2.4 | 14.6 | 81 % |
| 4 | 4 | 42.0 | 49.8 | 95.3 | 1.0 | 9.2 | 88 % |
| **4** | **16** | 154.8 | 175.9 | **103.4** | 0.8 | 8.9 | 92 % |
| 4 | 75 | 793.2 | 877.2 | 94.6 | 0.8 | 9.8 | 93 % |
| 8 | 1 | 24.5 | 32.6 | 40.8 | 2.4 | 21.2 | 87 % |
| 8 | 4 | 54.6 | 62.7 | 73.2 | 1.1 | 12.2 | 90 % |
| 8 | 16 | 172.7 | 188.1 | 92.6 | 0.7 | 10.0 | 92 % |
| 8 | 75 | 831.6 | 896.3 | 90.2 | 0.8 | 10.3 | 93 % |

| Threads | Model load (s) | Memory after load (MB) | Peak memory (MB) | Macro F1 (test) | Accuracy (test) |
|--------:|---------------:|-----------------------:|-----------------:|----------------:|----------------:|
| 1 | 0.43 | 421 | 553 | 0.317 | 0.693 |
| 4 | 0.49 | 342 | 409 | 0.317 | 0.693 |
| 8 | 0.44 | 394 | 499 | 0.317 | 0.693 |

*Memory is the process's peak resident set (RSS), so it includes Python, PyTorch, MLflow and pandas, not only the model.*

### Baseline template of the brief (reference configuration: 4 threads)

| Metric | Value | Measurement conditions |
|--------|-------|------------------------|
| Model and version | `ride-classifier` v1, PyTorch 2.15 eager, MLflow pyfunc | weights sha `9499b07b62e76afc` |
| Hardware | Apple M2, 8 cores (4P + 4E), 8 GB RAM, no GPU | macOS, Docker stopped |
| Model size (MB) | 0.16 (`weights.pt`); 0.87 for the full MLflow model directory | serialized files |
| Mean latency (ms) | **18.0** (1 ride) · 154.8 (16 rides) | 4 threads, ≥ 20 requests × 5 runs |
| p95 latency (ms) | **27.2** (1 ride) · 175.9 (16 rides) | same |
| Throughput (requests/s) | **55.6** rides/s at 1 ride per request; **103.4** rides/s at 16 rides per request | 1 client, in-process, no HTTP |
| Memory usage (MB) | 409 peak RSS (342 after loading the model) | CPU RAM |
| Classification quality | macro F1 **0.317**, accuracy 0.693 | the 75-ride test split; identical to the evaluation gate |

## What the numbers say

1. **The network dominates, not the preprocessing.** It takes 81–97 % of the time: 9–30 ms per ride, against
   about 1–2 ms of pandas/NumPy preprocessing. We expected the opposite before measuring. This makes the
   inference runtime (step 3) a valid optimization target for latency, not only for memory.
2. **Batching pays.** At 4 threads, the network cost per ride drops from 14.6 ms (1 ride per call) to 8.9 ms
   (16 rides), because one larger tensor uses the cores better. Throughput nearly doubles, 55.6 → 103.4 rides/s,
   while latency per request grows. A batch API (`/predict/file`) serves bulk scoring better than one call per ride.
3. **More threads is not better.** 8 threads is slower than 4 at every batch size. The M2 has 4 performance and
   4 efficiency cores, so with 8 threads each step waits for the slowest. With 1 thread, throughput stays at
   about 33–39 rides/s whatever the batch size.
4. **For capacity planning, use 1 thread.** A production pod is limited to **1 CPU** (`k8s/api.yaml`). One thread
   in-process gives **38.5 rides/s at a p95 of 40 ms** for single-ride requests: the upper bound of one replica.
5. **HTTP costs more than the model.** Through the API (Locust, 22/11, one uvicorn process), we measured
   **16.5 rides/s at a p95 of 340 ms**: roughly half the throughput and 8× the p95 of the in-process path. The
   difference is the HTTP layer: JSON parsing and Pydantic validation of ~3,000 samples per ride, then
   DataFrame conversion. This is a lead for later, and it is separate from the model runtime.

## Reproduce

```bash
make infra-env      # environment + hashes (refuses nothing, just records)
make benchmark      # threads 4, 1, 8 -> raw/baseline-inference.csv, then medians -> baseline-results.csv
```

Each run appends rows to the raw file. Delete `raw/baseline-inference.csv` first for a clean rerun. `summarize`
refuses to mix two models or two test sets in one experiment.
