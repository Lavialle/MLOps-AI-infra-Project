# Optimization: ONNX Runtime (step 3)

## Hypothesis (written before any test)

Step 2 measured that the **CNN forward pass takes 81–97 % of inference time** (9–30 ms per ride), while
preprocessing takes only 1–2 ms. The model is small (37,477 parameters), so eager PyTorch spends much of that
time in per-operator overhead rather than in arithmetic. We also know the serving process uses ~410 MB of RSS,
mostly for the runtime and not the 159 KB of weights.

We expect that **exporting the CNN to ONNX and running it with ONNX Runtime** (CPU execution provider, full graph
optimizations: BatchNorm fused into the convolutions, kernels for small tensors) will:

1. **cut the network time per ride** (lower mean and p95 latency, higher throughput), with a larger gain at
   1 ride per request (overhead-bound) than at 16 rides (compute-bound);
2. **keep the predictions**: maximum absolute probability difference below 1e-4, 100 % identical predicted classes,
   same macro F1 (0.317);
3. **reduce the memory of a serving process**, if PyTorch no longer needs to be imported: an ONNX-only process
   (onnxruntime + NumPy + pandas) instead of PyTorch + MLflow.

A second variant, **INT8 dynamic quantization** (`onnxruntime.quantization.quantize_dynamic`), may be faster
still, but **may change some predictions**. It is reported with its quality, and accepted only if macro F1 does not
drop.

The hypothesis can fail. A slower or equal ONNX runtime is a valid result, as long as it is explained.

## Fair comparison conditions (fixed before testing)

| Held fixed | Value |
|------------|-------|
| Model | v1 weights (sha `9499b07b62e76afc`), exported once to ONNX; the ONNX file's sha is recorded |
| Test workload | the same 75 test rides, same request groups (1 / 4 / 16 / 75 rides), same warm-up (3) and repeats (5) |
| Preprocessing and aggregation | identical code: only the **network runner** changes (`Pipeline` in `ride_classifier.benchmark`) |
| Machine | the same M2 laptop, same session, **Docker stopped**, baseline rerun back-to-back with the ONNX runs |
| Threads | 1 (the 1-CPU production pod) and 4 (best laptop setting): `torch.set_num_threads` for PyTorch, `intra_op_num_threads` for ONNX Runtime, 1 inter-op thread |
| Quality | macro F1 on the test split, plus max probability difference and prediction agreement vs PyTorch |
| Memory | peak RSS of a fresh process that loads the model and scores the test split, 3 repeats per runtime |

## Results

All **measured** on 25/11, 17:16–17:27, in one session on a quiet machine (Docker stopped). The PyTorch baseline was
rerun in the same session, interleaved with the ONNX runs. Medians of 5 runs; raw rows in
[`raw/onnx-inference.csv`](raw/onnx-inference.csv) and
[`../benchmarks/raw/baseline-inference.csv`](../benchmarks/raw/baseline-inference.csv).

### Export

| Artifact | Size | sha256 (16) |
|----------|------|-------------|
| PyTorch weights `weights.pt` (reference) | 155 KB | `9499b07b62e76afc` |
| `model.onnx`: CNN + softmax, FP32, opset 17, dynamic batch | 146 KB | `f57c100677f148a8` |
| `model.int8.onnx`: dynamic INT8 quantization of the weights | **43 KB** (3.4× smaller) | `4d38c6b74cbaf66e` |

`make onnx` re-creates them from `./model` (`python -m ride_classifier.optimize export`).

### Latency and throughput

| Threads | Rides / request | Runtime | Mean latency (ms) | p95 latency (ms) | Throughput (rides/s) | Speed-up vs PyTorch | Network (ms/ride) | Network share |
|--------:|----------------:|---------|------------------:|-----------------:|---------------------:|--------------------:|------------------:|--------------:|
| 1 | 1 | PyTorch | 24.1 | 37.5 | 41.5 | — | 21.48 | 89 % |
| 1 | 1 | ONNX Runtime FP32 | 4.5 | 5.6 | 222.3 | **5.4×** | 1.92 | 43 % |
| 1 | 1 | ONNX Runtime INT8 | 3.7 | 4.3 | 273.8 | **6.6×** | 1.05 | 29 % |
| 1 | 4 | PyTorch | 94.8 | 111.6 | 42.2 | — | 22.53 | 95 % |
| 1 | 4 | ONNX Runtime FP32 | 11.9 | 13.4 | 336.4 | **8.0×** | 1.88 | 63 % |
| 1 | 4 | ONNX Runtime INT8 | 8.6 | 9.6 | 464.4 | **11.0×** | 1.04 | 48 % |
| 1 | 16 | PyTorch | 415.3 | 443.2 | 38.5 | — | 25.17 | 97 % |
| 1 | 16 | ONNX Runtime FP32 | 42.3 | 44.3 | 378.5 | **9.8×** | 1.91 | 72 % |
| 1 | 16 | ONNX Runtime INT8 | 29.4 | 30.6 | 545.0 | **14.1×** | 1.08 | 59 % |
| 1 | 75 | PyTorch | 2,116.7 | 2,148.8 | 35.4 | — | 27.55 | 98 % |
| 1 | 75 | ONNX Runtime FP32 | 199.5 | 224.3 | 375.9 | **10.6×** | 1.96 | 74 % |
| 1 | 75 | ONNX Runtime INT8 | 131.9 | 132.8 | 568.5 | **16.0×** | 1.09 | 62 % |
| 4 | 1 | PyTorch | 16.4 | 22.0 | 60.9 | — | 13.61 | 83 % |
| 4 | 1 | ONNX Runtime FP32 | 3.7 | 4.3 | 273.1 | **4.5×** | 0.85 | 23 % |
| 4 | 1 | ONNX Runtime INT8 | 3.5 | 3.9 | 285.2 | **4.7×** | 0.64 | 18 % |
| 4 | 4 | PyTorch | 40.6 | 46.1 | 98.5 | — | 8.90 | 88 % |
| 4 | 4 | ONNX Runtime FP32 | 8.4 | 9.5 | 476.4 | **4.8×** | 0.85 | 41 % |
| 4 | 4 | ONNX Runtime INT8 | 7.2 | 8.1 | 553.7 | **5.6×** | 0.60 | 33 % |
| 4 | 16 | PyTorch | 143.0 | 152.8 | 111.9 | — | 8.17 | 91 % |
| 4 | 16 | ONNX Runtime FP32 | 26.4 | 27.7 | 605.3 | **5.4×** | 0.83 | 50 % |
| 4 | 16 | ONNX Runtime INT8 | 23.3 | 24.3 | 687.5 | **6.1×** | 0.65 | 45 % |
| 4 | 75 | PyTorch | 716.8 | 739.6 | 104.6 | — | 8.87 | 93 % |
| 4 | 75 | ONNX Runtime FP32 | 112.6 | 116.1 | 666.3 | **6.4×** | 0.80 | 53 % |
| 4 | 75 | ONNX Runtime INT8 | 101.2 | 103.8 | 740.8 | **7.1×** | 0.68 | 51 % |

### Prediction quality (all 75 test rides)

| Runtime | Macro F1 | Accuracy | Max probability difference vs PyTorch | Same predicted class as PyTorch |
|---------|---------:|---------:|--------------------------------------:|--------------------------------:|
| PyTorch (reference) | 0.317 | 0.693 | — | — |
| ONNX Runtime FP32 | **0.317** | 0.693 | < 1e-6 | **75 / 75** |
| ONNX Runtime INT8 | 0.321 | 0.707 | 0.024 | 74 / 75 |

### Memory of a serving process

A fresh process that loads the model like the API and scores the 75 test rides (1 thread, 3 repeats,
[`raw/serving-memory.csv`](raw/serving-memory.csv)). The ONNX processes import neither PyTorch nor MLflow; this
is checked by the script and by a test.

| Runtime | Peak RSS (median of 3) | vs PyTorch | PyTorch imported |
|---------|-----------------------:|-----------:|:----------------:|
| PyTorch + MLflow | 582 MB | — | yes |
| ONNX Runtime FP32 | **287 MB** | **−51 %** | no |
| ONNX Runtime INT8 | 316 MB | −46 % | no |

Installed package sizes in the project environment (macOS, measured with `du`): **torch 598 MB** and
mlflow 59 MB, against **onnxruntime 81 MB**. An ONNX-only API image could drop about 600 MB of dependencies
(695 MB for torch in the Linux image, measured 22/11). This is not yet measured as a built image.

## Was the hypothesis right?

| Prediction (written before testing) | Result |
|-------------------------------------|--------|
| 1. ONNX Runtime cuts the network time per ride | **Confirmed, beyond expectations:** the network is 11–16× faster (21.5 → 1.9 ms per ride at 1 thread; 13.6 → 0.85 ms at 4 threads). End-to-end throughput rises **4.5×–10.6×** (FP32) |
| 1b. …with a larger gain at 1 ride per request than at 16 | **Wrong for the end-to-end gain.** It is *smaller* at 1 ride (5.4× vs 9.8× at 1 thread). Once the network is fast, the ~2 ms of preprocessing per ride, which ONNX does not change, dominates small requests: Amdahl's law. The network-only gain does follow the prediction at 4 threads (16× vs 10×) |
| 2. Predictions identical, same macro F1 | **Confirmed** for FP32: probabilities within 1e-6, 75/75 identical classes. INT8 changes **1 ride** of 75: F1 0.321 instead of 0.317, a difference that is noise on 75 rides, not an improvement |
| 3. Less memory without PyTorch | **Confirmed:** 582 → 287 MB (−51 %) |

## Decision and consequences

- **Adopt ONNX Runtime FP32 for serving.** It is 4.5–10.6× faster, uses half the memory, removes ~600 MB of
  dependencies, and gives **identical predictions**. The PyTorch model stays the training and registry format,
  and the ONNX file is derived from it at export time, like `./model` today.
- **Keep INT8 as a candidate only.** It is a further 1.04–1.5× faster, but it changes predictions, and the
  quality change cannot be judged on 75 rides. It would have to pass the evaluation gate as its own model
  version before serving.
- **The bottleneck has moved.** With ONNX Runtime, preprocessing is up to 77 % of a single-ride request. The next
  optimizations are:
  - the pandas preprocessing, e.g. vectorised windowing without `groupby`
  - the HTTP layer, measured in step 2 at 2.5× slower than in-process
- **Cost (step 5):** per replica, capacity goes up about 5× and memory down about 2×. The minimum deployment
  of 2 replicas can use smaller pods, which reduces the main cost item.

Not done yet: building and measuring an ONNX-only API image, and serving it through the API and Kubernetes. That
is the follow-up if the team adopts the decision.
