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

*To be filled from measured runs only.*
