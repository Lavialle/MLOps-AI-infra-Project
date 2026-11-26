# Training resources and the distributed-training decision

**Decision: train on a single device. DDP, FSDP and ZeRO are not justified for this workload.** The arguments
below are labelled *calculated* (a formula from the course) or *measured* (raw rows in
[`../benchmarks/raw/`](../benchmarks/raw/): `training.csv`, `training-x86.csv`, `training-gpu.csv`). A measured
value marked "pending" has not been run yet, and no number replaces it.

## 1. The workload

| Item | Value | How |
|------|-------|-----|
| Model | 1D-CNN, **37,477 parameters** (FP32) | measured |
| Training set | 350 rides → **7,833 windows** of 6 × 256 float32 = **48.1 MB** | measured / calculated |
| Steps per epoch | 7,833 / 64 = **123 steps** (batch size 64) | calculated |
| Optimizer | Adam, lr 0.001; class-weighted cross-entropy | from `params.yaml` |
| One training run | best epoch 17, early stop at 25 epochs, about 2.5 min on the M2 CPU | observed once (22/11) |

## 2. Memory: what the course's formulas say

Session 3 gives the model-state memory per GPU, for P parameters in FP32 with Adam: 4 bytes of weights, 4 of
gradients and 8 of optimizer moments. With **P = 37,477**:

| Strategy | Formula (bytes) | N = 1 | N = 2 | N = 4 |
|----------|-----------------|-------|-------|-------|
| DDP | 16 P | **0.60 MB** | 0.60 MB | 0.60 MB |
| ZeRO-1 | (8 + 8/N) P | 0.60 MB | 0.45 MB | 0.37 MB |
| ZeRO-2 | (4 + 12/N) P | 0.60 MB | 0.37 MB | 0.26 MB |
| ZeRO-3 / full FSDP | 16 P / N | 0.60 MB | 0.30 MB | 0.15 MB |

*All values **calculated**, in decimal MB. They exclude activations, buffers and the CUDA context, as in the course.*

A Colab T4 has about **15 GB**. The full model state is **0.004 %** of it. Sharding exists to fit states that are
too big for one device: here, ZeRO-3 on 4 GPUs "saves" 0.45 MB.

**Activations** at batch size 64 (*calculated*, order of magnitude): each of the 3 convolution blocks outputs
64 × 32–128 channels × 64–256 steps, about 0.5 M floats, or **2 MB per block**. BatchNorm and ReLU copies
included, that's on the order of **10–20 MB**. The whole training set (48 MB) also fits in GPU memory. Activation
checkpointing (session 3, part B) would trade compute for memory we don't need.

**Measured peak memory:**

| Device | Peak memory | Source |
|--------|-------------|--------|
| Laptop CPU (M2), process RSS | **925 MB** (whole Python process: PyTorch, data, optimizer) | measured 25/11, `training.csv` |
| Laptop GPU (RTX 3060, 6 GB), `torch.cuda.max_memory_allocated` | **27 MB** (weights, gradients, Adam states, activations of a batch of 64): **0.4 %** of the GPU | measured 26/11, [`raw/training-gpu.csv`](../benchmarks/raw/training-gpu.csv) |
| Same run, process RSS | 2,101 MB (CUDA libraries and context, not the model) | same run |
| Colab T4 | not run: a local GPU was measured instead; notebook kept: [`colab_gpu_benchmark.ipynb`](colab_gpu_benchmark.ipynb) | |

## 3. Time: is there anything to parallelise?

| Device | Seconds per epoch (median of 3) | Windows / s | Source |
|--------|---------------------------------|-------------|--------|
| Laptop CPU (M2, 4 threads) | **4.82 s** (runs: 4.85, 4.82, 4.59) | **1,624** | measured 25/11, `make train-benchmark`, Docker stopped |
| Laptop CPU (AMD Ryzen 7 5800H, x86, 4 threads) | **1.84 s** (runs: 1.88, 1.84, 1.80) | **4,254** | measured 26/11, WSL2, idle kind cluster running ([`raw/training-x86.csv`](../benchmarks/raw/training-x86.csv), [environment](../benchmarks/x86/environment.md)) |
| Laptop CPU (Ryzen 7 5800H, 8 threads = physical cores) | **1.49 s** (runs: 1.54, 1.49, 1.49) | **5,249** | same run |
| **Laptop GPU (NVIDIA RTX 3060 Laptop, 6 GB, CUDA 13.2)** | **0.46 s** (runs: 0.55, 0.46, 0.45) | **17,083** | measured 26/11, same x86 laptop, PyTorch 2.15 cu132 ([`raw/training-gpu.csv`](../benchmarks/raw/training-gpu.csv), [environment](../benchmarks/gpu/environment.md)) |
| Colab T4 GPU | not run (local GPU measured instead) | | notebook kept |

On the laptop CPU, a full v1 training run (early stopping at 25 epochs) is about 25 × 4.8 s ≈ **2 min** of
training steps (*calculated* from the measured epoch time, plus a validation pass per epoch). It is already short
enough to retrain at every merge. On the x86 laptop that became the prod machine, the full retraining (split,
21 epochs with early stopping, evaluation gate) took **49 s wall-clock** (measured 26/11, one run). Going from 4 to 8
CPU threads only gains 1.2× (1.84 → 1.49 s/epoch): the workload is already too small to use more parallelism, the
same reason that makes multi-GPU pointless here.

**One GPU** (RTX 3060, measured 26/11): **0.46 s per epoch, 3.2× faster** than the best CPU setting (8 threads,
1.49 s) and 4.0× faster than 4 threads. The GPU uses **27 MB** of its 6 GB. So a single GPU does help, but only
turns ~1 min of training into ~15 s (*calculated*: 21 epochs × 0.46 s + validation): useful for phase B, not
needed for v1. The first run (0.55 s) is slower because CUDA kernels are still being selected and cached. Data preparation (read the train split, fit the scaler, cut 7,833 windows) takes
**0.33 s** (measured).

**Communication** (*calculated* volume, order-of-magnitude latency): with DDP, every step all-reduces the
gradients, 37,477 × 4 bytes = **150 KB**, 123 times per epoch. Each all-reduce also pays a fixed
synchronisation and launch latency, typically tens of microseconds or more. For a model this small, the compute
per step is of the same order. This is the **communication-bound case** of the session 2 lab (task 6): adding a
second GPU adds synchronisation without removing meaningful compute, so the speedup should stay well below 2×.
That expectation is not a measurement: we do not claim a multi-GPU result we could not run, since Colab provides
one GPU.

## 4. Decision

| Scenario (session 3, part E) | Our case | Answer |
|------------------------------|----------|--------|
| Fits on one GPU, speed matters most | **yes**, by 4 orders of magnitude | one device; DDP only if a single device were too slow, and it is not |
| Parameters do not fit on one GPU | no | FSDP and ZeRO-3 would shard parameters, but there is nothing to shard |
| Only one GPU available (laptop RTX 3060, or Colab) | yes | single-GPU time and memory **measured** on the RTX 3060; multi-GPU numbers stay calculated |

**When the decision would change:**
- **Phase B (semi-supervised pre-training on SRV3490):** about 17× more windows (~127k, 0.8 GB) for the same
  model. At the measured 1,624 windows/s, that is about **78 s per epoch on the laptop CPU**, or ~26 min for
  20 epochs (*calculated*; to be measured). The model state stays 0.6 MB, so sharding still has no purpose. Only training time grows, so we
  re-measure CPU against one GPU, and DDP becomes worth considering only if one GPU takes too long for the
  retraining cadence.
- **A much larger model** (e.g. a Transformer with hundreds of millions of parameters): the 16 P model-state
  term reaches GB scale, and FSDP or ZeRO become relevant.

## Reproduce

```bash
make train-benchmark                  # laptop CPU: 3 epochs x 3 repeats -> benchmarks/raw/training.csv
# laptop GPU: same script with a CUDA build of torch (outside the CPU-only project env)
PYTHONPATH=src python -m ride_classifier.train_benchmark --device cuda --threads 4 \
  --experiment training-gpu --out ai-infrastructure/benchmarks/raw/training-gpu.csv
make colab-bundle                     # alternative: data + model for Colab (private)
# then run colab_gpu_benchmark.ipynb on a T4 runtime and unzip colab_results.zip into ai-infrastructure/benchmarks/
```
