# Phase B: semi-supervised pre-training on the unlabelled rides (design)

**Status: design. Nothing here is implemented or measured yet.**

## Why

Model v1 learns from 350 labelled training rides, with only 4–26 rides for each rare class. Its test macro F1 is
0.317, and its scores are noisy. The course data also contains `SRV3490.parquet`: 3,490 rides in the same format
(10 Hz, same channels, same gravity orientation; checked 23/11), with **no labels** for our 5 classes. That is
about 127,000 windows, 17× the labelled windows.

Self-supervised pre-training lets the CNN learn what riding motion looks like from those 3,490 rides first.
Fine-tuning on the 350 labelled rides then only has to learn the 5 classes. It also makes the AI Infrastructure
contribution substantive: a 0.8 GB training set, a 30–40 min CPU pre-training run (estimated), and a real
CPU-vs-GPU and single-vs-multi-GPU question (step 4).

## Design

| Decision | Choice | Reason |
|----------|--------|--------|
| Pretext task | **Masked reconstruction**: hide random 2.5 s segments of a window (about 25 % of the time steps), reconstruct the original 6 channels, loss on the hidden parts | simple and stable; needs no negative pairs or large batches (unlike contrastive learning); forces the encoder to model motion dynamics |
| Encoder | the 3 convolution blocks of the current CNN (`build_net` without its head), optionally **wider** (channel multiplier in `params.yaml`) | the same architecture keeps serving unchanged; width is the lever if 17× more data supports a bigger model |
| Decoder | a small convolutional upsampler, used only during pre-training | discarded afterwards: the served model has the same shape as v1 |
| Fine-tuning | encoder initialised from pre-training, new linear head, class-weighted loss as in v1. The encoder learning rate is 10× lower than the head's for the first epochs | standard transfer recipe; protects the pre-trained features |
| Preprocessing | **unchanged**: the scaler fitted on the labelled train split, same windows | the API, the ONNX export and every benchmark work as they are |
| Data loading | stream the parquet file by row groups, write windows to a memory-mapped `.npy` file on disk | measured 23/11: about 1.5 GB peak RAM instead of swapping, so it runs on the 8 GB laptop |
| Hardware | laptop CPU for small runs; **Colab T4** for full pre-training | measured comparison in step 4 |

## Leakage and evaluation rules

1. **No test or validation ride in pre-training.** Before using SRV3490, check that no ride id (and no identical
   signal) overlaps with GBG500's validation and test splits. Unlabelled data from test rides would still leak.
2. **Same frozen test split, same gate** as v1 (cross-validated mean macro F1 ≥ 0.25, baseline beaten in every fold and on the test split). Phase B only reaches
   production if v2 beats v1 on it.
3. **Variance:** 1 Reckless ride in the test split makes one number fragile. Report 5-fold cross-validation by
   ride (mean ± std) for both v1 and v2 next to the test score.
4. **Measure, do not claim:** pre-training time, peak memory (CPU and GPU) and the F1 gain are measured; nothing
   is extrapolated as a result.

## Pipeline changes

```
SRV3490 (dvc add) ──▶ unlabelled windows (memmap) ──▶ pretrain (encoder) ─┐
GBG500 ──▶ split ──▶ train (fine-tune from the encoder) ◀────────────────┘ ──▶ evaluate (same gate) ──▶ @staging
```

- New DVC stages `unlabelled-windows` and `pretrain`; `train` gains a `pretrained_encoder` input.
- The pre-training run is logged to MLflow (reconstruction loss per epoch, time, hardware), and the fine-tuning
  run records which encoder it used.
- `retrain.yml` keeps working: pre-training is cached by DVC and only re-runs when its data or parameters change.

## After phase B: rerun the AI Infrastructure measurements

| Step | What changes |
|------|--------------|
| 1. Inventory | dataset 0.8 GB of windows, training time, possibly a wider model |
| 2. Baseline | same harness on v2, reported next to v1 |
| 3. ONNX | same export and comparison on v2 |
| 4. Distributed | pre-training on CPU vs one T4 (measured); the DDP decision revisited with real training times |
| 5. Cost | retraining cost with GPU hours for the pre-training |

## Open decisions (team)

- Keep the encoder at today's size (37k parameters: serving numbers stay comparable), or widen it?
- Masked reconstruction (proposed), or contrastive pre-training?
- Who runs the full pre-training on Colab?
