# Model, training and MLflow

## Pipeline

```
make pull        # dvc pull: raw data from S3
make pipeline    # dvc repro: split -> train, cv -> evaluate (needs `make up`)
make export      # model @staging -> ./model (baked into the API image)
```

| Stage | Module | Output |
|-------|--------|--------|
| split | `ride_classifier.data` | `data/processed/{train,val,test}.parquet` |
| train | `ride_classifier.train` | MLflow run + new registered version of `ride-classifier`, `reports/train.json` |
| cv | `ride_classifier.cross_validate` | 5-fold cross-validation by ride of the training recipe, `reports/cv.json` |
| evaluate | `ride_classifier.evaluate` | test metrics on the run, gate tag on the version, alias `@staging` if passed, `reports/eval.json` |
| export | `ride_classifier.export_model` | `./model` (MLflow pyfunc) + `model_meta.json` |

## Model

**1D-CNN** on 256-sample windows (25.6 s) of the 6 standardised IMU channels:

```
Conv1d(6→32, k7)-BN-ReLU-MaxPool → Conv1d(32→64, k5)-BN-ReLU-MaxPool → Conv1d(64→128, k3)-BN-ReLU
→ GlobalAvgPool → Dropout → Linear(128→5)
```

The ride-level prediction is the mean of its window probabilities. Global average pooling makes the network accept any window length.

**Class imbalance.** Training has 292 Safe rides against 4 Reckless. Two measures handle it:

- The cross-entropy is weighted by inverse class frequency (`class_weighting: balanced`).
- The epoch kept is the best on **ride-level macro F1** on the validation split (early stopping, `patience`).

## One artifact = preprocessing + model

The registered model is an MLflow **pyfunc** (`ride_classifier.model.RideClassifier`). It takes raw IMU rows and returns one row per ride:

| input columns | output columns |
|---------------|----------------|
| `ride_id, time_index, ax, ay, az, rx, ry, rz` | `ride_id, prediction, Safe, Safe overall, Bad weather, Tandem, Reckless` |

Everything below travels inside the artifact, so serving cannot drift from training:

- gap interpolation
- the channel scaler (fitted on train, also logged as `channel_scaler.json`)
- windowing
- the network weights (`weights.pt` artifact)
- per-ride aggregation

## What is logged to MLflow

| Kind | Content |
|------|---------|
| Tags | `git_commit`, `data_md5` (train split), `data_path` |
| Params | everything in `params.yaml` sections `data`, `features`, `train` |
| Metrics per epoch | `train_loss`, `val_macro_f1`, `val_balanced_accuracy`, `val_accuracy`, `val_recall_<class>` |
| Final metrics | best-epoch val scores, `val_baseline_*` (always predict the majority class), `test_*` added by the gate |
| Artifacts | the model (signature + input example), `val_confusion_matrix.json`, `test_confusion_matrix.json`, `channel_scaler.json` |
| Registry | each run registers a new version. The gate adds the tag `gate=passed|failed` and the alias `@staging` |

## Evaluation gate

A single 75-ride test split holds **one** Reckless ride: its macro F1 moves by about ±0.08 depending on which rides
land in it (measured below). The gate therefore judges the **training recipe** on a cross-validation, and only
sanity-checks the **version** that gets promoted on the frozen test split. It passes only if all three hold:

| Check | Source | Rule (`params.yaml`, section `gate`) |
|-------|--------|---------------------------------------|
| The recipe is good | `reports/cv.json` (stage `cv`) | mean macro F1 over the 5 folds `>= min_cv_macro_f1` (0.25) |
| The recipe beats the baseline everywhere | `reports/cv.json` | macro F1 above the majority baseline (~0.18) **in every fold** |
| The promoted version is not broken | frozen test split, model loaded from the registry | `test_macro_f1 > test_baseline_macro_f1` |

Otherwise the script exits with code 1, which stops `dvc repro` and the retraining workflow. The floor 0.25 is the
lowest measured CV mean (0.32) minus about two standard errors (std / √5 ≈ 0.04): a recipe at that level is
clearly worse than ours, a recipe at baseline level (0.18) fails every time. The frozen test split stays the fixed
reference for the benchmarks of [`ai-infrastructure/`](../ai-infrastructure/).

The served v1 passes: CV 0.371 ± 0.089 (baseline beaten in 5/5 folds), test 0.329 against a baseline of 0.181.

## Cross-validation: how noisy is one test split?

`make cv` (DVC stage `cv`, or `python -m ride_classifier.cross_validate --device cuda`) runs a **5-fold
cross-validation stratified by ride** with the same `params.yaml`: each ride is tested exactly once, by a model that
never saw it, so all 6 Reckless rides are tested instead of 1. In each fold, 15 % of all rides is held out from the
training folds for early stopping. Output: [`reports/cv.json`](../reports/cv.json).

Measured 26/11 on the x86 laptop CPU (the pipeline run, 2 min 44 s):

| Fold | Test rides (Reckless) | Macro F1 | Balanced accuracy | Best epoch |
|------|-----------------------|----------|-------------------|------------|
| 1 | 100 (1) | 0.424 | 0.631 | 9 |
| 2 | 100 (2) | 0.313 | 0.376 | 15 |
| 3 | 100 (1) | 0.242 | 0.263 | 1 |
| 4 | 100 (1) | 0.444 | 0.570 | 17 |
| 5 | 100 (1) | 0.432 | 0.446 | 14 |
| **Mean ± std** | 500 (6) | **0.371 ± 0.089** | **0.457 ± 0.148** | |
| Majority baseline | | 0.182 | 0.200 | |

Pooled over the 500 rides (each predicted once):

| | Safe | Safe overall | Bad weather | Tandem | Reckless |
|-|------|--------------|-------------|--------|----------|
| Recall | 0.721 | 0.263 | 0.364 | 0.333 | **0.500** (3 / 6) |

The same run on the laptop GPU (RTX 3060, 59 s) gave **0.323 ± 0.076**: GPU and CPU kernels round differently, and
the gap (0.05) is within one standard deviation. That is the noise level any comparison has to beat.

What it shows:
- **The model beats the baseline in every fold** (0.24–0.44 against 0.18): the signal is real.
- **One 75-ride split is not enough to judge a model**: fold scores range from 0.24 to 0.44. The old floor (0.30 on
  the frozen split) passed on 4 folds out of 5 on CPU and 3 out of 5 on GPU, which is why the gate now uses the CV mean.
- **Reckless recall is 0.5, not 1.** The single Reckless ride of the frozen test split was one of the easy ones.
- **Class weighting buys recall on rare classes with false alarms:** 116 Safe rides out of 416 are predicted as
  another class, and 8 of the 11 rides predicted Reckless are not Reckless (precision 0.27).
- **Early stopping is noisy too:** the best epoch ranges from 1 to 17, because the validation part holds about one
  Reckless ride per fold.
