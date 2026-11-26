# Model, training and MLflow

## Pipeline

```
make pull        # dvc pull: raw data from S3
make pipeline    # dvc repro: split -> train -> evaluate (needs `make up`)
make export      # model @staging -> ./model (baked into the API image)
```

| Stage | Module | Output |
|-------|--------|--------|
| split | `ride_classifier.data` | `data/processed/{train,val,test}.parquet` |
| train | `ride_classifier.train` | MLflow run + new registered version of `ride-classifier`, `reports/train.json` |
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

On the frozen **test** split (75 rides), the gate passes only if both hold:

- `test_macro_f1 >= gate.min_macro_f1` (see `params.yaml`), and
- `test_macro_f1 > test_baseline_macro_f1` (the majority-class model scores ~0.18).

Otherwise the script exits with code 1, which stops the CI pipeline. Val and test hold a single Reckless ride each,
so per-class metrics for Reckless are very noisy. Macro F1 and the confusion matrix are the numbers to read.

## Cross-validation: how noisy is the gate?

`make cv` (or `python -m ride_classifier.cross_validate --device cuda`) runs a **5-fold cross-validation stratified
by ride** with the same `params.yaml`: each ride is tested exactly once, by a model that never saw it, so all 6
Reckless rides are tested instead of 1. In each fold, 15 % of all rides is held out from the training folds for early
stopping. It is a measurement only: no MLflow run, no promotion, the frozen split and the gate are unchanged.
Output: [`reports/cv.json`](../reports/cv.json).

Measured 26/11 on the laptop GPU (RTX 3060, 59 s for the 5 folds):

| Fold | Test rides (Reckless) | Macro F1 | Balanced accuracy | Best epoch |
|------|-----------------------|----------|-------------------|------------|
| 1 | 100 (1) | 0.343 | 0.599 | 12 |
| 2 | 100 (2) | 0.274 | 0.405 | 5 |
| 3 | 100 (1) | 0.237 | 0.258 | 1 |
| 4 | 100 (1) | 0.324 | 0.504 | 28 |
| 5 | 100 (1) | 0.437 | 0.459 | 4 |
| **Mean ± std** | 500 (6) | **0.323 ± 0.076** | **0.445 ± 0.126** | |
| Majority baseline | | 0.182 | 0.200 | |

Pooled over the 500 rides (each predicted once):

| | Safe | Safe overall | Bad weather | Tandem | Reckless |
|-|------|--------------|-------------|--------|----------|
| Recall | 0.654 | 0.211 | 0.455 | 0.278 | **0.500** (3 / 6) |

What it shows:
- **The model beats the baseline in every fold** (0.24–0.44 against 0.18): the signal is real.
- **The gate floor (0.30) sits inside the noise**: it passes on 3 folds out of 5. The same training recipe can pass
  or fail depending on which rides land in the test split. The floor should be set from the cross-validated
  mean and standard deviation, not from one 75-ride split.
- **Reckless recall is 0.5, not 1.** The single test ride of the frozen split was one of the easy ones.
- **Class weighting buys recall on rare classes with false alarms:** 144 Safe rides out of 416 are predicted as
  another class, and 21 of the 24 rides predicted Reckless are not Reckless (precision 0.125).
- **Early stopping is noisy too:** the best epoch ranges from 1 to 28, because the validation part holds about one
  Reckless ride per fold.

GPU and CPU runs do not give bit-identical scores (different floating-point kernels), as between ARM and x86.
