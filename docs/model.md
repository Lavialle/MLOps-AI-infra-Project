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
