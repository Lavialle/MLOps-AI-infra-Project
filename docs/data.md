# Data

## Sources

| File | Rows | Rides | Columns |
|------|------|-------|---------|
| `GBG500.parquet` | 1,529,547 | 500 | `ride_id, time_index, ax, ay, az, rx, ry, rz` |
| `GBG500_labels.csv` | 500 | 500 | `RIDE_ID, LABEL` |
| `SRV3490.parquet` | 16,892,002 | 3,490 | `ride_id, crash_ride_id, city, time_index, ax, ay, az, rx, ry, rz, topple` |

- `time_index` always steps by 100 (assumed to be milliseconds, i.e. 10 Hz).
- GBG rides contain 1,807–4,767 samples each (median 2,906).
- `ax, ay, az` come from the accelerometer (m/s², gravity included: `ay ≈ -9.4` at rest). `rx, ry, rz` come from the gyroscope.

## Labels (GBG500)

| Label | Rides | Share |
|-------|-------|-------|
| Safe | 416 | 83.2 % |
| Safe overall | 38 | 7.6 % |
| Bad weather | 22 | 4.4 % |
| Tandem | 18 | 3.6 % |
| Reckless | 6 | 1.2 % |

The classes are heavily imbalanced: always predicting "Safe" already gives 83 % accuracy.
For that reason we report **macro-F1** and **balanced accuracy**, and use stratified splits by ride.

## Data quality

- **Sensor dropouts:** 10,201 rows (0.7 %) have NaN on all 6 channels at once. They are spread over 439 rides,
  at most 1.6 % of any single ride, and scattered as single samples.
  `ride_classifier.data.fill_gaps` interpolates them linearly inside each ride.
  A ride with more than `max_missing_share` (5 %) missing samples is rejected by `validate`.
- Labels and rides match 1:1 (500 ride ids). There are no duplicated ids and no unknown classes.

## Split (`python -m ride_classifier.data`, DVC stage `split`)

The split is stratified **by ride**, so no ride appears in two splits. It keeps at least one ride of each class in val and test.

| Split | Rides | Safe | Safe overall | Bad weather | Tandem | Reckless | Windows (256 × 6) |
|-------|-------|------|--------------|-------------|--------|----------|-------------------|
| train | 350 | 292 | 26 | 16 | 12 | 4 | 7,833 |
| val   | 75  | 62  | 6  | 3  | 3  | 1 | 1,679 |
| test  | 75  | 62  | 6  | 3  | 3  | 1 | 1,687 |

Val and test hold a single Reckless ride each, so per-class metrics on that class are very noisy.
Report macro-F1 and look at the confusion matrix rather than relying on one number.

## Windows (`ride_classifier.features`)

1. Each channel is standardised with mean and std **fitted on train only**. The `ChannelScaler` is saved with the model in step 2.
2. Each ride is cut into windows of 256 samples (25.6 s) with stride 128. A ride shorter than one window is edge-padded.
3. The model predicts each window. The ride prediction is the mean of its window probabilities (`aggregate_by_ride`).

## SRV3490 (auxiliary)

- 698 rides contain a user-confirmed crash (`topple = 1` on ~1 timestep). Each crash ride has 4 baseline rides linked through `crash_ride_id`.
- `city` is coded 0–7.
- This dataset is not required for the 5-class task. It may be used to pretrain an encoder.

## Storage

Raw files are placed in `data/raw/` and versioned with `dvc add`. Only the small `.dvc` pointer files are committed.
`dvc push` uploads the data to the S3 bucket `dvc`. `dvc repro` rebuilds `data/processed/` and later stages.

The DVC remote `storage` points to `s3://dvc` on the dev S3 (`http://localhost:8333`). Each teammate sets the credentials once.
They go in `.dvc/config.local`, which git ignores:

```bash
uv run dvc remote modify --local storage access_key_id mlops
uv run dvc remote modify --local storage secret_access_key mlops-secret
make pull      # dvc pull: downloads data/raw/*
make pipeline  # dvc repro
```
