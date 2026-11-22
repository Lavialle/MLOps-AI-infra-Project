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

## SRV3490 (auxiliary)

- 698 rides contain a user-confirmed crash (`topple = 1` on ~1 timestep). Each crash ride has 4 baseline rides linked through `crash_ride_id`.
- `city` is coded 0–7.
- This dataset is not required for the 5-class task. It may be used to pretrain an encoder.

## Storage

Raw files are placed in `data/raw/` and versioned with `dvc add`. Only the small `.dvc` pointer files are committed.
`dvc push` uploads the data to the S3 bucket `dvc`. `dvc repro` rebuilds `data/processed/` and later stages.
