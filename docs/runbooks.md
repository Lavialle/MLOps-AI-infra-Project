# Runbooks

One section per alert of [`infra/monitoring/prometheus/alerts.yml`](../infra/monitoring/prometheus/alerts.yml).
Prod commands assume the kind cluster (`K="kubectl --context kind-ride-classifier -n ride-classifier"`).

## api-down

**Alert `ApiDown`**: an API replica has not answered the Prometheus scrape for 1 minute.

1. `$K get pods -l app=ride-classifier-api`: look for `CrashLoopBackOff`, `OOMKilled` or `Pending`.
2. `$K logs deploy/ride-classifier-api --tail 50`: a `model_load_failed` line means a broken image or model.
3. `$K describe pod <pod>`: if it was OOMKilled, raise the memory limit in `k8s/api.yaml` (measured: ~550 MiB).
4. A bad release: `$K rollout undo deploy/ride-classifier-api`. The CD pipeline also does this automatically when the e2e tests fail.

## latency

**Alert `LatencyP95High`**: `/predict` P95 above 1 s (baseline 340 ms at 16 rides/s).

1. Grafana row 1: does the latency follow the request rate (load) or rise on its own (regression)?
2. Load: `$K scale deploy/ride-classifier-api --replicas=3` and check CPU throttling (`$K top pods` if metrics-server is installed).
3. Very long rides cost more windows: check *Median ride duration* (row 2).
4. Regression after a deployment: roll back (see api-down step 4).

## errors

**Alert `ErrorRateHigh`**: more than 5 % of 5xx responses.

1. `503`: the model is not loaded. `/ready` returns 503 and Kubernetes stops routing traffic to that pod: see api-down.
2. `500`: an exception in the model code. Find the stack trace in `$K logs deploy/ride-classifier-api`, reproduce it locally with the ride, and add a test.
3. 4xx responses (`422`, `413`, `415`) are client errors and are not counted here.

## input-drift

**Alerts `InputGravityShift`, `SensorDropoutsHigh`**: the inputs no longer look like the training data.

- **Gravity shift** (median `ay` above -5 m/s², normally about -9.4): a client sends the accelerometer in g instead
  of m/s², or the phone is mounted differently.
  - Reproduce: `make simulate-unit-bug`.
  - Fix it upstream (client): retraining would only hide a bug.
  - Until the fix lands, the predictions of that client are unreliable.
- **Dropouts** (P90 of the missing share above 5 %; training data has at most 1.6 % per ride): a sensor or connectivity
  problem. The API interpolates gaps, but rides with more than 50 % missing samples are rejected with 422.
  - Reproduce: `make simulate-dropouts`.

## prediction-drift

**Alerts `RecklessShareHigh`, `ConfidenceLow`**: the output distribution moved (baseline on the
test split: 69 % Safe, 5 % Reckless, median confidence 0.54).

1. Check row 2 first: an input incident (above) explains most prediction drifts.
2. If the inputs look normal, the riders or the context changed (season, city, new bike type). Collect labels
   (POST `/feedback`) and run **Actions → Retrain & redeploy** once the new data is versioned with DVC.
3. Right after a deployment: compare with the previous model version (`model_info` panel) and roll back if needed.

## performance

**Alert `FeedbackAccuracyLow`**: less than 50 % of the rides with a true label were classified correctly
(at least 10 labels in the last hour; test accuracy 69 %).

1. Row 4 per class: is one class collapsing (concept drift), or all of them (input incident)?
2. Confirm on more labels before acting: with a few dozen labels per hour the estimate is noisy.
3. Add the labelled rides to the training data (`dvc add`, `dvc push`), run **Retrain & redeploy**, and let the
   evaluation gate decide whether the new model is better.
