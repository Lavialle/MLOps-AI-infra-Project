#!/usr/bin/env bash
# Deliberate failing deployment on the kind prod (step 5): a new revision whose pods cannot load the model.
# Same mechanism as cd.yml: the rollout does not complete -> `kubectl rollout undo`.
#   bash ai-infrastructure/cost-resilience/rollback_test.sh
# Output: raw/rollback-probe.csv (HTTP status of /predict every second) and raw/rollback-events.csv (timestamps).
set -euo pipefail
cd "$(dirname "$0")/../.."
KUBECTL="kubectl --context kind-ride-classifier -n ride-classifier"
OUT=ai-infrastructure/cost-resilience/raw
API=http://localhost:8080
mkdir -p "$OUT"

# A small real ride from the test split: the probe checks that predictions keep being served, not only /health
uv run --no-sync python - > "$OUT/.probe-ride.json" <<'EOF'
import json
import pandas as pd
from ride_classifier.config import CHANNELS, RIDE_ID, ROOT, TIME, load_params
df = pd.read_parquet(ROOT / load_params()["data"]["test_path"], columns=[RIDE_ID, TIME, *CHANNELS])
ride_id, ride = next(iter(df.groupby(RIDE_ID)))
print(json.dumps({"ride_id": ride_id, "samples": json.loads(ride[[TIME, *CHANNELS]].to_json(orient="records"))}))
EOF

t0=$(date +%s.%N)
now() { echo "$(date +%s.%N) - $t0" | bc; }
echo "t_s,http_status" > "$OUT/rollback-probe.csv"
probe() {
  while true; do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' -H 'Content-Type: application/json' \
      --data @"$OUT/.probe-ride.json" "$API/predict" || echo 000)
    echo "$(now),$code" >> "$OUT/rollback-probe.csv"
    sleep 1
  done
}
probe & prober=$!
trap 'kill $prober 2>/dev/null || true; rm -f "$OUT/.probe-ride.json"' EXIT

echo "t_s,event" > "$OUT/rollback-events.csv"
event() { echo "$(now),$1" >> "$OUT/rollback-events.csv"; echo "[$(now)s] $1"; }

sleep 10
event "broken revision applied (MODEL_URI points to a missing model)"
$KUBECTL set env deploy/ride-classifier-api MODEL_URI=/app/missing-model
# Same detection as CD: the rollout must complete within a timeout, otherwise it is broken
if $KUBECTL rollout status deploy/ride-classifier-api --timeout=90s; then
  event "UNEXPECTED: broken rollout completed"
else
  event "rollout not complete after 90 s timeout: broken deployment detected"
fi
event "kubectl rollout undo"
$KUBECTL rollout undo deploy/ride-classifier-api
$KUBECTL rollout status deploy/ride-classifier-api --timeout=180s
event "previous revision fully rolled back"
sleep 10
event "end"
