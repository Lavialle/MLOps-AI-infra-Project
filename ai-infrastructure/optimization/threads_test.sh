#!/usr/bin/env bash
# Serving test: PyTorch intra-op threads vs the 1-CPU limit of an API pod, on the kind prod.
# By default torch starts one thread per physical core of the node (8 on the Ryzen 7 5800H) inside a 1-CPU quota.
# Same Locust load with the default and with OMP_NUM_THREADS=1; the HPA is pinned to 2 replicas so that only the
# thread count changes. CFS throttling counters of the pods' cgroups are read before and after each run.
#   bash ai-infrastructure/optimization/threads_test.sh        (needs `make pipeline` for the Locust payloads)
# Outputs: raw/threads-<config>-<repeat>_stats.csv (Locust) and raw/threads-throttling.csv.
# Prod is restored at the end (no OMP_NUM_THREADS, HPA 2-6).
set -euo pipefail
cd "$(dirname "$0")/../.."
K="kubectl --context kind-ride-classifier -n ride-classifier"
OUT=ai-infrastructure/optimization/raw
USERS=${USERS:-20}
DURATION=${DURATION:-2m}
REPEATS=${REPEATS:-2}
mkdir -p "$OUT"

restore() {
  $K set env deploy/ride-classifier-api OMP_NUM_THREADS- > /dev/null || true
  $K rollout status deploy/ride-classifier-api --timeout=180s > /dev/null || true
  $K patch hpa ride-classifier-api -p '{"spec":{"minReplicas":2,"maxReplicas":6}}' > /dev/null || true
}
trap restore EXIT
$K patch hpa ride-classifier-api -p '{"spec":{"minReplicas":2,"maxReplicas":2}}'

settle() {  # rollout done and only the 2 new pods left
  $K rollout status deploy/ride-classifier-api --timeout=180s
  until [ "$($K get pod -l app=ride-classifier-api --no-headers | wc -l)" -eq 2 ]; do sleep 2; done
}

throttling() {  # summed over the API pods: periods, throttled periods, throttled microseconds
  for p in $($K get pod -l app=ride-classifier-api -o name); do $K exec "${p#pod/}" -- cat /sys/fs/cgroup/cpu.stat; done \
    | awk '/^nr_periods/{p+=$2} /^nr_throttled/{t+=$2} /^throttled_usec/{u+=$2} END{print p","t","u}'
}

locust() {
  uv run --no-sync locust -f loadtest/locustfile.py --host http://localhost:8080 --headless --only-summary "$@" \
    > /dev/null 2>&1 || true
}

echo "config,repeat,threads_in_pod,periods_before,throttled_before,throttled_us_before,periods_after,throttled_after,throttled_us_after" \
  > "$OUT/threads-throttling.csv"

measure() {
  local config=$1 pod threads
  settle
  pod=$($K get pod -l app=ride-classifier-api -o jsonpath='{.items[0].metadata.name}')
  threads=$($K exec "$pod" -- python -c "import torch; print(torch.get_num_threads())")
  echo "[$config] torch threads in pod: $threads"
  for r in $(seq 1 "$REPEATS"); do
    locust -u 5 -r 5 -t 20s   # warm-up
    before=$(throttling)
    locust -u "$USERS" -r 5 -t "$DURATION" --csv "$OUT/threads-$config-$r"
    after=$(throttling)
    echo "$config,$r,$threads,$before,$after" >> "$OUT/threads-throttling.csv"
    echo "[$config] run $r done"
  done
}

measure default
$K set env deploy/ride-classifier-api OMP_NUM_THREADS=1
measure omp1
echo "done: $OUT/threads-*"
