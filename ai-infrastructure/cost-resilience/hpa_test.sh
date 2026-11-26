#!/usr/bin/env bash
# Autoscaling test on the kind prod (step 5): metrics-server + HPA, Locust ramp, replicas and CPU sampled every 10 s.
#   bash ai-infrastructure/cost-resilience/hpa_test.sh        (needs `make pipeline` for the Locust payloads)
# Outputs: raw/hpa-replicas.csv (time series) and raw/hpa-load_*.csv (Locust stats).
set -euo pipefail
cd "$(dirname "$0")/../.."
KUBECTL="kubectl --context kind-ride-classifier -n ride-classifier"
OUT=ai-infrastructure/cost-resilience/raw
USERS=${USERS:-60}
DURATION=${DURATION:-6m}
mkdir -p "$OUT"

# metrics-server: kind kubelets use self-signed certificates, hence --kubelet-insecure-tls
kubectl --context kind-ride-classifier apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl --context kind-ride-classifier -n kube-system patch deploy metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' || true
kubectl --context kind-ride-classifier -n kube-system rollout status deploy/metrics-server --timeout=180s
until $KUBECTL top pods > /dev/null 2>&1; do sleep 5; done

$KUBECTL apply -f ai-infrastructure/cost-resilience/hpa.yaml

echo "t_s,replicas_desired,replicas_current,cpu_utilization_pct" > "$OUT/hpa-replicas.csv"
start=$(date +%s)
sample() {
  while true; do
    $KUBECTL get hpa ride-classifier-api -o jsonpath='{.status.desiredReplicas},{.status.currentReplicas},{.status.currentMetrics[0].resource.current.averageUtilization}' \
      | sed "s/^/$(( $(date +%s) - start )),/" >> "$OUT/hpa-replicas.csv"
    echo >> "$OUT/hpa-replicas.csv"
    sleep 10
  done
}
sample & sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT

sleep 30   # baseline before load
uv run --no-sync locust -f loadtest/locustfile.py --host http://localhost:8080 --headless \
  -u "$USERS" -r 1 -t "$DURATION" --csv "$OUT/hpa-load" --csv-full-history
sleep 150  # after load: scale-down (stabilization window 60 s)
echo "done: $OUT/hpa-replicas.csv"
