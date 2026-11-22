"""Replay test rides against a running API, to feed the monitoring dashboard.

    python -m ride_classifier.simulate --api http://localhost:8080 --n 50
    python -m ride_classifier.simulate --n 50 --unit-bug          # accelerometer sent in g instead of m/s²
    python -m ride_classifier.simulate --n 50 --dropouts 0.2      # 20 % of the samples lost by the sensor

Each ride is sent to POST /predict, then its true label to POST /feedback (the "outcome" layer:
in real life the label arrives later, from a user report or a manual review).
The incidents reproduce what the alerts of infra/monitoring/prometheus/alerts.yml must catch.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd

from ride_classifier.client import ApiClient
from ride_classifier.config import CHANNELS, RIDE_ID, ROOT, TARGET, TIME, load_params

G = 9.81


def apply_incident(ride: pd.DataFrame, unit_bug: bool, dropouts: float, rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt one ride like an upstream incident would."""
    ride = ride.copy()
    if unit_bug:  # a new app version sends the accelerometer in g
        ride[["ax", "ay", "az"]] = ride[["ax", "ay", "az"]] / G
    if dropouts > 0:
        lost = rng.random(len(ride)) < dropouts
        ride.loc[lost, CHANNELS] = np.nan
    return ride


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8080", help="API URL (prod kind: 8080, dev: 8000)")
    parser.add_argument("--n", type=int, default=50, help="number of rides to send")
    parser.add_argument("--unit-bug", action="store_true", help="accelerometer in g instead of m/s²")
    parser.add_argument("--dropouts", type=float, default=0.0, help="share of samples lost (NaN)")
    parser.add_argument("--no-feedback", action="store_true", help="do not send the true labels")
    parser.add_argument("--pause", type=float, default=0.2, help="seconds between rides")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    test = pd.read_parquet(ROOT / load_params()["data"]["test_path"])
    rides = [ride for _, ride in test.groupby(RIDE_ID)]
    client = ApiClient(args.api, timeout=30)

    correct = 0
    for i in range(args.n):
        ride = apply_incident(random.choice(rides), args.unit_bug, args.dropouts, rng)
        truth = ride[TARGET].iloc[0]
        try:
            pred = client.predict(ride[[RIDE_ID, TIME, *CHANNELS]]).iloc[0]["prediction"]
        except RuntimeError as exc:  # e.g. 422 when the incident makes the ride invalid
            print(f"{i + 1:3d}/{args.n} rejected: {exc}")
            continue
        correct += pred == truth
        if not args.no_feedback:
            client.feedback(ride[RIDE_ID].iloc[0], pred, truth)
        print(f"{i + 1:3d}/{args.n} true={truth:12s} predicted={pred:12s}")
        time.sleep(args.pause)
    print(json.dumps({"rides": args.n, "accuracy": round(correct / max(args.n, 1), 3)}))


if __name__ == "__main__":
    main()
