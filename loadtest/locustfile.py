"""Load test.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 20 -r 5 -t 1m --csv reports/load

Then read reports/load_stats.csv: columns "95%" (P95, ms) and "Requests/s".
Payloads are real rides from the test split (run `make pipeline` first).
"""

import json
import random

import pandas as pd
from locust import HttpUser, between, task

from ride_classifier.config import CHANNELS, RIDE_ID, ROOT, TIME, load_params

_test = pd.read_parquet(ROOT / load_params()["data"]["test_path"], columns=[RIDE_ID, TIME, *CHANNELS])
POOL = [
    {"ride_id": ride_id, "samples": json.loads(ride[[TIME, *CHANNELS]].to_json(orient="records"))}
    for ride_id, ride in _test.groupby(RIDE_ID)
]


class Classifier(HttpUser):
    wait_time = between(0.5, 1.5)  # a ride is ~3,000 samples: far heavier than one transaction

    @task(20)
    def predict(self):
        self.client.post("/predict", json=random.choice(POOL))

    @task(1)
    def health(self):
        self.client.get("/health")
