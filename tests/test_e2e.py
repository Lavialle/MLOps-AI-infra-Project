"""End-to-end tests against a DEPLOYED environment (docker compose or Kubernetes).

    E2E_API_URL=http://localhost:8080 E2E_WEBAPP_URL=http://localhost:8081 uv run pytest tests/test_e2e.py

Skipped when E2E_API_URL is not set (unit/integration CI on GitHub-hosted runners).
Run by the CD workflow right after each deployment: a failing test fails the deployment.
"""

import os

import httpx
import pytest

from ride_classifier.client import ApiClient
from ride_classifier.config import CLASSES
from ride_classifier.model import synthetic_ride

API_URL = os.getenv("E2E_API_URL")
WEBAPP_URL = os.getenv("E2E_WEBAPP_URL")
EXPECTED_VERSION = os.getenv("E2E_MODEL_VERSION")  # set by the CD workflow: the version it just deployed

pytestmark = pytest.mark.skipif(not API_URL, reason="E2E_API_URL not set: no deployed environment to test")


@pytest.fixture(scope="module")
def api() -> ApiClient:
    return ApiClient(API_URL, timeout=30)


def test_api_is_ready_with_a_model(api):
    assert httpx.get(f"{API_URL}/ready", timeout=10).status_code == 200
    health = api.health()
    assert health["status"] == "ok" and health["model_loaded"]
    if EXPECTED_VERSION:
        assert health["model_version"] == EXPECTED_VERSION


def test_deployed_model_classifies_a_ride(api):
    out = api.predict(synthetic_ride(n=600))
    assert len(out) == 1
    assert out["prediction"].iloc[0] in CLASSES
    assert abs(out[CLASSES].iloc[0].sum() - 1) < 1e-4


def test_invalid_input_is_rejected_in_prod():
    r = httpx.post(f"{API_URL}/predict", json={"ride_id": "x", "samples": []}, timeout=10)
    assert r.status_code == 422


def test_swagger_and_metrics_are_exposed():
    assert httpx.get(f"{API_URL}/docs", timeout=10).status_code == 200
    assert "api_requests_total" in httpx.get(f"{API_URL}/metrics/", timeout=10).text


@pytest.mark.skipif(not WEBAPP_URL, reason="E2E_WEBAPP_URL not set")
def test_webapp_is_up():
    assert httpx.get(f"{WEBAPP_URL}/_stcore/health", timeout=10).text == "ok"
