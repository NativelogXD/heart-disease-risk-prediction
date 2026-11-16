"""End-to-end HTTP integration tests against the running FastAPI prediction service."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("SERVICE_URL", "http://localhost:9696")

LOW_RISK_PATIENT: dict[str, Any] = {
    "age": 41,
    "sex": "female",
    "cp": "atypical_angina",
    "trestbps": 118.0,
    "chol": 204.0,
    "fbs": "lte_120",
    "restecg": "normal",
    "thalach": 176.0,
    "exang": "no",
    "oldpeak": 0.0,
    "slope": "upsloping",
    "ca": 0.0,
    "thal": "normal",
    "clinical_center": "cleveland",
}

HIGH_RISK_PATIENT: dict[str, Any] = {
    "age": 64,
    "sex": "male",
    "cp": "asymptomatic",
    "trestbps": 152.0,
    "chol": 286.0,
    "fbs": "gt_120",
    "restecg": "lv_hypertrophy",
    "thalach": 108.0,
    "exang": "yes",
    "oldpeak": 2.8,
    "slope": "flat",
    "ca": 2.0,
    "thal": "reversible_defect",
    "clinical_center": "cleveland",
}

INCOMPLETE_PATIENT: dict[str, Any] = {
    "age": 55,
    "sex": "male",
    "cp": "asymptomatic",
    "clinical_center": "switzerland",
}

INVALID_PATIENT: dict[str, Any] = {
    "age": 55,
    "sex": "male",
    "cp": "invalid_chest_pain_type",
    "unexpected_field": 123,
}


def run_integration_checks(base_url: str = BASE_URL) -> None:
    """Verify /health, low/high/incomplete /predict scoring, and 422 schema rejection."""
    health_resp = requests.get(f"{base_url}/health", timeout=10)
    assert health_resp.status_code == 200, f"Unexpected /health status: {health_resp.status_code}"
    health_data = health_resp.json()
    assert health_data["status"] == "ok" and health_data["model_loaded"] is True
    logger.info("Health check passed: %s", health_data)

    low_resp = requests.post(f"{base_url}/predict", json=LOW_RISK_PATIENT, timeout=10)
    assert low_resp.status_code == 200, f"Low-risk request failed: {low_resp.text}"
    low_pred = low_resp.json()
    logger.info("Low-risk patient response: %s", low_pred)
    assert low_pred["heart_disease_probability"] < 0.50

    high_resp = requests.post(f"{base_url}/predict", json=HIGH_RISK_PATIENT, timeout=10)
    assert high_resp.status_code == 200, f"High-risk request failed: {high_resp.text}"
    high_pred = high_resp.json()
    logger.info("High-risk patient response: %s", high_pred)
    assert high_pred["heart_disease_probability"] > 0.50
    assert high_pred["heart_disease_prediction"] is True

    inc_resp = requests.post(f"{base_url}/predict", json=INCOMPLETE_PATIENT, timeout=10)
    assert inc_resp.status_code == 200, f"Incomplete patient request failed: {inc_resp.text}"
    logger.info("Incomplete patient (missingness-aware) response: %s", inc_resp.json())

    bad_resp = requests.post(f"{base_url}/predict", json=INVALID_PATIENT, timeout=10)
    assert bad_resp.status_code == 422, f"Expected 422 for invalid payload, got {bad_resp.status_code}"
    logger.info("Schema rejection test passed with HTTP 422.")


if __name__ == "__main__":
    try:
        run_integration_checks()
    except (requests.RequestException, AssertionError) as exc:
        logger.error("Service integration tests failed: %s", exc)
        sys.exit(1)
    logger.info("All integration tests passed successfully.")
