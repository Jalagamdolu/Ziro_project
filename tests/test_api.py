"""
Automated Test Suite for FastAPI REST Endpoints.

Validates:
1. GET /health
2. POST /predict response schema, status, and probability sum = 1.0.
3. Rejection of invalid symbol (400).
4. Rejection of reference timestamp after target timestamp (400).
5. Rejection of weekend target date (400).
6. Rejection of unsupported horizon > 7 sessions (400).
7. POST /paper/predict stateful record creation.
8. POST /paper/resolve resolution endpoint.
9. GET /paper/performance evaluation metrics endpoint.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert "Logistic Regression" in data["model_name"]

def test_api_predict_success():
    payload = {
        "symbol": "360ONE",
        "reference_timestamp": "2026-09-10 10:30",
        "target_timestamp": "2026-09-15 10:30"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "360ONE"
    assert data["reference_price"] > 0
    assert data["observed_sessions_ahead"] == 3
    assert data["calendar_days_ahead"] == 5
    assert data["predicted_class"] in ["UP", "DOWN", "STABLE"]
    assert 0.0 <= data["probability_down"] <= 1.0
    assert 0.0 <= data["probability_stable"] <= 1.0
    assert 0.0 <= data["probability_up"] <= 1.0
    prob_sum = data["probability_down"] + data["probability_stable"] + data["probability_up"]
    assert abs(prob_sum - 1.0) < 1e-3

def test_api_reject_invalid_symbol():
    payload = {
        "symbol": "TESTSYM",  # Excluded synthetic symbol
        "reference_timestamp": "2026-09-10 10:30",
        "target_timestamp": "2026-09-15 10:30"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "synthetic" in response.json()["detail"].lower() or "not found" in response.json()["detail"].lower()

def test_api_reject_ref_after_target():
    payload = {
        "symbol": "360ONE",
        "reference_timestamp": "2026-09-10 10:30",
        "target_timestamp": "2026-09-08 10:30"  # Before reference
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "before" in response.json()["detail"].lower() or "after" in response.json()["detail"].lower()

def test_api_reject_weekend_target():
    payload = {
        "symbol": "360ONE",
        "reference_timestamp": "2026-09-10 10:30",
        "target_timestamp": "2026-09-12 10:30"  # Saturday
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "saturday" in response.json()["detail"].lower() or "weekend" in response.json()["detail"].lower()

def test_api_reject_unsupported_horizon():
    payload = {
        "symbol": "360ONE",
        "reference_timestamp": "2026-09-10 10:30",
        "target_timestamp": "2026-09-22 10:30"  # 8 business sessions > 7
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "supported scope" in response.json()["detail"].lower() or "horizon" in response.json()["detail"].lower()

def test_api_paper_predict_and_resolve_lifecycle():
    payload = {
        "symbol": "360ONE",
        "reference_timestamp": "2026-09-09 10:30",
        "target_timestamp": "2026-09-10 10:30"
    }
    # Create paper prediction
    response = client.post("/paper/predict", json=payload)
    assert response.status_code == 200
    pred = response.json()
    assert pred["status"] == "OPEN"
    assert pred["is_resolved"] is False
    pid = pred["prediction_id"]
    
    # Resolve single
    resolve_resp = client.post(f"/paper/resolve/{pid}")
    assert resolve_resp.status_code == 200
    res_data = resolve_resp.json()
    assert res_data["status"] == "RESOLVED"
    assert res_data["is_resolved"] is True
    assert res_data["actual_price"] is not None

def test_api_paper_performance():
    response = client.get("/paper/performance")
    assert response.status_code == 200
    perf = response.json()
    assert "total_predictions" in perf
    assert "forward_paper_metrics" in perf
    assert "historical_benchmark" in perf
    assert perf["historical_benchmark"]["macro_f1"] == 0.3563
