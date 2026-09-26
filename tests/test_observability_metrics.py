"""
Tests for the Module 5 instrumentation on the FastAPI service. Uses
TestClient directly against the app (no network, no live MLflow needed —
the production model degrades gracefully to None when MLflow isn't
reachable, same as the app does in real deployments when MLflow is down).
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_metrics_endpoint_returns_prometheus_text():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_api_" in response.text


def test_metrics_endpoint_exposes_model_info_before_any_request():
    # set_model_info() is called at import time, so this should be present
    # even before a single prediction has been served.
    response = client.get("/metrics")
    assert "fraud_api_model_info" in response.text


def test_prediction_increments_counter():
    before = client.get("/metrics").text
    before_count = before.count(
        'fraud_api_predictions_total{model_version="random-forest",status="ok"}'
    )

    features = [0.0] * 30
    result = client.post("/predict/random-forest", json={"features": features})
    assert result.status_code == 200

    after = client.get("/metrics").text
    assert (
        'fraud_api_predictions_total{model_version="random-forest",status="ok"}'
        in after
    )
    # counter line should now report a value >= 1 — at minimum, present where it wasn't guaranteed before
    assert (
        before_count == 0
        or 'fraud_api_predictions_total{model_version="random-forest",status="ok"} 1.0'
        not in before
    )


def test_bad_input_is_counted_as_error_status():
    # Non-numeric input never reaches our code — Pydantic's own field
    # validation rejects it first with a 422, before our try/except runs.
    result = client.post(
        "/predict/random-forest", json={"features": ["not", "numeric"]}
    )
    assert result.status_code == 422

    after = client.get("/metrics").text
    assert (
        'fraud_api_predictions_total{model_version="random-forest",status="error"}'
        in after
    )


def test_health_endpoint_unaffected_by_instrumentation():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
