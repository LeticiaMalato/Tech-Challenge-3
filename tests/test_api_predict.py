"""Tests for the /predict endpoint."""

from app.api.main import app
from fastapi.testclient import TestClient


def test_predict_returns_urgency_for_valid_text():
    """The /predict endpoint should return a valid urgency prediction for valid input text."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "some text"})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert "urgency" in body
    assert body["urgency"] in ["urgent", "attention", "normal"]
    assert "probabilities" in body
    assert len(body["probabilities"]) == 3
    assert all(0 <= v <= 1 for v in body["probabilities"].values())


def test_predict_returns_422_for_invalid_payload():
    """The /predict endpoint should return a 422 error for invalid input."""
    with TestClient(app) as client:
        response = client.post("/predict", json={})  # Empty text is invalid

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body, dict)
    assert "detail" in body
