"""Tests for the /predict endpoint."""

from app.api.main import app
from fastapi.testclient import TestClient


class FakePreprocessor:
    """Stand-in preprocessor so API tests don't depend on real model artifacts."""

    def transform(self, texts):
        return texts


class FakeClassifier:
    """Stand-in classifier so API tests don't depend on real model artifacts."""

    def predict(self, features):
        return ["urgent"]

    def predict_proba(self, features):
        return [[0.7, 0.2, 0.1]]

    def classes(self):
        return ["urgent", "attention", "normal"]


def test_predict_returns_urgency_for_valid_text():
    """The /predict endpoint should return a valid urgency prediction for valid input text."""
    with TestClient(app) as client:
        client.app.state.preprocessor = FakePreprocessor()
        client.app.state.classifier = FakeClassifier()

        response = client.post("/predict", json={"text": "some text"})

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in ["urgent", "attention", "normal"]
    assert len(body["probabilities"]) == 3
    assert all(0 <= v <= 1 for v in body["probabilities"].values())


def test_predict_returns_422_for_invalid_payload():
    """The /predict endpoint should return a 422 error for a missing required field."""
    with TestClient(app) as client:
        response = client.post("/predict", json={})

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body, dict)
    assert "detail" in body
