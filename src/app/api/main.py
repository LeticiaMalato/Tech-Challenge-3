"""FastAPI application for the urgency triage service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
from app.api.schemas import PredictRequest, PredictResponse
from app.models.logistic_classifier import LogisticClassifier
from fastapi import FastAPI, Request


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the trained model artifacts once, when the API starts."""
    app.state.preprocessor = joblib.load("model_artifacts/preprocessor.joblib")

    app.state.classifier = LogisticClassifier()
    app.state.classifier.load("model_artifacts/classifier.joblib")

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classifies the urgency of a medical abstract."""
    preprocessor = request.app.state.preprocessor
    classifier = request.app.state.classifier

    # Preprocess the input text
    text_vector = preprocessor.transform([payload.text])

    # Make the prediction
    urgency = classifier.predict(text_vector)[0]
    probabilities = classifier.predict_proba(text_vector)[0]

    return PredictResponse(
        urgency=urgency,
        probabilities={classifier.classifier.classes_[i]: p for i, p in enumerate(probabilities)},
    )
