"""FastAPI application for the urgency triage service."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
from app.api.schemas import PredictRequest, PredictResponse
from app.models.logistic_classifier import LogisticClassifier
from fastapi import FastAPI, HTTPException, Request


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the trained model artifacts once, when the API starts."""
    model_artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR", "model_artifacts")

    app.state.preprocessor = joblib.load(f"{model_artifacts_dir}/preprocessor.joblib")

    app.state.classifier = LogisticClassifier()
    app.state.classifier.load(f"{model_artifacts_dir}/classifier.joblib")

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classifies the urgency of a medical abstract."""
    preprocessor = request.app.state.preprocessor
    classifier = request.app.state.classifier

    try:
        # Preprocess the input text
        text_vector = preprocessor.transform([payload.text])

        # Make the prediction
        urgency = classifier.predict(text_vector)[0]
        probabilities = classifier.predict_proba(text_vector)[0]
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to classify the provided text: {error}",
        ) from error

    return PredictResponse(
        urgency=urgency,
        probabilities={classifier.classes()[i]: p for i, p in enumerate(probabilities)},
    )
