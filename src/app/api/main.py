"""FastAPI application for the urgency triage service."""

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import joblib
from app.api.metrics import REQUEST_COUNT, REQUEST_DURATION
from app.api.schemas import PredictRequest, PredictResponse
from app.models.logistic_classifier import LogisticClassifier
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response as StarletteResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the trained model artifacts once, when the API starts."""
    model_artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR", "model_artifacts")
    app.state.preprocessor = joblib.load(f"{model_artifacts_dir}/preprocessor.joblib")
    app.state.classifier = LogisticClassifier()
    app.state.classifier.load(f"{model_artifacts_dir}/classifier.joblib")
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def track_metrics(
    request: Request, call_next: Callable[[Request], Awaitable[StarletteResponse]]
) -> StarletteResponse:
    """Tracks success/error status for /predict requests."""
    response = await call_next(request)

    if request.url.path == "/predict":
        if response.status_code < 400:
            REQUEST_COUNT.labels(status="success").inc()
        elif response.status_code == 400:
            REQUEST_COUNT.labels(status="error_400").inc()
        elif response.status_code == 422:
            REQUEST_COUNT.labels(status="error_422").inc()
        else:
            REQUEST_COUNT.labels(status="error_other").inc()

    return response


@app.get("/metrics")
async def metrics() -> Response:
    """Exposes Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check: confirms the API process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    """Readiness check: confirms the model artifacts are loaded."""
    if hasattr(request.app.state, "preprocessor") and hasattr(request.app.state, "classifier"):
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model artifacts are not loaded")


@app.post("/predict")
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classifies the urgency of a medical abstract."""
    with REQUEST_DURATION.time():
        preprocessor = request.app.state.preprocessor
        classifier = request.app.state.classifier

        try:
            text_vector = preprocessor.transform([payload.text])
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
