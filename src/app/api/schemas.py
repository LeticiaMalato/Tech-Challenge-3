"""Pydantic models for the /predict endpoint request and response."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    urgency: str
    probabilities: dict[str, float]
