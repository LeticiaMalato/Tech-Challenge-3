"""Pydantic models for the /predict endpoint request and response."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    urgency: str
    probabilities: dict[str, float]
