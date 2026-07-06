from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Expected payload for a single sensor reading snapshot."""

    sensor_readings: dict[str, float] = Field(
        ...,
        description="Mapping of sensor names to numerical readings.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata such as well id or timestamp.",
    )


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    """Structured response returned by the inference endpoint."""

    prediction: str
    abnormal_probability: float
    threshold_used: float
    alert_status: str
    top_features: list[FeatureImportance]
    message: str
