from fastapi import FastAPI, HTTPException

from app.model_service import model_service
from app.schemas import PredictionRequest, PredictionResponse
from app.utils import PredictionError


app = FastAPI(
    title="Oil Well Anomaly Detection API",
    version="0.1.0",
    description="Simple inference layer for industrial sensor anomaly detection.",
)


@app.get("/health")
def health() -> dict[str, object]:
    """Simple health check for deployment and smoke testing."""
    return {
        "status": "ok",
        "model_loaded": bool(model_service.model_loaded),
        "threshold_loaded": bool(model_service.threshold_loaded),
        "feature_columns_loaded": bool(model_service.feature_columns_loaded),
        "number_of_features": int(model_service.number_of_features or 0),
        "model_source": model_service.model_source,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run preprocessing and inference for a batch of sensor readings."""
    try:
        result = model_service.predict(payload.sensor_readings, metadata=payload.metadata or {})
        return PredictionResponse(**result)
    except PredictionError as exc:
        # Client-side error due to malformed/missing features
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive error handling
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
