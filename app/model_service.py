from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from app.alerting import evaluate_alert_status
from app.utils import PredictionError, get_project_root, load_json, logger


class ModelService:
    """Simple service that loads model artifacts and serves predictions."""

    def __init__(self) -> None:
        self.project_root = get_project_root()
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.model = None
        self.feature_columns: list[str] = []
        self.threshold = 0.65
        self.model_source = "placeholder"
        self.is_ready = False
        self.model_loaded = False
        self.threshold_loaded = False
        self.feature_columns_loaded = False
        self.number_of_features = 0

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        # In a real deployment, the trained model and preprocessing artifacts would
        # be exported into the models/ directory during the training step.
        # Prefer the exact exported model filename from Colab training
        model_path = self.models_dir / "rf_3w_abnormal_detector.joblib"
        if not model_path.exists():
            # fall back to older candidate names for compatibility
            candidate_paths = [
                self.models_dir / "model.joblib",
                self.models_dir / "model.pkl",
                self.project_root / "models" / "model.joblib",
                self.project_root / "outputs" / "model.joblib",
                self.project_root / "output" / "model.joblib",
            ]
            for path in candidate_paths:
                if path.exists():
                    model_path = path
                    break

        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.model_source = str(model_path)
                self.is_ready = True
                self.model_loaded = True
                logger.info("Loaded model artifact from %s", model_path)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Could not load model from %s: %s", model_path, exc)

        # Load explicit feature columns exported from training
        feature_path = self.models_dir / "feature_columns.json"
        if not feature_path.exists():
            feature_path = self.project_root / "feature_columns.json"
        feature_columns = load_json(feature_path)
        if isinstance(feature_columns, list):
            self.feature_columns = [str(item) for item in feature_columns]
            self.feature_columns_loaded = True
            self.number_of_features = len(self.feature_columns)

        # Load the chosen threshold exported from training (key: chosen_threshold)
        threshold_path = self.models_dir / "chosen_threshold.json"
        if not threshold_path.exists():
            threshold_path = self.project_root / "chosen_threshold.json"
        threshold_payload = load_json(threshold_path)
        if isinstance(threshold_payload, dict):
            threshold_value = threshold_payload.get("chosen_threshold")
            if isinstance(threshold_value, (int, float)):
                self.threshold = float(threshold_value)
                self.threshold_loaded = True
            elif isinstance(threshold_payload.get("threshold"), (int, float)):
                self.threshold = float(threshold_payload["threshold"])
                self.threshold_loaded = True

        if not self.feature_columns:
            logger.info("No feature list artifact found; falling back to input-based feature order.")

    def prepare_features(self, sensor_readings: dict[str, float]) -> np.ndarray:
        """Create a feature vector in the order expected by the model."""
        if self.feature_columns:
            feature_names = self.feature_columns
        else:
            feature_names = sorted(sensor_readings.keys())

        missing_features = [feature for feature in feature_names if feature not in sensor_readings]
        if feature_names and not sensor_readings:
            raise PredictionError("No sensor readings were provided.")
        if self.feature_columns and missing_features:
            # Enforce exact feature set when a feature_columns artifact is present
            raise PredictionError(f"Missing required features for model: {missing_features}")

        values: list[float] = []
        for feature_name in feature_names:
            if feature_name in sensor_readings:
                values.append(float(sensor_readings[feature_name]))
            else:
                # If we reach here, either feature_columns was not present or the
                # missing feature was intentionally set to a default of 0.0
                values.append(0.0)

        return np.asarray(values, dtype=float).reshape(1, -1)

    def _placeholder_probability(self, feature_frame: np.ndarray) -> float:
        """Safe heuristic for interview demo purposes when no real model is present."""
        if feature_frame.size == 0:
            return 0.2

        values = feature_frame[0]
        magnitude = float(np.mean(np.abs(values))) if values.size else 0.0
        probability = min(0.99, max(0.05, magnitude / 100.0))
        if magnitude > 70:
            probability = min(0.99, probability + 0.15)
        return round(probability, 4)

    def _top_features(self, sensor_readings: dict[str, float], probability: float) -> list[dict[str, Any]]:
        """Return a small feature-importance list for the response payload."""
        if not sensor_readings:
            return []
        ranked_features = sorted(
            sensor_readings.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        top_three = ranked_features[:3]
        return [
            {
                "feature": feature_name,
                "importance": round(abs(float(value)) / max(1.0, abs(float(value)) + 100.0) + probability * 0.1, 3),
            }
            for feature_name, value in top_three
        ]

    def predict(self, sensor_readings: dict[str, float], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run inference and return a structured prediction payload."""
        if not sensor_readings:
            raise PredictionError("No sensor readings were provided.")

        feature_frame = self.prepare_features(sensor_readings)

        try:
            if self.model is not None:
                probability = float(self.model.predict_proba(feature_frame)[0, 1])
                self.model_source = "trained_model"
                self.model_loaded = True
            else:
                probability = self._placeholder_probability(feature_frame)
                logger.info("Using placeholder probability logic because no trained model was found.")
        except Exception as exc:
            logger.warning("Model inference failed: %s", exc)
            probability = self._placeholder_probability(feature_frame)
        prediction = "abnormal" if probability >= self.threshold else "normal"
        alert_status = evaluate_alert_status(probability, self.threshold)
        if prediction == "abnormal":
            message = "Abnormal behavior detected. Engineering review recommended."
        else:
            message = "Sensor behavior appears normal based on the current threshold."

        # Attempt to extract top features from the trained model if available
        top_features: list[dict[str, Any]] = []
        try:
            if self.model is not None and hasattr(self.model, "feature_importances_") and self.feature_columns:
                importances = getattr(self.model, "feature_importances_")
                if len(importances) == len(self.feature_columns):
                    indices = np.argsort(importances)[::-1][:3]
                    for idx in indices:
                        top_features.append(
                            {
                                "feature": self.feature_columns[int(idx)],
                                "importance": round(float(importances[int(idx)]), 6),
                            }
                        )
        except Exception:
            top_features = []

        if not top_features:
            top_features = self._top_features(sensor_readings, probability)

        return {
            "prediction": prediction,
            "abnormal_probability": round(probability, 4),
            "threshold_used": round(self.threshold, 4),
            "alert_status": alert_status,
            "top_features": top_features,
            "message": message,
        }


model_service = ModelService()
