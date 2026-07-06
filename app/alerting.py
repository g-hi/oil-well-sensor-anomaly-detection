from __future__ import annotations


def evaluate_alert_status(probability: float, threshold: float, consecutive_abnormal: int = 0) -> str:
    """Convert a probability into a simple alert status."""
    if probability >= threshold:
        # In a production system, this counter can be persisted and incremented
        # for repeated abnormal readings so one-off spikes do not trigger noise.
        if consecutive_abnormal >= 2:
            return "alert"
        return "alert"

    if probability >= threshold * 0.8:
        return "watch"

    return "normal"
