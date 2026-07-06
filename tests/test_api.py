from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction_payload():
    payload = {
        "sensor_readings": {
            "P-PDG": 120.0,
            "P-TPT": 70.0,
            "T-TPT": 85.0,
            "P-MON-CKP": 40.0,
            "T-JUS-CKP": 50.0,
            "P-JUS-CKGL": 30.0,
            "T-JUS-CKGL": 60.0,
            "QGL": 8.0,
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "alert_status" in data
    assert "message" in data
