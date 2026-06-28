from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_register_model():
    payload = {
        "name": "Test Model",
        "version": "1.0",
        "description": "A test model",
        "feature_names": ["a", "b", "c"]
    }
    response = client.post("/register_model", json=payload)
    assert response.status_code == 200
    assert "model_id" in response.json()


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_log_prediction():
    payload = {
        "model_id": 1,
        "input_features": {"x": 1.0},
        "prediction": "BUY",
        "confidence": 0.8
    }
    response = client.post("/log_prediction", json=payload)
    assert response.status_code == 200