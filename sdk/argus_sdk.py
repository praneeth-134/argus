import requests


class ArgusClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url

    def register_model(self, name: str, version: str, feature_names: list, description: str = None, baseline_data: dict = None):
        payload = {
            "name": name,
            "version": version,
            "description": description,
            "feature_names": feature_names,
            "baseline_data": baseline_data
        }
        response = requests.post(f"{self.base_url}/register_model", json=payload)
        response.raise_for_status()
        return response.json()

    def log_prediction(self, model_id: int, input_features: dict, prediction: str, confidence: float):
        payload = {
            "model_id": model_id,
            "input_features": input_features,
            "prediction": prediction,
            "confidence": confidence
        }
        response = requests.post(f"{self.base_url}/log_prediction", json=payload)
        response.raise_for_status()
        return response.json()

    def get_predictions(self, model_id: int, n: int = 50):
        response = requests.get(f"{self.base_url}/predictions/{model_id}", params={"n": n})
        response.raise_for_status()
        return response.json()