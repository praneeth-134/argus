from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from cache.redis_client import store_prediction, get_recent_predictions

router = APIRouter()


class PredictionLogRequest(BaseModel):
    model_id: int
    input_features: Dict[str, Any]
    prediction: str
    confidence: float


@router.post("/log_prediction")
def log_prediction(payload: PredictionLogRequest):
    prediction_data = {
        "input_features": payload.input_features,
        "prediction": payload.prediction,
        "confidence": payload.confidence
    }
    store_prediction(payload.model_id, prediction_data)
    return {"message": "Prediction logged successfully"}


@router.get("/predictions/{model_id}")
def get_predictions(model_id: int, n: int = 50):
    predictions = get_recent_predictions(model_id, n)
    return {"model_id": model_id, "count": len(predictions), "predictions": predictions}