import redis
import json
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, decode_responses=True)


def store_prediction(model_id: int, prediction_data: dict):
    """Store a prediction in Redis with 24hr expiry."""
    key = f"predictions:{model_id}"
    prediction_data["timestamp"] = datetime.utcnow().isoformat()
    r.lpush(key, json.dumps(prediction_data))
    r.expire(key, 86400)  # 24 hours


def get_recent_predictions(model_id: int, n: int = 100):
    """Retrieve the last n predictions for a model from Redis."""
    key = f"predictions:{model_id}"
    raw_predictions = r.lrange(key, 0, n - 1)
    return [json.loads(p) for p in raw_predictions]


def publish_alert(model_id: int, message: str):
    """Publish an alert to a Redis pub/sub channel."""
    channel = f"alerts:{model_id}"
    r.publish(channel, message)