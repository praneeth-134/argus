from sdk.argus_sdk import ArgusClient

client = ArgusClient()

# This is the 3-line integration any model would use
result = client.log_prediction(
    model_id=1,
    input_features={"price": 178.3, "volume": 980000, "rsi": 58},
    prediction="SELL",
    confidence=0.79
)

print(result)

# Fetch recent predictions to confirm it worked
predictions = client.get_predictions(model_id=1, n=5)
print(predictions)