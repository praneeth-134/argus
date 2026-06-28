from sdk.argus_sdk import ArgusClient
import numpy as np

client = ArgusClient()

np.random.seed(99)

# Simulate 30 "current" predictions where price has drifted upward
# (baseline was centered at 100, this is centered at 150 — same shift as our earlier test)
drifted_prices = np.random.normal(150, 10, 30)
volumes = np.random.normal(1000000, 50000, 30)

for price, volume in zip(drifted_prices, volumes):
    client.log_prediction(
        model_id=1,
        input_features={"price": float(price), "volume": float(volume)},
        prediction="BUY" if price > 145 else "HOLD",
        confidence=0.85
    )

print("Logged 30 drifted predictions.")