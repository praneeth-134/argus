from sdk.argus_sdk import ArgusClient
import numpy as np

client = ArgusClient()

np.random.seed(42)
baseline_price = np.random.normal(100, 10, 500).tolist()
baseline_volume = np.random.normal(1000000, 50000, 500).tolist()

result = client.register_model(
    name="AI Trader",
    version="1.0",
    feature_names=["price", "volume"],
    description="DDPG/PPO trading agent",
    baseline_data={
        "price": baseline_price,
        "volume": baseline_volume
    }
)
print(result)