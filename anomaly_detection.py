import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Create simulated smart home IoT data
np.random.seed(42)

data = pd.DataFrame({
    "packet_size": np.random.normal(500, 50, 200),
    "connection_duration": np.random.normal(30, 5, 200),
    "traffic_frequency": np.random.normal(100, 10, 200)
})

# Inject anomalies
anomalies = pd.DataFrame({
    "packet_size": [900, 950, 1000],
    "connection_duration": [80, 90, 100],
    "traffic_frequency": [300, 320, 350]
})

data = pd.concat([data, anomalies])

# Train Isolation Forest
model = IsolationForest(contamination=0.05)
data["anomaly"] = model.fit_predict(data)

# Plot results
plt.scatter(data.index, data["packet_size"],
            c=data["anomaly"], cmap="coolwarm")
plt.title("Smart Home IoT Anomaly Detection")
plt.xlabel("Device Activity Index")
plt.ylabel("Packet Size")
plt.show()

print("Anomaly detection complete.")
