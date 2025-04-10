# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Step 1: Load and prepare the data
# Assume your CSV file (e.g., "weather_data.csv") has columns: "time", "temperature", "pressure", and "humidity"
df = pd.read_csv("weather_data.csv")

# Convert the 'time' column to datetime objects for easier processing
df['time'] = pd.to_datetime(df['time'])

# Optional: You can extract additional time-based features if relevant
# For example, we extract the hour of the day to capture diurnal patterns
df['hour'] = df['time'].dt.hour

# Step 2: Select features for anomaly detection
# Depending on the available data and domain expertise, you might choose to use time-based features.
# In this example, we use "hour", "temperature", "pressure", and "humidity".
features = df[['hour', 'temperature', 'pressure', 'humidity']]

# Step 3: Configure and fit the Isolation Forest model
# 'n_estimators': number of trees in the ensemble.
# 'contamination': the proportion of expected anomalies in the data. Adjust this according to domain knowledge.
# 'random_state': ensures reproducibility of results.
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(features)

# The model assigns a label of 1 for normal points and -1 for anomalies.
# Add these predictions to our dataframe.
df['anomaly_label'] = iso_forest.predict(features)

# For clarity, mark anomalies with a more descriptive label
df['anomaly_flag'] = df['anomaly_label'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# Step 4: Examine detected anomalies
anomalies = df[df['anomaly_label'] == -1]
print("Detected anomalies:")
print(anomalies)

# Step 5: (Optional) Visualization
# Plot temperature over time, marking anomalies
plt.figure(figsize=(12,6))
plt.plot(df['time'], df['temperature'], label="Temperature", color="blue", linewidth=1)
plt.scatter(anomalies['time'], anomalies['temperature'], color="red", label="Anomaly", marker="o", s=50)
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Readings with Anomalies Marked")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
