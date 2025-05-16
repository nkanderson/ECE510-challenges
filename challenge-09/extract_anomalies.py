import pandas as pd
import csv

# Load anomaly indices
anomalies = pd.read_csv("data/anomalies.csv")
anom_indices = anomalies["window_index"].tolist()

# Load original NOAA CSV (raw, not preprocessed)
df = pd.read_csv("data/10_100_00_110.csv")

# Extract 24-step windows around each anomaly index
rows = []
for idx in anom_indices:
    window = df.iloc[idx : idx + 24].copy()
    window.insert(0, "window_index", idx)
    window.insert(1, "row_index", range(idx, idx + len(window)))
    rows.append(window)

# Combine into single DataFrame
out_df = pd.concat(rows, ignore_index=True)

# Save anomaly windows
out_df.to_csv("anomaly_windows.csv", index=False)
print("Saved anomaly_windows.csv with full records for each anomalous window.")
