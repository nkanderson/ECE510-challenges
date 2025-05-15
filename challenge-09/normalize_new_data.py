import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 24

# --- Load new data ---
df = pd.read_csv("data/30_-160_20_-150.csv")

# --- Load saved scaler (from training phase) ---
scaler = joblib.load("saved_model/scaler.gz")

# --- Select columns used in training ---
selected_columns = [
    "AIR_TEMP",
    "SEA_SURF_TEMP",
    "SEA_LVL_PRES",
    "WIND_SPEED",
    "WAVE_HGT",
    "WAVE_PERIOD",
]
data = df[selected_columns].copy()

# --- Drop rows with missing values ---
data.dropna(inplace=True)

# --- Normalize with existing scaler ---
data_scaled = scaler.transform(data)

# --- Create overlapping sequences ---
sequences = []
for i in range(len(data_scaled) - WINDOW_SIZE + 1):
    window = data_scaled[i : i + WINDOW_SIZE]
    sequences.append(window)

sequences = np.array(sequences)
print(
    f"Prepared {len(sequences)} sequences from new data with shape: {sequences.shape}"
)

# Optional: save to .npz for use in inference script
np.savez("new_data_sequences.npz", sequences=sequences)
print("Saved new_data_sequences.npz")
