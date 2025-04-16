import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_noaa_csv(
    csv_path, window_size=24, feature_columns=None, datetime_column="DATE"
):
    if feature_columns is None:
        feature_columns = [
            "AIR_TEMP",
            "SEA_SURF_TEMP",
            "SEA_LVL_PRES",
            "WIND_SPEED",
            "WAVE_HGT",
            "WAVE_PERIOD",
        ]

    df = pd.read_csv(csv_path, parse_dates=[datetime_column])
    df.sort_values(datetime_column, inplace=True)
    df_clean = df[[datetime_column] + feature_columns].dropna()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_columns])
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    df_scaled[datetime_column] = df_clean[datetime_column].values

    def create_sequences(data, window_size):
        sequences = []
        for i in range(len(data) - window_size):
            window = data[i : i + window_size]
            sequences.append(window)
        return np.array(sequences)

    sequence_data = create_sequences(df_scaled[feature_columns].values, window_size)
    return sequence_data, scaler


# Allow running as standalone script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess NOAA CSV for LSTM training"
    )
    parser.add_argument("csv_path", help="Path to NOAA CSV file")
    parser.add_argument(
        "--window_size", type=int, default=24, help="Sliding window size"
    )

    args = parser.parse_args()
    sequences, _ = preprocess_noaa_csv(args.csv_path, window_size=args.window_size)
    print(
        f"Processed {sequences.shape[0]} sequences of shape {sequences.shape[1:]} from '{args.csv_path}'"
    )
