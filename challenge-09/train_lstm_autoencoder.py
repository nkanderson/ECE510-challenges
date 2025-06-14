"""
File        : train_lstm_autoencoder.py
Author      : Niklas Anderson with ChatGPT assistance
Created     : Spring 2025
Description : Train an LSTM autoencoder on NOAA weather data.
This script preprocesses the NOAA CSV data, builds an LSTM autoencoder model using TensorFlow/Keras,
              trains the model, and saves the trained model, weights, and scaler for external use.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess_data import preprocess_noaa_csv
import joblib
import os

# === Parameters ===
CSV_PATH = "data/10_100_00_110.csv"
WINDOW_SIZE = 24
EPOCHS = 50
BATCH_SIZE = 32
MODEL_SAVE_DIR = "saved_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === Load data ===
sequences, scaler = preprocess_noaa_csv(CSV_PATH, window_size=WINDOW_SIZE)
print(f"Training on {sequences.shape[0]} sequences, shape: {sequences.shape[1:]}")


# === Model architecture ===
def build_lstm_autoencoder(timesteps, features):
    model = models.Sequential(
        [
            layers.Input(shape=(timesteps, features)),
            layers.LSTM(64, activation="tanh", return_sequences=True),
            layers.LSTM(32, activation="tanh", return_sequences=False),
            layers.RepeatVector(timesteps),
            layers.LSTM(32, activation="tanh", return_sequences=True),
            layers.LSTM(64, activation="tanh", return_sequences=True),
            layers.TimeDistributed(layers.Dense(features)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# === Build and train model ===
model = build_lstm_autoencoder(WINDOW_SIZE, sequences.shape[2])
model.summary()

history = model.fit(
    sequences,
    sequences,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True,
)

# === Save model (for optional reloading) ===
model_path = os.path.join(MODEL_SAVE_DIR, "lstm_autoencoder.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

# === Export weights ===
weights = model.get_weights()
weights_path = os.path.join(MODEL_SAVE_DIR, "lstm_weights.npz")
np.savez_compressed(weights_path, *weights)
print(f"Weights saved to: {weights_path}")

# === Save scaler ===
scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.gz")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")
