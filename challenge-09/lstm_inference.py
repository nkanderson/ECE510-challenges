import math
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import preprocess_noaa_csv
import sys
import csv
from lstm_core import (
    LSTMCell,
    Dense,
)


try:
    profile
except NameError:

    def profile(func):
        return func


# --- LSTM Autoencoder Wrapper ---
class LSTMAutoencoder:
    def __init__(self, weights_path):
        weights = np.load(weights_path, allow_pickle=True)
        weights = [w.tolist() for w in weights.values()]

        def T(mat):  # safe transpose helper
            return list(map(list, zip(*mat)))

        # self.W1, self.U1, self.b1 = weights[0], weights[1], weights[2]
        # self.W2, self.U2, self.b2 = weights[3], weights[4], weights[5]
        # self.W3, self.U3, self.b3 = weights[6], weights[7], weights[8]
        # self.W4, self.U4, self.b4 = weights[9], weights[10], weights[11]
        # self.Wd, self.bd = weights[12], weights[13]

        # Transpose all weight matrices
        self.W1 = T(weights[0])
        self.U1 = T(weights[1])
        self.b1 = weights[2]

        self.W2 = T(weights[3])
        self.U2 = T(weights[4])
        self.b2 = weights[5]

        self.W3 = T(weights[6])
        self.U3 = T(weights[7])
        self.b3 = weights[8]

        self.W4 = T(weights[9])
        self.U4 = T(weights[10])
        self.b4 = weights[11]

        self.Wd = T(weights[12])
        self.bd = weights[13]

        # print(f"W2: {np.shape(self.W2)}")
        # print(f"U2: {np.shape(self.U2)}")
        # print(f"b2: {np.shape(self.b2)}")

        input_size = len(self.W1[0])
        self.hidden1 = len(self.b1) // 4
        self.hidden2 = len(self.b2) // 4
        self.hidden3 = len(self.b3) // 4
        self.hidden4 = len(self.b4) // 4

        # print("hidden1:", self.hidden1)
        # print("hidden2:", self.hidden2)
        # print("hidden3:", self.hidden3)
        # print("hidden4:", self.hidden4)

        self.lstm1 = LSTMCell(input_size, self.hidden1, self.W1, self.U1, self.b1)
        self.lstm2 = LSTMCell(self.hidden1, self.hidden2, self.W2, self.U2, self.b2)
        self.lstm3 = LSTMCell(self.hidden2, self.hidden2, self.W3, self.U3, self.b3)
        self.lstm4 = LSTMCell(self.hidden2, self.hidden1, self.W4, self.U4, self.b4)
        self.dense = Dense(self.Wd, self.bd)

    @profile
    def reconstruct(self, sequence):
        h1 = [0.0] * self.hidden1
        c1 = [0.0] * self.hidden1
        h2 = [0.0] * self.hidden2
        c2 = [0.0] * self.hidden2

        for x in sequence:
            h1, c1 = self.lstm1.step(x, h1, c1)
            h2, c2 = self.lstm2.step(h1, h2, c2)

        # print(f"final h1 after encoder: len={len(h1)}")

        repeated = [h2 for _ in range(len(sequence))]
        # print(f"h2 = {h2}, len = {len(h2)}")
        # print(f"repeated[0] = {repeated[0]}, len = {len(repeated[0])}")
        h3 = [0.0] * self.hidden3
        c3 = [0.0] * self.hidden3
        h4 = [0.0] * self.hidden4
        c4 = [0.0] * self.hidden4

        output_seq = []
        for x in repeated:
            h3, c3 = self.lstm3.step(x, h3, c3)
            h4, c4 = self.lstm4.step(h3, h4, c4)
            y = self.dense.forward(h4)
            output_seq.append(y)

        return output_seq

    def compute_mse(self, original, reconstructed):
        return sum(
            sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
            for a, b in zip(original, reconstructed)
        ) / len(original)


# --- Main ---
if __name__ == "__main__":
    model = LSTMAutoencoder("saved_model/lstm_weights.npz")

    if "--npz" in sys.argv:
        idx = sys.argv.index("--npz") + 1
        npz_path = sys.argv[idx]
        data = np.load(npz_path)
        sequences = data["sequences"]
    else:
        sequences, _ = preprocess_noaa_csv("data/10_100_00_110.csv", window_size=24)

    errors = []
    for i, seq in enumerate(sequences):
        recon = model.reconstruct(seq.tolist())
        error = model.compute_mse(seq.tolist(), recon)
        errors.append((i, error))

    if "--csv" in sys.argv:
        with open("error_report.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_index", "mse"])
            for i, err in errors:
                writer.writerow([i, f"{err:.6f}"])
        print("Saved error_report.csv")

    elif "--anomalies" in sys.argv:
        threshold = np.percentile([err for _, err in errors], 95)
        with open("anomalies.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_index", "mse"])
            for i, err in errors:
                if err > threshold:
                    writer.writerow([i, f"{err:.6f}"])
        print("Saved anomalies.csv")

    elif "--plot" in sys.argv:
        x_vals = [i for i, _ in errors]
        y_vals = [err for _, err in errors]

        plt.figure(figsize=(12, 4))
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", label="Reconstruction MSE")
        plt.axhline(
            y=np.percentile(y_vals, 95),
            color="r",
            linestyle="--",
            label="95th percentile",
        )
        plt.xlabel("Window Index")
        plt.ylabel("MSE")
        plt.title("Reconstruction Error per Window")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        for i, err in errors:
            print(f"Window {i:03d}: MSE = {err:.6f}")
