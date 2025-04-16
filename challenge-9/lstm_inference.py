import math
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import preprocess_noaa_csv
import sys


# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return math.tanh(x)


def vector_sigmoid(vec):
    return [sigmoid(x) for x in vec]


def vector_tanh(vec):
    return [tanh(x) for x in vec]


def elementwise_add(a, b):
    return [x + y for x, y in zip(a, b)]


def elementwise_mul(a, b):
    return [x * y for x, y in zip(a, b)]


def matvec_mul(matrix, vec):
    return [sum(m * v for m, v in zip(row, vec)) for row in matrix]


# --- LSTM Cell ---
class LSTMCell:
    def __init__(self, input_size, hidden_size, W, U, b):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i, self.W_f, self.W_c, self.W_o = self._split(W)
        self.U_i, self.U_f, self.U_c, self.U_o = self._split(U)
        self.b_i, self.b_f, self.b_c, self.b_o = self._split(b)

    def _split(self, mat):
        size = self.hidden_size
        return (
            mat[:size],
            mat[size : 2 * size],
            mat[2 * size : 3 * size],
            mat[3 * size : 4 * size],
        )

    def step(self, x, h_prev, c_prev):
        i = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_i, x),
                elementwise_add(matvec_mul(self.U_i, h_prev), self.b_i),
            )
        )
        f = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_f, x),
                elementwise_add(matvec_mul(self.U_f, h_prev), self.b_f),
            )
        )
        c_tilde = vector_tanh(
            elementwise_add(
                matvec_mul(self.W_c, x),
                elementwise_add(matvec_mul(self.U_c, h_prev), self.b_c),
            )
        )
        o = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_o, x),
                elementwise_add(matvec_mul(self.U_o, h_prev), self.b_o),
            )
        )
        c = elementwise_add(elementwise_mul(f, c_prev), elementwise_mul(i, c_tilde))
        h = elementwise_mul(o, vector_tanh(c))
        return h, c


# --- Dense layer ---
class Dense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return elementwise_add(matvec_mul(self.W, x), self.b)


# --- LSTM Autoencoder Wrapper ---
class LSTMAutoencoder:
    def __init__(self, weights_path):
        weights = np.load(weights_path, allow_pickle=True)
        weights = [w.tolist() for w in weights.values()]

        self.W1, self.U1, self.b1 = weights[0], weights[1], weights[2]
        self.W2, self.U2, self.b2 = weights[3], weights[4], weights[5]
        self.W3, self.U3, self.b3 = weights[6], weights[7], weights[8]
        self.W4, self.U4, self.b4 = weights[9], weights[10], weights[11]
        self.Wd, self.bd = weights[12], weights[13]

        input_size = len(self.W1[0])
        self.hidden1 = len(self.b1) // 4
        self.hidden2 = len(self.b2) // 4
        self.hidden3 = len(self.b3) // 4
        self.hidden4 = len(self.b4) // 4

        self.lstm1 = LSTMCell(input_size, self.hidden1, self.W1, self.U1, self.b1)
        self.lstm2 = LSTMCell(self.hidden1, self.hidden2, self.W2, self.U2, self.b2)
        self.lstm3 = LSTMCell(self.hidden2, self.hidden2, self.W3, self.U3, self.b3)
        self.lstm4 = LSTMCell(self.hidden2, self.hidden1, self.W4, self.U4, self.b4)
        self.dense = Dense(self.Wd, self.bd)

    def reconstruct(self, sequence):
        h1 = [0.0] * self.hidden1
        c1 = [0.0] * self.hidden1
        h2 = [0.0] * self.hidden2
        c2 = [0.0] * self.hidden2

        for x in sequence:
            h1, c1 = self.lstm1.step(x, h1, c1)
            h2, c2 = self.lstm2.step(h1, h2, c2)

        repeated = [h2 for _ in range(len(sequence))]
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
    sequences, _ = preprocess_noaa_csv("data/10_100_00_110.csv", window_size=24)

    errors = []
    for i, seq in enumerate(sequences):
        recon = model.reconstruct(seq.tolist())
        error = model.compute_mse(seq.tolist(), recon)
        errors.append((i, error))

    if "--csv" in sys.argv:
        print("window_index,mse")
        for i, err in errors:
            print(f"{i},{err:.6f}")
    else:
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
