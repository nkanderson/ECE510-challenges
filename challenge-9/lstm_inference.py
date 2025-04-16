import math
import numpy as np
import json


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
        # Split weights
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
        # input gate
        i = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_i, x),
                elementwise_add(matvec_mul(self.U_i, h_prev), self.b_i),
            )
        )
        # forget gate
        f = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_f, x),
                elementwise_add(matvec_mul(self.U_f, h_prev), self.b_f),
            )
        )
        # candidate memory
        c_tilde = vector_tanh(
            elementwise_add(
                matvec_mul(self.W_c, x),
                elementwise_add(matvec_mul(self.U_c, h_prev), self.b_c),
            )
        )
        # output gate
        o = vector_sigmoid(
            elementwise_add(
                matvec_mul(self.W_o, x),
                elementwise_add(matvec_mul(self.U_o, h_prev), self.b_o),
            )
        )

        # new cell state
        c = elementwise_add(elementwise_mul(f, c_prev), elementwise_mul(i, c_tilde))
        h = elementwise_mul(o, vector_tanh(c))
        return h, c


# --- Dense layer ---
class Dense:
    def __init__(self, W, b):
        self.W = W  # shape: [output_dim][input_dim]
        self.b = b

    def forward(self, x):
        return elementwise_add(matvec_mul(self.W, x), self.b)


# --- Load weights from .npz file ---
def load_weights(path):
    weights = np.load(path, allow_pickle=True)
    return [w.tolist() for w in weights.values()]


# --- Demo forward pass ---
def run_demo(sequence, weights):
    # Unpack weights
    W1, U1, b1 = weights[0], weights[1], weights[2]  # LSTM encoder 1
    W2, U2, b2 = weights[3], weights[4], weights[5]  # LSTM encoder 2
    W3, U3, b3 = weights[6], weights[7], weights[8]  # LSTM decoder 1
    W4, U4, b4 = weights[9], weights[10], weights[11]  # LSTM decoder 2
    W_dense, b_dense = weights[12], weights[13]  # TimeDistributed(Dense)

    input_size = len(sequence[0])
    hidden1 = len(b1) // 4
    hidden2 = len(b2) // 4

    # Create cells
    lstm1 = LSTMCell(input_size, hidden1, W1, U1, b1)
    lstm2 = LSTMCell(hidden1, hidden2, W2, U2, b2)
    lstm3 = LSTMCell(hidden2, hidden2, W3, U3, b3)
    lstm4 = LSTMCell(hidden2, hidden1, W4, U4, b4)
    dense = Dense(W_dense, b_dense)

    # Encoder forward
    h1 = [0.0] * hidden1
    c1 = [0.0] * hidden1
    h2 = [0.0] * hidden2
    c2 = [0.0] * hidden2

    for x in sequence:
        h1, c1 = lstm1.step(x, h1, c1)
        h2, c2 = lstm2.step(h1, h2, c2)

    # Repeat vector (decoder input)
    repeated = [h2 for _ in range(len(sequence))]

    # Decoder forward
    h3 = [0.0] * hidden2
    c3 = [0.0] * hidden2
    h4 = [0.0] * hidden1
    c4 = [0.0] * hidden1

    output_seq = []
    for x in repeated:
        h3, c3 = lstm3.step(x, h3, c3)
        h4, c4 = lstm4.step(h3, h4, c4)
        y = dense.forward(h4)
        output_seq.append(y)

    return output_seq


# Example usage:
if __name__ == "__main__":
    weights = load_weights("saved_model/lstm_weights.npz")
    # Load a preprocessed sequence from somewhere
    from preprocess_data import preprocess_noaa_csv

    sequences, _ = preprocess_noaa_csv("data/10_100_00_110.csv", window_size=24)
    test_seq = sequences[0].tolist()

    recon = run_demo(test_seq, weights)

    # Compute MSE
    error = sum(
        sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
        for a, b in zip(test_seq, recon)
    ) / len(test_seq)

    print(f"Reconstruction MSE: {error:.6f}")
