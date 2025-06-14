"""
File        : lstm_core.py
Author      : Niklas Anderson with ChatGPT assistance
Created     : Spring 2025
Description : Core LSTM cell and dense layer implementations.
This module provides the core LSTM cell and dense layer functionality
for an LSTM autoencoder.
It includes activation functions, matrix-vector multiplication,
and element-wise operations.
"""

import math

try:
    profile
except NameError:

    def profile(func):
        return func


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


@profile
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
        # print(f"W_i: {np.shape(self.W_i)}, b_i: {np.shape(self.b_i)}")

    def _split(self, mat):
        size = self.hidden_size
        return (
            mat[:size],
            mat[size : 2 * size],
            mat[2 * size : 3 * size],
            mat[3 * size : 4 * size],
        )

    @profile
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
        # print(
        #     f"[debug] i={i[0]:.4f}, f={f[0]:.4f}, c_tilde={c_tilde[0]:.4f}, o={o[0]:.4f}"
        # )
        c = elementwise_add(elementwise_mul(f, c_prev), elementwise_mul(i, c_tilde))
        h = elementwise_mul(o, vector_tanh(c))
        return h, c


# --- Dense Layer ---
class Dense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    @profile
    def forward(self, x):
        return elementwise_add(matvec_mul(self.W, x), self.b)
