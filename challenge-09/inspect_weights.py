"""
File        : inspect_weights.py
Author      : Niklas Anderson with ChatGPT assistance
Created     : Spring 2025
Description : Basic script to inspect the weights of the saved model.
"""

import numpy as np
import sys


def print_minmax(weights, weights_path):
    """Prints the min and max values of each weight array in the .npz file."""
    print(f"\nContents of: {weights_path}\n")
    for name in weights.files:
        arr = weights[name]
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size == 0:
            print("  ⚠️  WARNING: empty array!")
            continue

        min_val = arr.min()
        max_val = arr.max()
        print(f"  min = {min_val:.6f}, max = {max_val:.6f}")


GATE_NAMES = ["i", "f", "c", "o"]

# Match order in which weights are returned by model.get_weights()
LAYER_MAPPING = [
    ("enc1", "lstm"),
    ("enc2", "lstm"),
    ("dec1", "lstm"),
    ("dec2", "lstm"),
    ("dense", "dense"),
]


def print_lstm_weights(W, U, b, prefix):
    hidden_size = b.shape[0] // 4
    print(f"\nLayer: {prefix}")
    print(f"  W shape (input -> 4*hidden): {W.shape}")
    print(f"  U shape (hidden -> 4*hidden): {U.shape}")
    print(f"  b shape: {b.shape} -> hidden size: {hidden_size}")

    for idx, gate in enumerate(GATE_NAMES):
        W_gate = W[:, idx * hidden_size : (idx + 1) * hidden_size]
        U_gate = U[:, idx * hidden_size : (idx + 1) * hidden_size]
        b_gate = b[idx * hidden_size : (idx + 1) * hidden_size]
        print(f"    Gate '{gate}':")
        print(
            f"      W[:, {idx}*{hidden_size}:{(idx+1)*hidden_size}] → shape: {W_gate.shape}"
        )
        print(
            f"      U[:, {idx}*{hidden_size}:{(idx+1)*hidden_size}] → shape: {U_gate.shape}"
        )
        print(
            f"      b[{idx}*{hidden_size}:{(idx+1)*hidden_size}] → shape: {b_gate.shape}"
        )


def print_dense_weights(W, b, prefix):
    print(f"\nLayer: {prefix}")
    print(f"  Dense W shape: {W.shape}")
    print(f"  Dense b shape: {b.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_weights.py path/to/weights.npz")
        sys.exit(1)

    weights_path = sys.argv[1]
    data = np.load(weights_path, allow_pickle=True)
    weights = list(data.values())

    print(f"\nInspecting weights in: {weights_path}")

    i = 0
    for prefix, layer_type in LAYER_MAPPING:
        if layer_type == "lstm":
            W, U, b = weights[i : i + 3]
            print_lstm_weights(W, U, b, prefix)
            i += 3
        elif layer_type == "dense":
            W, b = weights[i : i + 2]
            print_dense_weights(W, b, prefix)
            i += 2


if __name__ == "__main__":
    main()
