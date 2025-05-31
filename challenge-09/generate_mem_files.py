import os
import numpy as np
import argparse
from utils import save_mem_file

GATE_NAMES = ["i", "f", "c", "o"]  # input, forget, cell, output


def save_lstm_weights(weights, prefix, output_dir):
    W, U, b = weights  # W: input weights, U: recurrent weights, b: biases
    hidden_size = b.shape[0] // 4
    os.makedirs(output_dir, exist_ok=True)

    for idx, gate in enumerate(GATE_NAMES):
        # input weights
        save_mem_file(
            W[:, idx * hidden_size : (idx + 1) * hidden_size],
            os.path.join(output_dir, f"{prefix}_W_{gate}.mem"),
            q_format=(2, 14),
        )

        # recurrent weights
        save_mem_file(
            U[:, idx * hidden_size : (idx + 1) * hidden_size],
            os.path.join(output_dir, f"{prefix}_U_{gate}.mem"),
            q_format=(2, 14),
        )

        # biases
        save_mem_file(
            b[idx * hidden_size : (idx + 1) * hidden_size],
            os.path.join(output_dir, f"{prefix}_b_{gate}.mem"),
            q_format=(2, 14),
        )


def save_dense_weights(weights, prefix, output_dir):
    W, b = weights
    os.makedirs(output_dir, exist_ok=True)
    save_mem_file(W, os.path.join(output_dir, f"{prefix}_W.mem"))
    save_mem_file(b, os.path.join(output_dir, f"{prefix}_b.mem"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default="saved_model/lstm_weights.npz",
        help="Path to .npz weights file",
    )
    parser.add_argument(
        "--outdir", type=str, default="weights", help="Output directory for .mem files"
    )
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    weights = list(data.values())

    # Expected order based on build_lstm_autoencoder
    layer_mapping = [
        ("enc1", save_lstm_weights),
        ("enc2", save_lstm_weights),
        ("dec1", save_lstm_weights),
        ("dec2", save_lstm_weights),
        ("dense", save_dense_weights),
    ]

    i = 0
    for prefix, save_func in layer_mapping:
        if save_func == save_lstm_weights:
            save_func(weights[i : i + 3], prefix, args.outdir)
            i += 3
        elif save_func == save_dense_weights:
            save_func(weights[i : i + 2], prefix, args.outdir)
            i += 2


if __name__ == "__main__":
    main()
