# generate_golden_vectors.py

import os
import argparse
import numpy as np
from lstm_core import LSTMCell
from utils import save_mem_file


def generate_vectors(npz_path, weights_path, out_dir, num_windows, fmt):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path)["sequences"]
    print(f"[info] Loaded sequences shape: {data.shape}")
    data = data[:num_windows]  # (num_windows, seq_len, input_dim)

    input_dim = data.shape[2]
    hidden1 = 64
    hidden2 = 32

    # Load weights
    weights = np.load(weights_path)
    # W1, U1, b1 = weights["arr_0"], weights["arr_1"], weights["arr_2"]
    # W2, U2, b2 = weights["arr_3"], weights["arr_4"], weights["arr_5"]
    W1 = np.reshape(weights["arr_0"], (4 * hidden1, input_dim)).tolist()
    U1 = np.reshape(weights["arr_1"], (4 * hidden1, hidden1)).tolist()
    b1 = np.reshape(weights["arr_2"], (4 * hidden1,)).tolist()

    W2 = np.reshape(weights["arr_3"], (4 * hidden2, hidden1)).tolist()
    U2 = np.reshape(weights["arr_4"], (4 * hidden2, hidden2)).tolist()
    b2 = np.reshape(weights["arr_5"], (4 * hidden2,)).tolist()

    lstm1 = LSTMCell(input_dim, hidden1, W1, U1, b1)
    lstm2 = LSTMCell(hidden1, hidden2, W2, U2, b2)

    x_in_lines, h_prev_lines, c_prev_lines = [], [], []
    h_out_lines, c_out_lines = [], []

    for window in data:
        h1 = c1 = np.zeros((hidden1,), dtype=np.float32)
        h2 = c2 = np.zeros((hidden2,), dtype=np.float32)

        for x in window:
            print(
                f"x shape: {np.shape(x)}, h1 shape: {np.shape(h1)}, W_i shape: {np.shape(lstm1.W_i)}"
            )
            h1, c1 = lstm1.step(x, h1, c1)
            h2, c2 = lstm2.step(h1, h2, c2)

            x_in_lines.append(x)
            h_prev_lines.append(h2.copy())
            c_prev_lines.append(c2.copy())
            h_out_lines.append(h2)
            c_out_lines.append(c2)

    save_mem_file(x_in_lines, os.path.join(out_dir, "x_in.mem"))
    save_mem_file(h_prev_lines, os.path.join(out_dir, "h_prev.mem"))
    save_mem_file(c_prev_lines, os.path.join(out_dir, "c_prev.mem"))
    save_mem_file(h_out_lines, os.path.join(out_dir, "h_out.mem"))
    save_mem_file(c_out_lines, os.path.join(out_dir, "c_out.mem"))

    print(f"[info] Saved {len(h_out_lines)} vectors to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", default="new_data_sequences.npz", help="Path to input .npz file"
    )
    parser.add_argument(
        "--weights", default="saved_model/lstm_weights.npz", help="Path to weights file"
    )
    parser.add_argument(
        "--windows", type=int, default=1, help="Number of sequence windows to use"
    )
    parser.add_argument(
        "--format", choices=["q4.12"], default="q4.12", help="Fixed-point format"
    )
    parser.add_argument(
        "--outdir", default="golden_vectors", help="Output directory for .mem files"
    )
    args = parser.parse_args()

    generate_vectors(args.npz, args.weights, args.outdir, args.windows, args.format)


if __name__ == "__main__":
    main()
