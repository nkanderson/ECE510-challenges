# generate_golden_vectors.py

import os
import argparse
import numpy as np
from lstm_core import LSTMCell


def float_to_fixed_hex(value, bits=16, frac=12):
    """
    Converts a float to fixed-point hex string in Qm.n format.
    Default is Q4.12 (m=4, n=12, total bits=16).
    Supports signed two's complement representation.
    """
    scale = 1 << frac
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    fixed_val = int(round(value * scale))

    if fixed_val > max_val:
        fixed_val = max_val
    elif fixed_val < min_val:
        fixed_val = min_val

    fixed_val &= (1 << bits) - 1  # mask to bits
    return f"{fixed_val:04X}"


def save_mem_file(filepath, matrix, bits=16, frac=12):
    with open(filepath, "w") as f:
        for row in matrix:
            hex_line = " ".join(
                float_to_fixed_hex(val, bits=bits, frac=frac) for val in row
            )
            f.write(f"{hex_line}\n")


def generate_vectors(npz_path, out_dir, num_windows, fmt):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path)["arr_0"]
    print(f"[info] Loaded sequences shape: {data.shape}")
    data = data[:num_windows]  # (num_windows, seq_len, input_dim)

    input_dim = data.shape[2]
    hidden_dim = 32  # adjust if needed

    lstm1 = LSTMCell(input_dim, hidden_dim)
    lstm2 = LSTMCell(hidden_dim, hidden_dim)

    # Optionally load weights here if needed
    # lstm1.load_weights(...)
    # lstm2.load_weights(...)

    x_in_lines, h_prev_lines, c_prev_lines = [], [], []
    h_out_lines, c_out_lines = [], []

    for window in data:
        h1 = c1 = np.zeros((hidden_dim,), dtype=np.float32)
        h2 = c2 = np.zeros((hidden_dim,), dtype=np.float32)

        for x in window:
            h1, c1 = lstm1.step(x, h1, c1)
            h2, c2 = lstm2.step(h1, h2, c2)

            x_in_lines.append(x)
            h_prev_lines.append(h2.copy())
            c_prev_lines.append(c2.copy())
            h_out_lines.append(h2)
            c_out_lines.append(c2)

    save_mem_file(os.path.join(out_dir, "x_in.mem"), x_in_lines)
    save_mem_file(os.path.join(out_dir, "h_prev.mem"), h_prev_lines)
    save_mem_file(os.path.join(out_dir, "c_prev.mem"), c_prev_lines)
    save_mem_file(os.path.join(out_dir, "h_out.mem"), h_out_lines)
    save_mem_file(os.path.join(out_dir, "c_out.mem"), c_out_lines)

    print(f"[info] Saved {len(h_out_lines)} vectors to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", default="saved_model/lstm_windows.npz", help="Path to input .npz file"
    )
    parser.add_argument(
        "--windows", type=int, default=5, help="Number of sequence windows to use"
    )
    parser.add_argument(
        "--format", choices=["q4.12"], default="q4.12", help="Fixed-point format"
    )
    parser.add_argument(
        "--outdir", default="golden_vectors", help="Output directory for .mem files"
    )
    args = parser.parse_args()

    generate_vectors(args.npz, args.outdir, args.windows, args.format)


if __name__ == "__main__":
    main()
