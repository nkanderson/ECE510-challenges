import numpy as np
from utils import float_to_fixed_hex
from lstm_core import LSTMCell
import argparse


def load_mem_file(path, q_format=(4, 12)):
    q_int, q_frac = q_format
    scale = 2**q_frac
    total_bits = q_int + q_frac
    data = []
    with open(path, "r") as f:
        for line in f:
            val = int(line.strip(), 16)
            if val & (1 << (total_bits - 1)):  # negative value
                val -= 1 << total_bits
            data.append(val / scale)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="new_data_sequences.npz")
    parser.add_argument("--weights", default="saved_model/lstm_weights.npz")
    parser.add_argument("--golden", default="golden_vectors/h_out.mem")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    # Load data
    sequences = np.load(args.data)["sequences"]
    x_seq = sequences[args.window]
    input_dim = x_seq.shape[1]

    # Load weights
    weights = np.load(args.weights)
    W1, U1, b1 = weights["arr_0"], weights["arr_1"], weights["arr_2"]
    W2, U2, b2 = weights["arr_3"], weights["arr_4"], weights["arr_5"]
    hidden1 = len(b1) // 4
    hidden2 = len(b2) // 4

    # Init cells
    lstm1 = LSTMCell(input_dim, hidden1, W1, U1, b1)
    lstm2 = LSTMCell(hidden1, hidden2, W2, U2, b2)

    # Run inference
    h1 = np.zeros(hidden1, dtype=np.float32)
    c1 = np.zeros(hidden1, dtype=np.float32)
    h2 = np.zeros(hidden2, dtype=np.float32)
    c2 = np.zeros(hidden2, dtype=np.float32)

    reference = []
    for x in x_seq:
        h1, c1 = lstm1.step(x, h1, c1)
        h2, c2 = lstm2.step(h1, h2, c2)
        reference.extend(h2)

    # Load golden output
    golden = load_mem_file(args.golden)

    # Compare
    print("[info] Comparing output...")
    mismatches = 0
    for i, (r, g) in enumerate(zip(reference, golden)):
        if abs(r - g) > args.tol:
            print(
                f"Mismatch at timestep {i // hidden2}, neuron {i % hidden2}: expected {r:.6f}, got {g:.6f}"
            )
            mismatches += 1

    print(f"[done] Total mismatches: {mismatches} / {len(golden)}")


if __name__ == "__main__":
    main()
