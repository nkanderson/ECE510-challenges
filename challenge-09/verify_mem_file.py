"""
File        : verify_mem_file.py
Author      : Niklas Anderson with ChatGPT assistance
Created     : Spring 2025
Description : Verify memory files against expected numpy arrays.
This script loads memory files containing weights and biases from an LSTM autoencoder,
flattens them to a fixed-point representation, and compares them against expected values.
"""

import os
import argparse
import numpy as np

GATES = ["i", "f", "c", "o"]


def load_mem_file(path):
    with open(path) as f:
        hex_vals = [line.strip() for line in f if line.strip()]
    return [int(val, 16) for val in hex_vals]


def flatten_row_major(arr):
    # Assume arr is 2D or 1D. Convert to flat list of Q4.12 integers
    if arr.ndim == 1:
        return [float_to_q412(x) for x in arr]
    elif arr.ndim == 2:
        return [float_to_q412(x) for row in arr for x in row]
    else:
        raise ValueError("Unexpected array shape")


def float_to_q412(x):
    val = int(round(x * (1 << 12)))
    val &= 0xFFFF  # 16-bit two's complement wrap
    return val


def verify(mem_path, expected, label):
    actual = load_mem_file(mem_path)
    expected_flat = flatten_row_major(expected)

    print(f"\nVerifying: {mem_path} [{label}]")
    if len(actual) != len(expected_flat):
        print(f"❌ Size mismatch: mem={len(actual)}, expected={len(expected_flat)}")
        return

    mismatches = [
        (i, a, e) for i, (a, e) in enumerate(zip(actual, expected_flat)) if a != e
    ]
    if not mismatches:
        print("✅ Match")
    else:
        print(f"❌ {len(mismatches)} mismatches")
        for i, a, e in mismatches[:10]:
            print(f"  idx {i}: mem = {a:#06x}, expected = {e:#06x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to weights.npz")
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory with .mem files"
    )
    args = parser.parse_args()

    weights = np.load(args.npz, allow_pickle=True)
    wlist = list(weights.values())

    # 4 gates each
    layers = [
        ("enc1", wlist[0:3]),
        ("enc2", wlist[3:6]),
        ("dec1", wlist[6:9]),
        ("dec2", wlist[9:12]),
    ]

    for layer_name, (W, U, b) in layers:
        hidden = b.shape[0] // 4
        for idx, gate in enumerate(GATES):
            W_gate = W[:, idx * hidden : (idx + 1) * hidden]
            U_gate = U[:, idx * hidden : (idx + 1) * hidden]
            b_gate = b[idx * hidden : (idx + 1) * hidden]

            verify(
                os.path.join(args.outdir, f"{layer_name}_W_{gate}.mem"),
                W_gate,
                f"{layer_name} W_{gate}",
            )
            verify(
                os.path.join(args.outdir, f"{layer_name}_U_{gate}.mem"),
                U_gate,
                f"{layer_name} U_{gate}",
            )
            verify(
                os.path.join(args.outdir, f"{layer_name}_b_{gate}.mem"),
                b_gate,
                f"{layer_name} b_{gate}",
            )

    # Dense layer
    Wd, bd = wlist[12:14]
    verify(os.path.join(args.outdir, "dense_W.mem"), Wd, "dense W")
    verify(os.path.join(args.outdir, "dense_b.mem"), bd, "dense b")


if __name__ == "__main__":
    main()
