import os
import numpy as np


def float_to_fixed_hex(val, q_format=(4, 12)):
    q_int, q_frac = q_format
    scale = 2**q_frac
    max_val = 2 ** (q_int + q_frac - 1) - 1
    min_val = -(2 ** (q_int + q_frac - 1))
    fixed_val = int(round(val * scale))
    fixed_val = max(min_val, min(max_val, fixed_val))  # clamp
    if fixed_val < 0:
        fixed_val = (1 << (q_int + q_frac)) + fixed_val  # 2's complement
    return f"{fixed_val:04X}"


def save_mem_file(array, path, q_format=(4, 12)):
    with open(path, "w") as f:
        if isinstance(array[0], list):
            for vec in array:
                for val in vec:
                    f.write(float_to_fixed_hex(val, q_format) + "\n")
        else:
            for val in array:
                f.write(float_to_fixed_hex(val, q_format) + "\n")


def generate_mem_files(
    npz_path="saved_model/lstm_weights.npz", output_dir="weights", q_format=(4, 12)
):
    os.makedirs(output_dir, exist_ok=True)
    weights = np.load(npz_path, allow_pickle=True)
    for key in weights.files:
        array = weights[key].tolist()
        filename = f"{key}.mem"
        save_mem_file(array, os.path.join(output_dir, filename), q_format)
    print(f"Saved {len(weights.files)} .mem files to {output_dir}/")


if __name__ == "__main__":
    generate_mem_files()
