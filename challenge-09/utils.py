import numpy as np


def float_to_fixed_hex(val, q_format=(4, 12)):
    q_int, q_frac = q_format
    scale = 2**q_frac
    max_val = 2 ** (q_int + q_frac - 1) - 1
    min_val = -(2 ** (q_int + q_frac - 1))
    fixed_val = int(round(float(val) * scale))  # ensure scalar
    fixed_val = max(min_val, min(max_val, fixed_val))  # clamp
    if fixed_val < 0:
        fixed_val = (1 << (q_int + q_frac)) + fixed_val  # 2's complement
    return f"{fixed_val:04X}"


def save_mem_file(array, path, q_format=(4, 12)):
    with open(path, "w") as f:
        array = np.array(array).flatten()
        for val in array:
            f.write(float_to_fixed_hex(val, q_format) + "\n")
