import torch
import time
import matplotlib.pyplot as plt
from systolic_bubble_sort import systolic_bubble_sort
from python_bubble_sort import benchmark_bubble_sort_python
from numpy_bubble_sort import benchmark_bubble_sort_numpy


def estimate_total_memory(tensor: torch.Tensor) -> float:
    """Estimate total working memory used during sorting in MB."""
    n = tensor.nelement()
    total_bytes = 12 * n  # clone + intermediate min/max + original input
    return total_bytes / 1024 / 1024


def run_benchmark():
    sizes = [10, 100, 1000, 2500, 5000, 7500, 10000]
    mps_times, mps_memories = [], []
    python_times, python_memories = [], []
    numpy_times, numpy_memories = [], []

    print("Starting benchmark...\n")

    for size in sizes:
        print(f"Running for size: {size}")

        # --- MPS ---
        if torch.backends.mps.is_available():
            device = "mps"
            input_tensor = torch.randint(0, 10000, (size,), dtype=torch.int32)
            start = time.perf_counter()
            _ = systolic_bubble_sort(input_tensor, device=device)
            end = time.perf_counter()
            mps_times.append(end - start)
            mps_memories.append(estimate_total_memory(input_tensor))
        else:
            print("MPS not available. Skipping MPS benchmark.")
            mps_times.append(0)
            mps_memories.append(0)

        # --- Pure Python Bubble Sort ---
        py_time, py_mem = benchmark_bubble_sort_python(size)
        python_times.append(py_time)
        python_memories.append(py_mem)

        # --- NumPy Bubble Sort ---
        np_time, np_mem = benchmark_bubble_sort_numpy(size)
        numpy_times.append(np_time)
        numpy_memories.append(np_mem)

        print(f"  MPS: {mps_times[-1]:.6f}s, ~{mps_memories[-1]:.3f} MB (est)")
        print(
            f"  Python: {python_times[-1]:.6f}s, {python_memories[-1]:.3f} MB (actual)"
        )
        print(f"  NumPy: {np_time:.6f}s, {np_mem:.3f} MB (actual)\n")

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax2 = ax1.twinx()

    x = range(len(sizes))
    width = 0.12
    offset = [
        -1.5 * width,
        -0.5 * width,
        0.5 * width,
        1.5 * width,
        2.5 * width,
        3.5 * width,
    ]

    bars1 = ax1.bar(
        [i + offset[0] for i in x],
        mps_times,
        width=width,
        label="MPS Time (s)",
        color="darkgreen",
    )
    bars2 = ax1.bar(
        [i + offset[1] for i in x],
        python_times,
        width=width,
        label="Python Time (s)",
        color="darkblue",
    )
    bars3 = ax1.bar(
        [i + offset[2] for i in x],
        numpy_times,
        width=width,
        label="NumPy Time (s)",
        color="orange",
    )
    bars4 = ax2.bar(
        [i + offset[3] for i in x],
        mps_memories,
        width=width,
        label="MPS Memory (MB)",
        color="lightgreen",
    )
    bars5 = ax2.bar(
        [i + offset[4] for i in x],
        python_memories,
        width=width,
        label="Python Memory (MB)",
        color="lightblue",
    )
    bars6 = ax2.bar(
        [i + offset[5] for i in x],
        numpy_memories,
        width=width,
        label="NumPy Memory (MB)",
        color="moccasin",
    )

    ax1.set_xlabel("Array Input Size")
    ax1.set_ylabel("Execution Time (s)")
    ax2.set_ylabel("Memory Usage (MB)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(sizes)
    ax1.set_title(
        "Systolic Bubble Sort: Metal Performance Shaders (MPS) vs Python vs NumPy Execution"
    )

    lines = [bars1, bars2, bars3, bars4, bars5, bars6]
    labels = [bar.get_label() for bar in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_benchmark()
