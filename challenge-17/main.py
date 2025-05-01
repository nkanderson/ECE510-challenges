import torch
import time
import matplotlib.pyplot as plt
import tracemalloc
from systolic_bubble_sort import systolic_bubble_sort


def estimate_total_memory(tensor: torch.Tensor) -> float:
    """Estimate total working memory used during sorting in MB."""
    n = tensor.nelement()
    total_bytes = 12 * n  # clone + intermediate min/max + original input
    return total_bytes / 1024 / 1024


def run_benchmark():
    sizes = [10, 100, 1000, 10000]
    mps_times = []
    mps_memories = []
    cpu_times = []
    cpu_memories = []

    print("Starting benchmark...\n")

    for size in sizes:
        print(f"Running for size: {size}")
        # --- MPS ---
        if torch.backends.mps.is_available():
            device = "mps"
            input_tensor = torch.randint(0, 10000, (size,), dtype=torch.float32)

            start = time.perf_counter()
            _ = systolic_bubble_sort(input_tensor, device=device)
            end = time.perf_counter()

            mps_times.append(end - start)
            mps_memories.append(estimate_total_memory(input_tensor))
        else:
            print("MPS not available. Skipping MPS benchmark.")
            mps_times.append(0)
            mps_memories.append(0)

        # --- CPU ---
        device = "cpu"
        input_tensor = torch.randint(0, 10000, (size,), dtype=torch.float32)

        tracemalloc.start()
        start = time.perf_counter()
        _ = systolic_bubble_sort(input_tensor, device=device)
        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_times.append(end - start)
        cpu_memories.append(peak / 1024 / 1024)  # MB

        print(f"  MPS: {mps_times[-1]:.6f}s, ~{mps_memories[-1]:.3f} MB (est)")
        print(f"  CPU: {cpu_times[-1]:.6f}s, {cpu_memories[-1]:.3f} MB (actual)\n")

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = range(len(sizes))
    width = 0.2

    # Offset bars
    offset = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    # Bars
    bars1 = ax1.bar(
        [i + offset[0] for i in x],
        mps_times,
        width=width,
        label="MPS Time (s)",
        color="darkgreen",
    )
    bars2 = ax2.bar(
        [i + offset[1] for i in x],
        mps_memories,
        width=width,
        label="MPS Memory (MB)",
        color="lightgreen",
    )
    bars3 = ax1.bar(
        [i + offset[2] for i in x],
        cpu_times,
        width=width,
        label="CPU Time (s)",
        color="darkblue",
    )
    bars4 = ax2.bar(
        [i + offset[3] for i in x],
        cpu_memories,
        width=width,
        label="CPU Memory (MB)",
        color="lightblue",
    )

    ax1.set_xlabel("Array Size")
    ax1.set_ylabel("Execution Time (s)")
    ax2.set_ylabel("Memory Usage (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.set_title("Systolic Bubble Sort: MPS vs CPU Execution")

    # Legend
    lines = [bars1, bars2, bars3, bars4]
    labels = [bar.get_label() for bar in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_benchmark()
