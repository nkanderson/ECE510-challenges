import torch
import time
import matplotlib.pyplot as plt
import psutil
import os
import gc

NUM_RUNS = 10
MATRIX_SIZES = [512, 1024, 2048, 4096, 8192]


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in MB


def benchmark_matmul(device, size):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    # Warm-up
    _ = torch.matmul(a, b)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    peak_mem = get_memory_usage()

    for _ in range(NUM_RUNS):
        c = torch.matmul(a, b)
        current_mem = get_memory_usage()
        if current_mem > peak_mem:
            peak_mem = current_mem

    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    elapsed = end - start
    return elapsed, peak_mem


def main():
    devices = [("MPS", torch.device("mps")), ("CPU", torch.device("cpu"))]
    all_results = {}

    for label, dev in devices:
        if label == "MPS" and not torch.backends.mps.is_available():
            print("MPS not available, skipping.\n")
            continue

        runtimes = []
        mem_usages = []
        print(f"\n--- Benchmarking on {label} ---")
        for size in MATRIX_SIZES:
            elapsed, mem_used = benchmark_matmul(dev, size)
            print(f"Size {size}x{size}: {elapsed:.3f}s, Peak Memory: {mem_used:.2f} MB")
            runtimes.append(elapsed)
            mem_usages.append(mem_used)

        all_results[label] = {
            "sizes": MATRIX_SIZES,
            "times": runtimes,
            "mem": mem_usages,
        }

    # Plot runtimes
    for label in all_results:
        plt.plot(
            all_results[label]["sizes"], all_results[label]["times"], label=f"{label}"
        )
    plt.title("Matrix Multiplication Runtime")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot memory usage
    for label in all_results:
        plt.plot(
            all_results[label]["sizes"], all_results[label]["mem"], label=f"{label}"
        )
    plt.title("Peak Memory Usage During Matmul")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Peak Memory (MB)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
