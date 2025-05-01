import torch
import time

# Benchmark parameters
MATRIX_SIZE = 8192
NUM_RUNS = 10


def benchmark_matmul(device):
    print(f"Running on device: {device}")
    a = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)
    b = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)

    # Warm-up
    _ = torch.matmul(a, b)

    # Timed runs
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(NUM_RUNS):
        c = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    elapsed = end - start
    print(f"{device} - Time for {NUM_RUNS} matmul operations: {elapsed:.3f} seconds\n")
    return elapsed


def main():
    devices = [("MPS", torch.device("mps")), ("CPU", torch.device("cpu"))]
    results = {}

    for label, dev in devices:
        if label == "MPS" and not torch.backends.mps.is_available():
            print("MPS not available, skipping.\n")
            continue
        elapsed = benchmark_matmul(dev)
        results[label] = elapsed

    print("\n=== Summary ===")
    for label, time_taken in results.items():
        print(f"{label}: {time_taken:.3f} seconds")
    if "CPU" in results and "MPS" in results:
        speedup = results["CPU"] / results["MPS"]
        print(f"Speedup (CPU / MPS): {speedup:.2f}x")


if __name__ == "__main__":
    main()
