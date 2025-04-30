import subprocess
import time
import sys


def run_with_timing(script_path):
    print(f"\n--- Running {script_path} ---")
    start = time.time()
    subprocess.run([sys.executable, script_path], check=True)
    end = time.time()
    duration = end - start
    print(f"Execution time for {script_path}: {duration:.3f} seconds")
    return duration


def run_with_line_profiler(script_path):
    print(f"\n--- Profiling {script_path} with line_profiler ---")
    subprocess.run(["kernprof", "-l", "-v", script_path], check=True)


def run_with_memory_profiler(script_path):
    print(f"\n--- Profiling {script_path} with memory_profiler ---")
    subprocess.run(["python", "-m", "memory_profiler", script_path], check=True)


if __name__ == "__main__":
    original = "q_learning.py"
    optimized = "pytorch_q_learning.py"

    # Run both versions with timing
    time_original = run_with_timing(original)
    time_optimized = run_with_timing(optimized)

    # Optionally: Profile line-by-line (requires @profile decorators added to both scripts)
    # run_with_line_profiler(original)
    # run_with_line_profiler(optimized)

    # Optionally: Profile memory usage
    # run_with_memory_profiler(original)
    # run_with_memory_profiler(optimized)

    print("\n=== Comparison Summary ===")
    print(f"Original Runtime: {time_original:.3f}s")
    print(f"Optimized Runtime: {time_optimized:.3f}s")
    speedup = time_original / time_optimized if time_optimized else float("inf")
    print(f"Speedup: {speedup:.2f}x")
