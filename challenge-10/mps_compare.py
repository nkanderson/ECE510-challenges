import torch
import time

# Try both "cpu" and "mps"
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Dimensions: large enough to stress the backend
N = 8192

# Allocate two large matrices
a = torch.randn((N, N), device=device)
b = torch.randn((N, N), device=device)

# Warm-up (important for fair timing)
_ = a @ b

# Benchmark
start = time.time()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

print(f"Time for 10 matmul operations: {end - start:.3f} seconds")
