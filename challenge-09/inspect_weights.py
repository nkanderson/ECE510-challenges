"""Basic script to inspect the weights of the saved model"""

import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_weights.py path/to/weights.npz")
    sys.exit(1)

weights_path = sys.argv[1]
weights = np.load(weights_path)

print(f"\nContents of: {weights_path}\n")
for name in weights.files:
    arr = weights[name]
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size == 0:
        print("  ⚠️  WARNING: empty array!")
