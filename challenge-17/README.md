# Sorting on a Systolic Array

## Overview

Standard bubble sort has a Big O runtime of O(n^2). Using a 1-dimensional systolic array with parallel pairwise comparisons, we can expect a worst case runtime of O(n). As noted by ChatGPT, the number of comparisons is still O(n^2), but the reduction in execution time or cycles comes from the parallelization.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).

## Implementation with PyTorch

An implementation of a 1-D systolic array using PyTorch was generated using ChatGPT. In this implementation, each processing element (PE) of the systolic array is simulated by a position in a tensor, where the tensor holds all of the elements. The sort algorithm iterates over the tensor indexes and on each iteration, makes a pairwise comparison and swap over a vector of tensor values. It alternates between comparing odd first and even first items. That is, on one iteration, it will compare items beginning with odd indexes ((1, 2), (3, 4), etc.) and on the next, it compares those beginning on even indexes ((0, 1), (2, 3), etc.). An individual iteration is comparable to a systolic pulse.

## Results

```sh
(venv) ➜  challenge-17 git:(main) ✗ python main.py
Starting benchmark...

Running for size: 10
  MPS: 0.055549s, ~0.000 MB (est)
  Python: 0.000020s, 0.000 MB (actual)

Running for size: 100
  MPS: 0.043363s, ~0.001 MB (est)
  Python: 0.000285s, 0.001 MB (actual)

Running for size: 1000
  MPS: 0.163990s, ~0.011 MB (est)
  Python: 0.395203s, 0.008 MB (actual)

Running for size: 2000
  MPS: 0.295364s, ~0.023 MB (est)
  Python: 1.990786s, 0.016 MB (actual)

Running for size: 5000
  MPS: 0.674899s, ~0.057 MB (est)
  Python: 14.606755s, 0.038 MB (actual)

Running for size: 10000
  MPS: 1.444358s, ~0.114 MB (est)
  Python: 59.808547s, 0.077 MB (actual)
```

ChatGPT interpretation of results:
Execution Time
- Python time increases super-linearly, consistent with O(n²) behavior.
- MPS time increases much more slowly — this supports your theory that the systolic-style implementation with Metal performs better (though it's still more than linear).
- MPS becomes significantly faster at 1000+ elements — it's ~40x faster than Python at 10,000 elements.

Memory Usage
- Both stay low overall, scaling linearly with input size, which is expected.
- MPS memory is a rough estimate (based on 12 × n float bytes).
- Python memory is very efficient due to native list usage — though tracemalloc may under-report memory used by internal structures (e.g., temporaries).

Future work
- It would be interesting to go back and fix the threaded Python version and benchmark that instead of the basic sequential Python bubble sort
