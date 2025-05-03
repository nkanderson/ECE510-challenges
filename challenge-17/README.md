# Sorting on a Systolic Array

## Overview

Standard bubble sort has a Big O runtime of O(n^2). Using a 1-dimensional systolic array with parallel pairwise comparisons, we can expect a worst case runtime of O(n). As noted by ChatGPT, the number of comparisons is still O(n^2), but the reduction in execution time or cycles comes from the parallelization.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).

## Implementation with PyTorch

An implementation of a 1-D systolic array using PyTorch was generated using ChatGPT. In this implementation, each processing element (PE) of the systolic array is simulated by a position in a tensor, where the tensor holds all of the elements. The sort algorithm iterates over the tensor indexes and on each iteration, makes a pairwise comparison and swap over a vector of tensor values. It alternates between comparing odd first and even first items. That is, on one iteration, it will compare items beginning with odd indexes ((1, 2), (3, 4), etc.) and on the next, it compares those beginning on even indexes ((0, 1), (2, 3), etc.). An individual iteration is comparable to a systolic pulse.

The initial implementation provided by ChatGPT used `float32` values in the tensor. This may have been fine for benchmarking that code by itself, but it became a problem later when running comparisons between different implementations of the bubble sort. As such, the code was updated to use `int32` in order to match the alternative implementations.

## Results

```sh
(venv) ➜  challenge-17 git:(main) ✗ python main.py
Starting benchmark...

Running for size: 10
  MPS: 0.315651s, ~0.000 MB (est)
  Python: 0.000026s, 0.000 MB (actual)
  NumPy: 0.000321s, 0.001 MB (actual)

Running for size: 100
  MPS: 0.035995s, ~0.001 MB (est)
  Python: 0.000270s, 0.001 MB (actual)
  NumPy: 0.000931s, 0.003 MB (actual)

Running for size: 1000
  MPS: 0.164218s, ~0.011 MB (est)
  Python: 0.516384s, 0.008 MB (actual)
  NumPy: 0.012091s, 0.024 MB (actual)

Running for size: 2500
  MPS: 0.342374s, ~0.029 MB (est)
  Python: 3.236944s, 0.019 MB (actual)
  NumPy: 0.039871s, 0.058 MB (actual)

Running for size: 5000
  MPS: 0.794238s, ~0.057 MB (est)
  Python: 13.757728s, 0.038 MB (actual)
  NumPy: 0.104072s, 0.115 MB (actual)

Running for size: 7500
  MPS: 1.174017s, ~0.086 MB (est)
  Python: 32.928045s, 0.057 MB (actual)
  NumPy: 0.205220s, 0.173 MB (actual)

Running for size: 10000
  MPS: 1.574206s, ~0.114 MB (est)
  Python: 62.373392s, 0.077 MB (actual)
  NumPy: 0.503727s, 0.230 MB (actual)
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
