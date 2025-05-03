import numpy as np
import time
import tracemalloc
import unittest


def systolic_bubble_sort_numpy(arr):
    x = arr.copy()
    n = len(x)

    for t in range(n):
        if t % 2 == 0:
            idx1 = np.arange(0, n - 1, 2)
        else:
            idx1 = np.arange(1, n - 1, 2)
        idx2 = idx1 + 1

        vals1 = x[idx1]
        vals2 = x[idx2]

        min_vals = np.minimum(vals1, vals2)
        max_vals = np.maximum(vals1, vals2)

        x[idx1] = min_vals
        x[idx2] = max_vals

    return x


def benchmark_bubble_sort_numpy(size):
    arr = np.random.randint(0, 10000, size).astype(np.int32)
    tracemalloc.start()
    start = time.perf_counter()
    systolic_bubble_sort_numpy(arr)
    end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end - start
    peak_memory_mb = peak / 1024 / 1024
    return elapsed_time, peak_memory_mb


class TestSystolicBubbleSortNumpy(unittest.TestCase):
    def test_sorted_output(self):
        arr = np.array([5, 3, 2, 4, 1])
        sorted_arr = systolic_bubble_sort_numpy(arr)
        self.assertTrue(np.all(sorted_arr == np.sort(arr)))

    def test_empty_array(self):
        arr = np.array([])
        sorted_arr = systolic_bubble_sort_numpy(arr)
        self.assertEqual(sorted_arr.size, 0)

    def test_single_element(self):
        arr = np.array([42])
        sorted_arr = systolic_bubble_sort_numpy(arr)
        self.assertEqual(sorted_arr[0], 42)


if __name__ == "__main__":
    unittest.main()
