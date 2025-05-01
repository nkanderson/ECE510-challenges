import unittest
import random
import time
import tracemalloc


def bubble_sort_python(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def benchmark_bubble_sort_python(size):
    arr = [random.randint(0, 10000) for _ in range(size)]
    tracemalloc.start()
    start = time.perf_counter()
    bubble_sort_python(arr.copy())
    end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end - start
    peak_memory_mb = peak / 1024 / 1024
    return elapsed_time, peak_memory_mb


class TestBubbleSortPython(unittest.TestCase):
    def test_sorted_output(self):
        original = [5, 1, 4, 2, 8]
        expected = sorted(original)
        result = bubble_sort_python(original.copy())
        self.assertEqual(result, expected)

    def test_empty_list(self):
        result = bubble_sort_python([])
        self.assertEqual(result, [])

    def test_single_element(self):
        result = bubble_sort_python([42])
        self.assertEqual(result, [42])

    def test_already_sorted(self):
        original = [1, 2, 3, 4, 5]
        result = bubble_sort_python(original.copy())
        self.assertEqual(result, original)

    def test_reverse_sorted(self):
        original = [5, 4, 3, 2, 1]
        expected = [1, 2, 3, 4, 5]
        result = bubble_sort_python(original.copy())
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
