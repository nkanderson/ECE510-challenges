import unittest
import random
import cProfile
import pstats
import io


def quicksort(arr):
    """
    Sorts an array using the quicksort algorithm.

    Args:
        arr: The array to be sorted.

    Returns:
        The sorted array.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


class TestQuicksort(unittest.TestCase):
    def test_quicksort_empty_array(self):
        """Tests quicksort with an empty array."""
        arr = []
        result = quicksort(arr)
        self.assertEqual(result, [])

    def test_quicksort_sorted_array(self):
        """Tests quicksort with an already sorted array."""
        arr = [1, 2, 3, 4, 5]
        result = quicksort(arr)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_quicksort_reversed_array(self):
        """Tests quicksort with a reversed array."""
        arr = [5, 4, 3, 2, 1]
        result = quicksort(arr)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_quicksort_random_array(self):
        """Tests quicksort with a randomly ordered array."""
        arr = [random.randint(1, 100) for _ in range(10)]
        result = quicksort(arr)
        self.assertEqual(result, sorted(arr))


def main():
    """Main function to run the quicksort program and tests."""
    print("Running Quicksort program execution...")
    # Example
    arr = [5, 2, 9, 1, 5, 6]
    sorted_arr = quicksort(arr)
    print("Sorted Array:", sorted_arr)

    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()
    quicksort(arr)  # Or any other function you want to profile
    profiler.disable()

    # Save the profiling results to a file
    profiler.dump_stats("quicksort.prof")

    # Print the statistics to the console
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats()


if __name__ == "__main__":
    unittest.main()
