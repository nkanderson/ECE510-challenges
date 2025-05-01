import threading
import unittest
import random


class ProcessingElement(threading.Thread):
    def __init__(self, index, values, lock, barrier, exit_event):
        super().__init__()
        self.index = index
        self.values = values
        self.lock = lock
        self.barrier = barrier
        self.exit_event = exit_event
        self.active_phase = False

    def run(self):
        while not self.exit_event.is_set():
            if self.active_phase and self.index < len(self.values) - 1:
                with self.lock:
                    if self.values[self.index] > self.values[self.index + 1]:
                        self.values[self.index], self.values[self.index + 1] = (
                            self.values[self.index + 1],
                            self.values[self.index],
                        )
            self.barrier.wait()

    def set_phase(self, is_even_phase):
        self.active_phase = (
            (self.index % 2 == 0) if is_even_phase else (self.index % 2 == 1)
        )


def systolic_bubble_sort_threaded(values):
    n = len(values)
    if n <= 1:
        return values

    lock = threading.Lock()
    barrier = threading.Barrier(n - 1)
    exit_event = threading.Event()
    pes = []

    for i in range(n - 1):
        pe = ProcessingElement(i, values, lock, barrier, exit_event)
        pes.append(pe)
        pe.start()

    for phase in range(n):
        is_even = phase % 2 == 0
        for pe in pes:
            pe.set_phase(is_even)
        barrier.wait()  # Synchronize all PEs per phase

    exit_event.set()
    for pe in pes:
        pe.join()

    return values


class TestSystolicThreadedSort(unittest.TestCase):
    def test_sorted_output(self):
        original = [5, 1, 4, 2, 8]
        expected = sorted(original)
        result = systolic_bubble_sort_threaded(original.copy())
        self.assertEqual(result, expected)

    def test_empty_list(self):
        result = systolic_bubble_sort_threaded([])
        self.assertEqual(result, [])

    def test_single_element(self):
        result = systolic_bubble_sort_threaded([42])
        self.assertEqual(result, [42])


if __name__ == "__main__":
    unittest.main()
