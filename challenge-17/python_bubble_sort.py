import threading
import unittest
import random


class ProcessingElement(threading.Thread):
    def __init__(self, index, values, lock, barrier):
        super().__init__()
        self.index = index
        self.values = values
        self.lock = lock
        self.barrier = barrier
        self.active_phase = False
        self.running = True

    def run(self):
        while self.running:
            if self.active_phase and self.index < len(self.values) - 1:
                with self.lock:
                    if self.values[self.index] > self.values[self.index + 1]:
                        self.values[self.index], self.values[self.index + 1] = (
                            self.values[self.index + 1],
                            self.values[self.index],
                        )
            self.barrier.wait()

    def stop(self):
        self.running = False

    def set_phase(self, is_even_phase):
        self.active_phase = (
            (self.index % 2 == 0) if is_even_phase else (self.index % 2 == 1)
        )


def systolic_bubble_sort_threaded(values):
    n = len(values)
    lock = threading.Lock()
    barrier = threading.Barrier(n)  # Synchronize all threads per phase
    pes = []

    for i in range(n):
        pe = ProcessingElement(i, values, lock, barrier)
        pes.append(pe)
        pe.start()

    for phase in range(n):
        is_even = phase % 2 == 0
        for pe in pes:
            pe.set_phase(is_even)
        barrier.wait()  # Wait for all PEs to complete the phase

    for pe in pes:
        pe.stop()
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
