import threading
import unittest
import random


class ProcessingElement(threading.Thread):
    def __init__(self, index, values, lock, phase_event, next_phase_event):
        super().__init__()
        self.index = index
        self.values = values
        self.lock = lock
        self.phase_event = phase_event
        self.next_phase_event = next_phase_event
        self.running = True

    def run(self):
        while self.running:
            self.phase_event.wait()
            self.phase_event.clear()
            if self.index < len(self.values) - 1 and self.active_phase:
                with self.lock:
                    if self.values[self.index] > self.values[self.index + 1]:
                        self.values[self.index], self.values[self.index + 1] = (
                            self.values[self.index + 1],
                            self.values[self.index],
                        )
            self.next_phase_event.set()

    def stop(self):
        self.running = False

    def set_phase(self, is_even_phase):
        self.active_phase = (
            (self.index % 2 == 0) if is_even_phase else (self.index % 2 == 1)
        )


def systolic_bubble_sort_threaded(values):
    n = len(values)
    lock = threading.Lock()
    phase_event = threading.Event()
    next_phase_event = threading.Event()
    pes = []

    for i in range(n - 1):
        pe = ProcessingElement(i, values, lock, phase_event, next_phase_event)
        pes.append(pe)
        pe.start()

    for phase in range(n):
        is_even = phase % 2 == 0
        for pe in pes:
            pe.set_phase(is_even)

        for _ in range(n - 1):
            next_phase_event.clear()
            phase_event.set()
            next_phase_event.wait()

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
