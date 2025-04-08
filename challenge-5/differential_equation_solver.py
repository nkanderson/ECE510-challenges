import unittest
import numpy as np
import cProfile
import pstats
import io


def solve_exponential_decay(initial_condition, t_span, t_eval, k):
    """Solves the exponential decay equation."""
    y0 = initial_condition[0]
    t0, tf = t_span
    solution = y0 * np.exp(-k * t_eval)
    return solution


class TestODE(unittest.TestCase):
    def test_solve_exponential_decay(self):
        """Tests the solve_exponential_decay function."""
        initial_condition = [5.0]
        t_span = (0, 3)
        t_eval = np.linspace(0, 3, 4)
        k = 0.5
        solution = solve_exponential_decay(initial_condition, t_span, t_eval, k)
        expected_solution = 5.0 * np.exp(-0.5 * t_eval)
        for i in range(len(t_eval)):
            self.assertAlmostEqual(solution[i], expected_solution[i], places=6)


def main():
    """Main function to run the ODE program."""
    print("ODE program execution...")
    # Example
    initial_condition = [5.0]
    t_span = (0, 3)
    t_eval = np.linspace(0, 3, 100)
    k = 0.5
    solution = solve_exponential_decay(initial_condition, t_span, t_eval, k)
    print(f"ODE Solution: {solution[:5]}...")

    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()
    solve_exponential_decay(initial_condition, t_span, t_eval, k)
    profiler.disable()

    # Save the profiling results to a file
    profiler.dump_stats("differential_equation_solver.prof")

    # Print the statistics to the console
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats()


if __name__ == "__main__":
    unittest.main()
