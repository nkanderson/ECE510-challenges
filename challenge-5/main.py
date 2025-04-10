import sys
import os
import importlib
import dis
import cProfile
import io
import pstats
import unittest
import argparse
import matplotlib.pyplot as plt
from pstats import SortKey

from bytecode_analysis import (
    analyze_function_recursive,
    categorize_instructions,
    BENCHMARK_FUNCTIONS,
)
from visualization import plot_instruction_categories, plot_benchmark_test_by_name

PROGRAM_MODULES = {
    "ode": "differential_equation_solver",
    "cnn": "convolutional_network",
    "matrix": "matrix_multiplication",
    "qs": "quicksort",
    "crypt": "cryptography",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or analyze individual programs.")
    parser.add_argument(
        "program", help="Program name (ode, cnn, matrix, qs, crypt, or dummy)"
    )
    parser.add_argument(
        "action",
        choices=["run", "analyze", "analyze-bytecode", "test-bytecode"],
        help="Action to perform",
    )
    parser.add_argument(
        "benchmark", nargs="?", help="Benchmark name for test-bytecode action"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize profiling with snakeviz"
    )
    return parser.parse_args()


def disassemble_to_file(func, filename):
    with open(filename, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            dis.dis(func)
        finally:
            sys.stdout = old_stdout


def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()
    pr.dump_stats(f"{func.__name__}.prof")
    return s.getvalue()


def analyze_bytecode(dis_filename):
    counts = {}
    try:
        with open(dis_filename, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 1 and parts[0].isdigit():
                    opcode = parts[1].strip()
                    if opcode.isdigit():
                        opcode = parts[2].strip()
                    if "(" in opcode:
                        opcode = opcode.split("(")[0].strip()
                    counts[opcode] = counts.get(opcode, 0) + 1
    except FileNotFoundError:
        print(f"Error: .dis file not found: {dis_filename}")
    return counts


def run_tests(module_name):
    print(f"Running tests for {module_name}...")
    test_suite = unittest.defaultTestLoader.discover(
        os.path.dirname(os.path.abspath(__file__)), pattern=f"{module_name}.py"
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result.wasSuccessful()


def main():
    args = parse_args()
    program = args.program.lower()
    action = args.action.lower()

    if action == "test-bytecode":
        if not args.benchmark:
            print("Please specify a benchmark name for test-bytecode.")
            sys.exit(1)
        plot_benchmark_test_by_name(
            args.benchmark, analyze_function_recursive, categorize_instructions
        )
        return

    module_name = PROGRAM_MODULES.get(program)
    if not module_name:
        print(f"Unknown program: {program}")
        sys.exit(1)

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import module '{module_name}'.")
        sys.exit(1)

    if action == "run":
        if not run_tests(module_name):
            sys.exit(1)
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"Module '{module_name}' has no main() function to execute.")

    elif action == "analyze":
        if hasattr(module, "main"):
            disassemble_to_file(module.main, f"{module_name}.dis")
            print(f"Bytecode analysis of {program}:")
            print(analyze_bytecode(f"{module_name}.dis"))
            print(f"\nProfiling {program}:")
            profile_output = profile_function(module.main)
            print(profile_output)
            if args.visualize:
                import subprocess

                print("\nLaunching snakeviz...")
                subprocess.run(["snakeviz", f"{module.main.__name__}.prof"])
        else:
            print(f"Module '{module_name}' has no main() function to analyze.")

    elif action == "analyze-bytecode":
        if hasattr(module, "main"):
            print(f"Analyzing bytecode for '{program}'...")
            op_counts = analyze_function_recursive(module.main)
            categorized = categorize_instructions(op_counts)
            # TODO: Add opcode breakdowns per category
            plot_instruction_categories(categorized, program)
        else:
            print(f"Module '{module_name}' has no main() function to analyze.")


if __name__ == "__main__":
    main()
