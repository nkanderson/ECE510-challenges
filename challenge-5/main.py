import sys
import os
import importlib
import dis
import cProfile
import io
import pstats
from pstats import SortKey
import unittest


def disassemble_to_file(func, filename):
    """Disassembles a function and writes the output to a file."""
    with open(filename, "w") as f:
        import dis

        old_stdout = sys.stdout
        sys.stdout = f
        try:
            dis.dis(func)
        finally:
            sys.stdout = old_stdout


def profile_function(func, *args, **kwargs):
    """Profiles the execution of a function and returns the stats."""
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    return s.getvalue()


def analyze_bytecode_optimized(func):
    """
    Analyzes the bytecode of a function and returns instruction counts,
    reflecting optimizations performed by the Python interpreter.
    """
    code = func.__code__
    instructions = []
    i = 0
    while i < len(code.co_code):
        op = code.co_code[i]
        opname = dis.opname[op]
        instructions.append(opname)
        i += 1 if op < dis.HAVE_ARGUMENT else 3
    counts = {}
    for instr in instructions:
        counts[instr] = counts.get(instr, 0) + 1
    return counts


def analyze_bytecode(dis_filename):
    """Analyzes the disassembled bytecode from a .dis file and returns instruction counts."""
    counts = {}
    try:
        with open(dis_filename, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 1:
                    # Check if the first part is a line number
                    if parts[0].strip().isdigit():
                        # Opcode is the second part
                        opcode = parts[1].strip()
                        if parts[1].strip().isdigit():
                            opcode = parts[2].strip()
                        # Handle opcodes with arguments
                        if "(" in opcode:
                            opcode = opcode.split("(")[0].strip()
                        counts[opcode] = counts.get(opcode, 0) + 1
    except FileNotFoundError:
        print(f"Error: .dis file not found: {dis_filename}")
        return {}
    return counts


def main():
    """Main function to parse arguments and run program/analysis."""
    if len(sys.argv) < 3:
        print("Usage: python main.py <program> <action>")
        print("  <program>: ode, cnn, matrix, qs, or crypt")
        print("  <action>: run, analyze")
        sys.exit(1)
    program = sys.argv[1].lower()
    action = sys.argv[2].lower()

    if program not in ["ode", "cnn", "matrix", "qs", "crypt"]:
        print("Invalid program argument.")
        print("Usage: python main.py <program> <action>")
        print("  <program>: ode, cnn, matrix, qs, or crypt")
        print("  <action>: run, analyze")
        sys.exit(1)

    if action not in ["run", "analyze"]:
        print("Invalid action argument.")
        print("Usage: python main.py <program> <action>")
        print("  <program>: ode, cnn, matrix, qs, or crypt")
        print("  <action>: run, analyze")
        sys.exit(1)

    # Use a match statement for module selection (Python 3.10+)
    match program:
        case "ode":
            module_name = "differential_equation_solver"
        case "cnn":
            module_name = "convolutional_network"
            print(f"Error: Module {module_name} not working yet!")
            sys.exit(1)
        case "matrix":
            module_name = "matrix_multiplication"
        case "qs":
            module_name = "quicksort"
        case "crypt":
            module_name = "cryptography"
        case _:
            # This should be caught above, but is included as a last resort.
            print("Invalid program argument.")
            sys.exit(1)

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(
            f"Error: Could not import module {module_name}.  Make sure the file {module_name}.py exists."
        )
        sys.exit(1)

    if action == "run":
        # Run tests *before* the main program logic
        test_suite = unittest.defaultTestLoader.discover(
            os.path.dirname(os.path.abspath(__file__)), pattern=f"{module_name}.py"
        )
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        if not result.wasSuccessful():
            sys.exit(1)  # Exit if tests fail

        module.main()  # Call the module's main function
    elif action == "analyze":
        disassemble_to_file(
            module.main, f"{module_name}.dis"
        )  # Disassemble the module's main
        print(f"Bytecode analysis of {program}:")
        print(
            analyze_bytecode(f"{module_name}.dis")
        )  # Use the main module's analyze_bytecode
        print(f"\nProfiling {program}:")
        print(profile_function(module.main))  # Use the main module's profile_function
    else:
        # This should be caught above, but is included as a last resort.
        print("Invalid action argument.")
        sys.exit(1)


if __name__ == "__main__":
    main()
