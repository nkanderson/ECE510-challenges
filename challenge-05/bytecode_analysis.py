import dis
import types
from collections import defaultdict

# Updated instruction categories for Python 3.12+
INSTRUCTION_CATEGORIES = {
    "Arithmetic": {"BINARY_OP", "UNARY_OP"},
    "Control Flow": {
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "JUMP_IF_TRUE_OR_POP",
        "JUMP_IF_FALSE_OR_POP",
        "RETURN_VALUE",
    },
    "Function Calls": {
        "CALL",
        "PRECALL",
        "CALL_FUNCTION",
        "CALL_METHOD",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_EX",
    },
    "Load/Store": {
        "LOAD_FAST",
        "STORE_FAST",
        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "LOAD_CONST",
        "LOAD_NAME",
        "STORE_NAME",
        "LOAD_ATTR",
        "STORE_ATTR",
        "LOAD_DEREF",
        "STORE_DEREF",
    },
    "Other": set(),  # Catch-all
}


def analyze_function_recursive(func):
    """
    Recursively analyze the bytecode instructions of a function and any nested functions.

    Returns:
        A dictionary mapping opcode names to counts.
    """
    op_counts = defaultdict(int)
    visited = set()

    def walk_code(code_obj):
        if code_obj in visited:
            return
        visited.add(code_obj)

        for instr in dis.get_instructions(code_obj):
            op_counts[instr.opname] += 1
            if instr.opname == "LOAD_CONST" and isinstance(
                instr.argval, types.CodeType
            ):
                walk_code(instr.argval)

    walk_code(func.__code__)
    return dict(op_counts)


def categorize_instructions(op_counts):
    """
    Groups instruction counts into defined categories.

    Args:
        op_counts: Dict[str, int] of individual opcode counts.

    Returns:
        Dict[str, int] of category counts.
    """
    category_counts = defaultdict(int)
    for opname, count in op_counts.items():
        placed = False
        for category, ops in INSTRUCTION_CATEGORIES.items():
            if opname in ops:
                category_counts[category] += count
                placed = True
                break
        if not placed:
            category_counts["Other"] += count
    return dict(category_counts)


# Optional benchmark/test functions to validate bytecode categorization
def example_arithmetic():
    a = 1
    b = 2
    c = a + b * 3 - 4
    return c


def example_control_flow(n=3):
    if n > 0:
        for i in range(n):
            if i % 2 == 0:
                print(i)
    else:
        return -1


def example_function_calls():
    def inner():
        return sum([1, 2, 3])

    return inner()


def example_load_store():
    x = 42
    y = x
    z = y + 1
    return z


# Benchmark registry
BENCHMARK_FUNCTIONS = {
    "example_arithmetic": example_arithmetic,
    "example_control_flow": example_control_flow,
    "example_function_calls": example_function_calls,
    "example_load_store": example_load_store,
}
