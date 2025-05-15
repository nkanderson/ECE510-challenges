import matplotlib.pyplot as plt


def plot_instruction_categories(category_counts, program_name):
    """
    Plots a bar chart of instruction categories for a single program.

    Args:
        category_counts: dict[str, int] - mapping of category names to counts
        program_name: str - name of the program analyzed
    """
    categories = list(category_counts.keys())
    counts = [category_counts[cat] for cat in categories]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.title(f"Bytecode Instruction Categories for '{program_name}'")
    plt.xlabel("Instruction Category")
    plt.ylabel("Instruction Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_benchmark_test_by_name(benchmark_name, analyzer, categorizer):
    """
    Plots categorized instruction counts for a given benchmark by name.

    Args:
        benchmark_name: str - the name of the test function to analyze
        analyzer: callable - function to analyze bytecode (e.g., analyze_function_recursive)
        categorizer: callable - function to group instructions into categories
    """
    from bytecode_analysis import BENCHMARK_FUNCTIONS

    if benchmark_name not in BENCHMARK_FUNCTIONS:
        print(f"Benchmark '{benchmark_name}' not found.")
        return

    func = BENCHMARK_FUNCTIONS[benchmark_name]
    op_counts = analyzer(func)
    categorized = categorizer(op_counts)
    plot_instruction_categories(categorized, benchmark_name)
