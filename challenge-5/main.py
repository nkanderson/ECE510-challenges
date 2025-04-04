import unittest
import numpy as np
import py_compile
import os
import dis
import sys  # Import the sys module
import logging

# Import the functions from the other files
from differential_equation_solver import solve_exponential_decay
from matrix_multiplication import matrix_multiply

# Conditional import of tensorflow and keras
def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.layers import Input # Import Input layer
        from convolutional_network import create_cnn, train_and_evaluate_cnn # Import CNN functions
        return tf, keras, create_cnn, train_and_evaluate_cnn, Input  # Return Input
    except ImportError:
        return None, None, None, None, None

def disassemble_to_file(func, filename):
    """
    Disassembles a function and writes the output to a file.

    Args:
        func: The function to disassemble.
        filename: The name of the file to write to.
    """
    with open(filename, "w") as f:
        # Redirect standard output to the file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            dis.dis(func)
        finally:
            # Restore standard output
            sys.stdout = old_stdout

class TestAll(unittest.TestCase):
    def test_solve_exponential_decay(self):
        """
        Tests the solve_exponential_decay function.  Checks a known solution.
        """
        initial_condition = [5.0]  # Initial value of y
        t_span = (0, 3)         # Time span from t=0 to t=3
        t_eval = np.linspace(0, 3, 4)  # Evaluate at 4 points
        k = 0.5                   # Decay constant
        solution = solve_exponential_decay(initial_condition, t_span, t_eval, k)
        # Expected solution: y(t) = y0 * exp(-0.5 * t_eval)
        expected_solution = 5.0 * np.exp(-0.5 * t_eval)
        # Use assertAlmostEqual for floating-point comparisons
        for i in range(len(t_eval)):
            self.assertAlmostEqual(solution[i], expected_solution[i], places=6)

    def test_cnn_training(self):
        """
        Tests CNN training on a small, random dataset. Checks for reasonable accuracy.
        """
        tf, keras, create_cnn, train_and_evaluate_cnn, Input = import_tensorflow()
        if not tf:
            self.skipTest("TensorFlow is not installed")
            return

        input_shape = (28, 28, 1)
        num_classes = 10
        model = create_cnn(input_shape, num_classes)
        x_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 10, 100).astype(np.int32)
        x_test = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 10, 20).astype(np.int32)
        loss, accuracy = train_and_evaluate_cnn(model, x_train, y_train, x_test, y_test, epochs=3)
        self.assertGreaterEqual(accuracy, 0.1) # Check for minimal accuracy
        self.assertGreaterEqual(loss, 0)

    def test_multiply_matrices(self):
        """
        Tests matrix multiplication with a simple 2x2 case.
        """
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(matrix1, matrix2)
        expected_result = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_multiply_incompatible_matrices(self):
        """
        Tests the case where matrix dimensions are incompatible.
        """
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6, 7], [8, 9, 10]])  # 2x3 matrix
        result = matrix_multiply(matrix1, matrix2)
        self.assertIsNone(result) #Check the function returns None

if __name__ == "__main__":
    # Get the program argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <program>")
        print("  <program>: ode, cnn, or matrix")
        sys.exit(1)
    program = sys.argv[1].lower()

    # Run the specified program
    if program == "ode":
        print("Running ODE program...")
        py_compile.compile("differential_equation_solver.py")
        disassemble_to_file(solve_exponential_decay, "differential_equation_solver.dis")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)  # Run tests
        # ODE Example
        initial_condition_1 = [5.0]
        t_span_1 = (0, 3)
        t_eval_1 = np.linspace(0, 3, 100)
        k_1 = 0.5
        solution_1 = solve_exponential_decay(initial_condition_1, t_span_1, t_eval_1, k_1)
        print(f"ODE Solution 1 (k=0.5): {solution_1[:5]}...")

    elif program == "cnn":
        print("Running CNN program...")
        # Import tensorflow and suppress warnings only when cnn is selected
        tf, keras, create_cnn, train_and_evaluate_cnn, Input = import_tensorflow()
        if tf:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error
            logging.getLogger('tensorflow').setLevel(logging.ERROR)

        py_compile.compile("convolutional_network.py")
        disassemble_to_file(create_cnn, "convolutional_network.dis")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)  # Run tests
        # CNN Example
        if not tf:
            print("TensorFlow is not installed. Skipping CNN example.")
        else:
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
            model_1 = create_cnn((28, 28, 1), 10)
            loss_1, accuracy_1 = train_and_evaluate_cnn(model_1, x_train, y_train, x_test, y_test, epochs=5)
            print(f"CNN (Default) - Loss: {loss_1:.4f}, Accuracy: {accuracy_1:.4f}")

    elif program == "matrix":
        print("Running Matrix program...")
        py_compile.compile("matrix_multiplication.py")
        disassemble_to_file(matrix_multiply, "matrix_multiplication.dis")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)  # Run tests
        # Matrix Multiplication Example
        matrix_a_1 = np.array([[1, 2, 3], [4, 5, 6]])
        matrix_b_1 = np.array([[7, 8], [9, 10], [11, 12]])
        result_1 = matrix_multiply(matrix_a_1, matrix_b_1)
        print("Matrix Multiplication Result 1:")
        print(result_1)
    else:
        print("Invalid program argument.")
        print("Usage: python main.py <program>")
        print("  <program>: ode, cnn, or matrix")
        sys.exit(1)
