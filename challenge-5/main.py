import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import py_compile
import os

# Import the functions from the other filesfrom differential_equation_solver import solve_exponential_decay
from convolutional_network import create_cnn, train_and_evaluate_cnn
from matrix_multiplication import matrix_multiply
from differential_equation_solver import solve_exponential_decay

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
        # Expected solution: y(t) = y0 * exp(-kt)
        expected_solution = 5.0 * np.exp(-0.5 * t_eval)
        # Use assertAlmostEqual for floating-point comparisons
        for i in range(len(t_eval)):
            self.assertAlmostEqual(solution[i], expected_solution[i], places=6)

    def test_cnn_training(self):
        """
        Tests CNN training on a small, random dataset. Checks for reasonable accuracy.
        """
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
    # Compile the modules
    py_compile.compile("differential_equation_solver.py")
    py_compile.compile("convolutional_network.py")
    py_compile.compile("matrix_multiplication.py")

    # ODE Example
    initial_condition_1 = [5.0]
    t_span_1 = (0, 3)
    t_eval_1 = np.linspace(0, 3, 100)
    k_1 = 0.5
    solution_1 = solve_exponential_decay(initial_condition_1, t_span_1, t_eval_1, k_1)
    print(f"ODE Solution 1 (k=0.5): {solution_1[:5]}...")

    # CNN Example
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    model_1 = create_cnn((28, 28, 1), 10)
    loss_1, accuracy_1 = train_and_evaluate_cnn(model_1, x_train, y_train, x_test, y_test, epochs=5)
    print(f"CNN (Default) - Loss: {loss_1:.4f}, Accuracy: {accuracy_1:.4f}")

    # Matrix Multiplication Example
    matrix_a_1 = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b_1 = np.array([[7, 8], [9, 10], [11, 12]])
    result_1 = matrix_multiply(matrix_a_1, matrix_b_1)
    print("Matrix Multiplication Result 1:")
    print(result_1)

    unittest.main(argv=['first-arg-is-ignored'], exit=False)
