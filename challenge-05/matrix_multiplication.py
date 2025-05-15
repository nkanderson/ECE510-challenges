import unittest
import numpy as np


def matrix_multiply(matrix1, matrix2):
    """
    Multiplies two matrices. Includes a check for compatible dimensions.

    Args:
        matrix1: The first matrix (NumPy array).
        matrix2: The second matrix (NumPy array).

    Returns:
        The product of the two matrices, or None if dimensions are incompatible.
    """
    if matrix1.shape[1] != matrix2.shape[0]:
        return None  # Indicate incompatible dimensions
    return np.dot(matrix1, matrix2)


class TestMatrixMultiplication(unittest.TestCase):
    def test_multiply_matrices(self):
        """Tests matrix multiplication with a simple 2x2 case."""
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(matrix1, matrix2)
        expected_result = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_multiply_incompatible_matrices(self):
        """Tests the case where matrix dimensions are incompatible."""
        matrix1 = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
        matrix2 = np.array([[5, 6, 7]])  # shape: (1, 3)
        result = matrix_multiply(matrix1, matrix2)
        self.assertIsNone(result)


def main():
    """Main function to run matrix multiplication examples."""
    print("Running Matrix program execution...")

    # Example 1: 2x3 times 3x2
    matrix_a_1 = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b_1 = np.array([[7, 8], [9, 10], [11, 12]])
    result_1 = matrix_multiply(matrix_a_1, matrix_b_1)
    print("Matrix Multiplication Result 1:")
    print(result_1)

    # Example 2: 3x1 times 1x3
    matrix_a_2 = np.array([[1], [2], [3]])
    matrix_b_2 = np.array([[4, 5, 6]])
    result_2 = matrix_multiply(matrix_a_2, matrix_b_2)
    print("Matrix Multiplication Result 2:")
    print(result_2)

    # Example 3: Incompatible dimensions (2x3 and 2x3)
    matrix_a_3 = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b_3 = np.array([[7, 8, 9], [10, 11, 12]])
    result_3 = matrix_multiply(matrix_a_3, matrix_b_3)
    print("Matrix Multiplication Result 3 (Incompatible):")
    print(result_3)  # Will print None


if __name__ == "__main__":
    unittest.main()
