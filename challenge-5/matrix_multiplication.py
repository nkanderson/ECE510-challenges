import unittest
import numpy as np
import logging
import time
import cProfile
import io
import sys
from pstats import SortKey

def matrix_multiply(matrix1, matrix2):
    """Multiplies two matrices."""
    if matrix1.shape[1] != matrix2.shape[0]:
        return None
    return np.dot(matrix1, matrix2)

class TestMatrix(unittest.TestCase):
    def test_multiply_matrices(self):
        """Tests matrix multiplication with a simple 2x2 case."""
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(matrix1, matrix2)
        expected_result = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_multiply_incompatible_matrices(self):
        """Tests the case where matrix dimensions are incompatible."""
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6, 7], [8, 9, 10]])
        result = matrix_multiply(matrix1, matrix2)
        self.assertIsNone(result)



def main():
    """Main function to run the matrix program and tests."""
    print("Running Matrix program and tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # Example
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
    result = matrix_multiply(matrix_a, matrix_b)
    print("Matrix Multiplication Result:")
    print(result)

if __name__ == "__main__":
    main()
