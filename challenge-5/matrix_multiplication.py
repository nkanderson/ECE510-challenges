# matrix_multiplication.py
import numpy as np

def matrix_multiply(matrix1, matrix2):
    """
    Multiplies two matrices.  Includes a check for compatible dimensions.

    Args:
        matrix1: The first matrix (NumPy array).
        matrix2: The second matrix (NumPy array).

    Returns:
        The product of the two matrices, or None if dimensions are incompatible.
    """
    if matrix1.shape[1] != matrix2.shape[0]:
        return None  # Indicate incompatible dimensions
    return np.dot(matrix1, matrix2)
