"""Unit tests for code matrix normalization."""

import unittest

import numpy as np

from codedcompute.numerics.linear_algebra import normalize_code

ROW_AXIS = "row"
COLUMN_AXIS = "column"


class TestNormalizeCode(unittest.TestCase):
    """Test row and column normalization across numeric dtypes."""

    def test_normalizes_integer_rows(self) -> None:
        """Integer matrices are promoted and each row has unit norm."""
        matrix = np.array([[3, 4, 0], [0, 0, 5]], dtype=np.int32)

        normalized = normalize_code(matrix, ROW_AXIS)

        self.assertEqual(normalized.dtype, np.float64)
        np.testing.assert_allclose(normalized, [[0.6, 0.8, 0], [0, 0, 1]])
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1)

    def test_normalizes_complex_columns(self) -> None:
        """Complex columns have unit norm after normalization."""
        matrix = np.array([[3 + 4j, 0], [0, -12j], [0, 5]], dtype=np.complex64)

        normalized = normalize_code(matrix, COLUMN_AXIS)

        self.assertEqual(normalized.dtype, np.complex128)
        np.testing.assert_allclose(
            normalized, [[0.6 + 0.8j, 0], [0, -12j / 13], [0, 5 / 13]]
        )
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=0), 1)

    def test_column_axis_normalizes_columns_of_a_non_square_matrix(self):
        """Column normalization differs from row normalization on the same input."""
        matrix = np.array([[3, 4, 0], [0, 0, 5]], dtype=np.float64)

        normalized = normalize_code(matrix, COLUMN_AXIS)

        np.testing.assert_array_equal(normalized, [[1, 1, 0], [0, 0, 1]])

    def test_normalization_and_result_mutation_leave_source_values_unchanged(self):
        """Double-precision inputs are neither modified nor shared with outputs."""
        for axis in (ROW_AXIS, COLUMN_AXIS):
            for dtype in (np.float64, np.complex128):
                with self.subTest(axis=axis, dtype=dtype):
                    matrix = np.array([[3, 4], [5, 12], [8, 15]], dtype=dtype)
                    original = matrix.copy()

                    normalized = normalize_code(matrix, axis)

                    np.testing.assert_array_equal(matrix, original)
                    normalized[:] = 0
                    np.testing.assert_array_equal(matrix, original)

    def test_rejects_zero_norm_vectors(self) -> None:
        """A zero vector cannot be normalized."""
        with self.assertRaises(ValueError):
            normalize_code(np.zeros((2, 2)), ROW_AXIS)


if __name__ == "__main__":
    unittest.main()
