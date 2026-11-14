"""Code matrix normalization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from codedcompute.numerics.matrix_shapes import MatrixAxis
from codedcompute.utils.numpy_types import as_inexact_array


def normalize_code(
    code: ArrayLike,
    axis: MatrixAxis,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Normalize each row or column of a matrix to unit norm.

    Args:
        code: A two-dimensional real or complex matrix.
        axis: Vectors to normalize: `row` or `column`.

    Returns:
        A double-precision matrix with unit norm along the selected axis.

    Raises:
        ValueError: If `code` is not two-dimensional, has a zero-norm vector,
            or `axis` is unsupported.
        TypeError: If `code` has a non-numeric dtype.
    """
    values = as_inexact_array(code)
    if values.ndim != 2:
        raise ValueError("code must be two-dimensional")
    if axis == "row":
        axis_index = 1
    elif axis == "column":
        axis_index = 0
    else:
        raise ValueError("axis must be 'row' or 'column'")

    norms = np.linalg.norm(values, axis=axis_index, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("code cannot contain zero-norm vectors")
    return values / norms
