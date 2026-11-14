"""Code family and linear worker operation for the recovery example."""

from collections.abc import Sequence

import numpy as np

from codedcompute.codes.code_family import CodeFamily
from codedcompute.codes.code_parameters import CodeParameters


class VandermondeCode(CodeFamily):
    """Construct a real Vandermonde code with one row per worker."""

    def _generate_new_code(self, parameters: CodeParameters) -> np.ndarray:
        return np.vander(
            np.arange(1, parameters.n + 1, dtype=float),
            N=parameters.m,
            increasing=True,
        )


def sum_information(
    *, information: Sequence[np.ndarray], calculation_type: np.dtype
) -> np.ndarray:
    """Sum a worker's coefficient-weighted inputs.

    Args:
        information: Weighted arrays assigned to one code row.
        calculation_type: NumPy dtype configured for the worker.

    Returns:
        The elementwise sum of the weighted arrays.
    """
    return np.sum(information, axis=0, dtype=calculation_type)
