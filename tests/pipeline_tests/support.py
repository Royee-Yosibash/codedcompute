"""Fixed code and numerical operation used by pipeline configurations."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from codedcompute.codes.code_family import CodeFamily
from codedcompute.codes.code_parameters import CodeParameters

FIXED_MATRIX = ((1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, -1.0))


class FixedMatrixCode(CodeFamily):
    """Encode two inputs as themselves, their sum, and their difference."""

    def _generate_new_code(self, parameters: CodeParameters) -> NDArray:
        """Return the fixed matrix transposed into worker-by-information layout.

        Args:
            parameters: Dimensions required to match the fixed matrix.

        Returns:
            A new four-by-two real matrix.

        Raises:
            ValueError: If the requested dimensions differ from the fixed code.
        """
        dimensions = (len(FIXED_MATRIX), len(FIXED_MATRIX[0]))
        if (parameters.m, parameters.n) != dimensions:
            raise ValueError("parameters must match the fixed test matrix dimensions")
        return np.array(FIXED_MATRIX, dtype=np.float64).T


def scaled_sum(
    *,
    information: Sequence[np.ndarray],
    calculation_type: np.dtype,
    scale: float,
) -> np.ndarray:
    """Sum weighted inputs and apply the common linear operation parameter.

    Args:
        information: Weighted arrays for a worker's code row.
        calculation_type: Numeric dtype configured for the calculation.
        scale: Multiplier shared by every worker in the trial.

    Returns:
        The scaled encoded answer in the configured calculation dtype.
    """
    total = np.zeros_like(information[0], dtype=calculation_type)
    for value in information:
        total = total + value
    return total * np.asarray(scale, dtype=calculation_type)
