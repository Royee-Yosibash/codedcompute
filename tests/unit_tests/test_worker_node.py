"""Unit tests for worker precision and operation invocation."""

import unittest
from collections.abc import Sequence
from unittest.mock import Mock, sentinel

import numpy as np

from codedcompute.worker_nodes.worker_node import (
    WorkerNode,
    WorkerNodeConfig,
)

INFORMATION_ARGUMENT = "information"
CALCULATION_TYPE_ARGUMENT = "calculation_type"
SCALE_ARGUMENT = "scale"


class TestWorkerNode(unittest.TestCase):
    """Verify execution uses converted inputs and forwards operation options."""

    @staticmethod
    def _sum_information(
        *, information: Sequence[np.ndarray], calculation_type: np.dtype
    ) -> np.ndarray:
        """Add inputs without promoting the dtype selected by the worker.

        Args:
            information: Arrays converted by the worker.
            calculation_type: Expected dtype of the calculation.

        Returns:
            The elementwise sum in the input dtype.
        """
        total = np.zeros_like(information[0])
        for value in information:
            np.testing.assert_equal(value.dtype, calculation_type)
            total = total + value
        return total

    def test_calculation_precision_changes_cancellation_result(self):
        """Inputs are cast before arithmetic, producing distinct rounded sums."""
        information = [np.array([value], dtype=np.float64) for value in (2**24, 1, -(2**24))]

        for calculation_type, expected in ((np.float32, [0]), (np.float64, [1])):
            with self.subTest(calculation_type=calculation_type):
                worker = WorkerNode(
                    0, WorkerNodeConfig(self._sum_information, calculation_type)
                )

                result = worker.execute(information=information)

                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.dtype, np.dtype(calculation_type))

    def test_complex_calculation_preserves_imaginary_components(self):
        """Complex inputs remain complex when converted to worker precision."""
        worker = WorkerNode(0, WorkerNodeConfig(self._sum_information, np.complex64))

        result = worker.execute(
            information=[
                np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
                np.array([5 - 3j, -2 + 6j], dtype=np.complex128),
            ]
        )

        np.testing.assert_array_equal(result, [6 - 1j, 1 + 2j])
        self.assertEqual(result.dtype, np.dtype(np.complex64))

    def test_forwards_converted_inputs_options_and_operation_result(self):
        """The operation receives ordered copies, its dtype, and caller options."""
        operation = Mock(return_value=sentinel.answer)
        worker = WorkerNode(0, WorkerNodeConfig(operation, np.float32))
        information = [np.array([1, 2]), np.array([3, 4])]
        originals = [value.copy() for value in information]

        result = worker.execute(information=information, scale=3)

        self.assertIs(result, sentinel.answer)
        operation.assert_called_once()
        arguments = operation.call_args.kwargs
        self.assertEqual(arguments[CALCULATION_TYPE_ARGUMENT], np.dtype(np.float32))
        self.assertEqual(arguments[SCALE_ARGUMENT], 3)
        self.assertEqual(len(arguments[INFORMATION_ARGUMENT]), 2)
        for converted, original, source in zip(
            arguments[INFORMATION_ARGUMENT], originals, information
        ):
            np.testing.assert_array_equal(converted, original)
            self.assertEqual(converted.dtype, np.dtype(np.float32))
            converted[:] = 0
            np.testing.assert_array_equal(source, original)

    def test_complex_inputs_cannot_silently_lose_imaginary_components(self):
        """An incompatible real dtype fails before invoking the operation."""
        operation = Mock()
        worker = WorkerNode(0, WorkerNodeConfig(operation, np.float64))

        with self.assertRaises(TypeError):
            worker.execute(information=[np.array([1 + 2j])])

        operation.assert_not_called()
