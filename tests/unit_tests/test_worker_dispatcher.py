"""Unit tests for dispatch through workers constructed by the dispatcher."""

import unittest
from collections.abc import Sequence
from functools import partial

import numpy as np

from codedcompute.state_utils.information import (
    EncodedInformation,
    WeightedInformation,
)
from codedcompute.worker_nodes.worker_dispatcher import (
    WorkerDispatcher,
)
from codedcompute.worker_nodes.worker_node import WorkerNodeConfig

SCALE_ARGUMENT = "scale"


class TestWorkerDispatcher(unittest.TestCase):
    """Verify weighting, routing, response collection, and pool replacement."""

    def setUp(self):
        """Construct two real workers with distinct operations and precisions."""
        self.dispatcher = WorkerDispatcher()
        self.dispatcher.setup(
            2,
            [
                WorkerNodeConfig(self._sum_information, np.complex64),
                WorkerNodeConfig(partial(self._sum_information, scale=2), np.complex128),
            ],
        )

    @staticmethod
    def _sum_information(
        *,
        information: Sequence[np.ndarray],
        calculation_type: np.dtype,
        scale: float = 1,
    ) -> np.ndarray:
        """Calculate a scaled sum of the weighted inputs supplied to a worker.

        Args:
            information: Coefficient-weighted arrays for one code row.
            calculation_type: Numeric dtype configured for the worker.
            scale: Multiplier applied to the sum.

        Returns:
            The scaled elementwise sum.
        """
        total = np.zeros_like(information[0], dtype=calculation_type)
        for value in information:
            total = total + value
        return total * scale

    def test_setup_assigns_configurations_by_worker_id(self):
        """Created workers execute their assigned operations at their own precision."""
        row = [WeightedInformation(np.array([1, 2]), 1)]

        results = self.dispatcher.dispatch(
            [EncodedInformation(0, [row]), EncodedInformation(1, [row])]
        )

        self.assertEqual([result.worker_id for result in results], [0, 1])
        for result, expected, dtype in zip(
            results, ([1, 2], [2, 4]), (np.complex64, np.complex128)
        ):
            self.assertEqual(len(result.values), 1)
            np.testing.assert_array_equal(result.values[0], expected)
            self.assertEqual(result.values[0].dtype, np.dtype(dtype))

    def test_applies_complex_coefficients_and_routes_by_id_not_record_order(self):
        """Each worker receives its weighted arrays and its own options."""
        first = np.array([1 + 2j, 2 - 1j])
        second = np.array([3 - 1j, 4 + 2j])
        encoded = [
            EncodedInformation(
                1, [[WeightedInformation(first, -1j), WeightedInformation(second, 3)]]
            ),
            EncodedInformation(
                0, [[WeightedInformation(first, 2), WeightedInformation(second, -1)]]
            ),
        ]
        results = self.dispatcher.dispatch(
            encoded, worker_kwargs_by_id={0: {SCALE_ARGUMENT: 2}, 1: {SCALE_ARGUMENT: 3}}
        )

        answers = {result.worker_id: result.values for result in results}
        self.assertEqual(set(answers), {0, 1})
        self.assertTrue(all(len(values) == 1 for values in answers.values()))
        np.testing.assert_array_equal(answers[0][0], [-2 + 10j, -8j])
        np.testing.assert_array_equal(answers[1][0], [33 - 12j, 33 + 12j])
        np.testing.assert_array_equal(first, [1 + 2j, 2 - 1j])
        np.testing.assert_array_equal(second, [3 - 1j, 4 + 2j])

    def test_preserves_multiple_row_order_and_zero_answers(self):
        """Rows are executed and collected separately, including zero outputs."""
        encoded = [
            EncodedInformation(
                0,
                [
                    [
                        WeightedInformation(np.array([1, 2]), 1),
                        WeightedInformation(np.array([1, 2]), -1),
                    ],
                    [WeightedInformation(np.array([3, 4]), -1)],
                ],
            )
        ]
        results = self.dispatcher.dispatch(encoded)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].worker_id, 0)
        self.assertEqual(len(results[0].values), 2)
        np.testing.assert_array_equal(results[0].values[0], [0, 0])
        np.testing.assert_array_equal(results[0].values[1], [-3, -4])

    def test_failed_row_discards_prior_answers_and_stops_only_that_worker(self):
        """A partial failure omits the worker without interrupting other workers."""
        attempted_answers = []

        def fail_on_second_row(
            *, information: Sequence[np.ndarray], calculation_type: np.dtype
        ) -> np.ndarray | None:
            """Calculate each attempted row and omit the second answer.

            Args:
                information: Weighted arrays for the current row.
                calculation_type: Dtype configured for the worker.

            Returns:
                The calculated array, or `None` on the second invocation.
            """
            answer = self._sum_information(
                information=information, calculation_type=calculation_type
            )
            attempted_answers.append(answer)
            return None if len(attempted_answers) == 2 else answer

        self.dispatcher.setup(
            2,
            [
                WorkerNodeConfig(fail_on_second_row, np.float64),
                WorkerNodeConfig(partial(self._sum_information, scale=2), np.float64),
            ],
        )
        encoded = [
            EncodedInformation(
                0,
                [[WeightedInformation(np.array([value]), 1)] for value in (1, 2, 3)],
            ),
            EncodedInformation(1, [[WeightedInformation(np.array([4]), 1)]]),
        ]

        results = self.dispatcher.dispatch(encoded)

        np.testing.assert_array_equal(attempted_answers, [[1], [2]])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].worker_id, 1)
        self.assertEqual(len(results[0].values), 1)
        np.testing.assert_array_equal(results[0].values[0], [8])

    def test_setup_replaces_previous_workers_and_reset_clears_the_pool(self):
        """Reconfiguration replaces old workers rather than retaining their state."""
        previous_workers = list(self.dispatcher.workers.values())

        self.dispatcher.setup(
            1, [WorkerNodeConfig(partial(self._sum_information, scale=3), np.float64)]
        )
        results = self.dispatcher.dispatch(
            [EncodedInformation(0, [[WeightedInformation(np.array([1]), 1)]])]
        )

        self.assertEqual(set(self.dispatcher.workers), {0})
        for worker in previous_workers:
            self.assertIsNot(self.dispatcher.workers[0], worker)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].worker_id, 0)
        self.assertEqual(len(results[0].values), 1)
        np.testing.assert_array_equal(results[0].values[0], [3])
        self.assertEqual(results[0].values[0].dtype, np.dtype(np.float64))

        self.dispatcher.reset()

        self.assertEqual(self.dispatcher.workers, {})
