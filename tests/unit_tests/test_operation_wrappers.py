"""Unit tests for worker omission wrappers."""

import unittest
from unittest.mock import Mock, call, sentinel

import numpy as np

from codedcompute.worker_nodes.operation_wrappers import (
    with_deterministic_straggler,
    with_probabilistic_straggler,
)

STRAGGLE_ARGUMENT = "straggle"


class TestDeterministicStraggler(unittest.TestCase):
    """Verify the explicit omission flag controls operation execution."""

    def test_straggler_does_not_execute_the_operation(self):
        """A straggling call returns no answer without running its operation."""
        operation = Mock()
        wrapped = with_deterministic_straggler(operation)

        result = wrapped(
            information=[np.array([1, 2])],
            calculation_type=np.dtype(np.float64),
            straggle=True,
        )

        self.assertIsNone(result)
        operation.assert_not_called()

    def test_response_forwards_arguments_and_preserves_zero_result(self):
        """Default and explicit responding calls forward options and return zero."""
        information = [np.array([1, 2])]
        calculation_type = np.dtype(np.float32)
        for options in ({}, {STRAGGLE_ARGUMENT: False}):
            with self.subTest(options=options):
                operation = Mock(return_value=0)
                wrapped = with_deterministic_straggler(operation)

                result = wrapped(
                    information=information,
                    calculation_type=calculation_type,
                    scale=3,
                    **options,
                )

                self.assertEqual(result, 0)
                operation.assert_called_once_with(
                    information=information,
                    calculation_type=calculation_type,
                    scale=3,
                )


class TestProbabilisticStraggler(unittest.TestCase):
    """Verify omission boundaries and reproducible seeded response sequences."""

    def test_zero_and_one_probabilities_control_every_operation_call(self):
        """Probability zero always executes and probability one always omits."""
        information = [np.array([1 + 2j])]
        calculation_type = np.dtype(np.complex128)
        for probability, expected_calls, expected_result in (
            (0, 4, sentinel.answer),
            (1, 0, None),
        ):
            with self.subTest(probability=probability):
                operation = Mock(return_value=sentinel.answer)
                wrapped = with_probabilistic_straggler(
                    operation, probability, rng=np.random.default_rng(7)
                )

                results = [
                    wrapped(
                        information=information,
                        calculation_type=calculation_type,
                        scale=3,
                    )
                    for _ in range(4)
                ]

                self.assertEqual(results, [expected_result] * 4)
                self.assertEqual(
                    operation.call_args_list,
                    [
                        call(
                            information=information,
                            calculation_type=calculation_type,
                            scale=3,
                        )
                    ]
                    * expected_calls,
                )

    def test_seeded_generator_produces_expected_mixed_response_sequence(self):
        """Each invocation consumes a draw from the supplied seeded generator."""
        expected_results = [
            sentinel.answer,
            sentinel.answer,
            sentinel.answer,
            None,
            None,
            sentinel.answer,
            None,
            sentinel.answer,
        ]
        for _ in range(2):
            operation = Mock(return_value=sentinel.answer)
            wrapped = with_probabilistic_straggler(
                operation, 0.5, rng=np.random.default_rng(7)
            )

            results = [
                wrapped(information=[1], calculation_type=np.dtype(np.float64))
                for _ in range(8)
            ]

            self.assertEqual(results, expected_results)
            self.assertEqual(operation.call_count, 5)
