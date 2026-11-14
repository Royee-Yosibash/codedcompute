"""Full coded-computation tests entered through Hydra composition and execution."""

import unittest
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from codedcompute.hydra_pipeline.runner import run_experiment

CONFIG_DIRECTORY = Path(__file__).resolve().parent / "configs"
TRIAL_RESULTS_KEY = "trial_results"
SURVIVOR_IDS_KEY = "survivor_ids"
WORKER_RESULTS_KEY = "worker_results"
DECODED_KEY = "decoded_information"
WORKERS_KEY = "workers"
INFORMATION_KEY = "information_set"
DISPATCHER_KEY = "dispatcher"
REAL_INFORMATION = np.array(
    [[[1.0, 2.0], [3.0, 4.0]], [[-2.0, 4.0], [3.0, -1.0]]]
)
COMPLEX_INFORMATION = np.array(
    [
        [[1 + 2j, 3 - 1j], [2 + 4j, -1 - 2j]],
        [[-2 + 0.5j, 4 + 3j], [1 - 3j, 2 + 1j]],
    ]
)
DATA_CASES = (
    ("data=real", REAL_INFORMATION, np.float64),
    ("data=complex", COMPLEX_INFORMATION, np.complex128),
)
PRECISION_INPUT_OVERRIDE = "init.information_set.a=[[16777216.0],[1.0]]"
PRECISION_OVERRIDE = "calculation_type={dtype}"
SCALE_OVERRIDE = "scale={scale}"
SURVIVOR_PAIRS = tuple(combinations(range(4), 2))
PAIR_SWEEP_OVERRIDE = f"survivor_sets={[list(pair) for pair in SURVIVOR_PAIRS]}"


class TestHydraWorkerPipeline(unittest.TestCase):
    """Exercise real encoding, workers, straggling, and pseudoinverse recovery."""

    @staticmethod
    def _compose_config(overrides: Sequence[str] = ()) -> DictConfig:
        """Compose the experiment and its component, data, and trial configurations.

        Args:
            overrides: Hydra overrides applied to the composed configuration.

        Returns:
            The complete experiment configuration.
        """
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIRECTORY)):
            return compose(config_name="experiment", overrides=list(overrides))

    @staticmethod
    def _expected_worker_answers(information: np.ndarray) -> list[np.ndarray]:
        """Calculate the four fixed-code answers without using the encoder.

        Args:
            information: The two original information arrays.

        Returns:
            The original inputs, their sum, and their difference, in worker order.
        """
        first, second = information
        return [first, second, first + second, first - second]

    def _assert_trial(
        self,
        trial: dict[str, Any],
        survivor_ids: Sequence[int],
        expected_workers: Sequence[np.ndarray],
        expected_decoded: np.ndarray,
        calculation_type: Any,
        atol: float = 1e-12,
    ):
        """Check surviving answers and reconstruction against numerical expectations.

        Args:
            trial: Recorded results from one configured trial.
            survivor_ids: Expected responding workers, in dispatch order.
            expected_workers: Independently calculated answers for all four workers.
            expected_decoded: Expected reconstructed operation outputs.
            calculation_type: Expected worker calculation dtype.
            atol: Absolute tolerance for pseudoinverse reconstruction.
        """
        self.assertEqual(trial[SURVIVOR_IDS_KEY], list(survivor_ids))
        results = trial[WORKER_RESULTS_KEY]
        self.assertEqual([result.worker_id for result in results], list(survivor_ids))
        for result in results:
            self.assertEqual(len(result.values), 1)
            np.testing.assert_array_equal(
                result.values[0], expected_workers[result.worker_id]
            )
            self.assertEqual(result.values[0].dtype, np.dtype(calculation_type))
        self.assertEqual(len(trial[DECODED_KEY]), len(expected_decoded))
        np.testing.assert_allclose(
            np.stack(trial[DECODED_KEY]), expected_decoded, rtol=0, atol=atol
        )

    def test_full_pipeline_recovers_real_and_complex_calculation_outputs(self):
        """Composed experiments encode and recover arrays without changing inputs."""
        for data_override, information, calculation_type in DATA_CASES:
            with self.subTest(data=data_override):
                config = self._compose_config([data_override])

                workspace = run_experiment(config)

                self.assertEqual(len(workspace[TRIAL_RESULTS_KEY]), 1)
                self._assert_trial(
                    workspace[TRIAL_RESULTS_KEY][0],
                    range(4),
                    self._expected_worker_answers(information),
                    information,
                    calculation_type,
                )
                np.testing.assert_array_equal(workspace[INFORMATION_KEY], information)

    def test_precision_and_scale_overrides_change_answers_and_reconstruction(self):
        """Configured precision controls rounding before the common scaling step."""
        cases = (
            (np.float64, 1, [16777216, 1, 16777217, 16777215]),
            (np.float32, 1, [16777216, 1, 16777216, 16777215]),
            (np.float64, 3, [50331648, 3, 50331651, 50331645]),
            (np.float32, 3, [50331648, 3, 50331648, 50331644]),
        )
        for calculation_type, scale, answers in cases:
            with self.subTest(calculation_type=calculation_type, scale=scale):
                config = self._compose_config(
                    [
                        PRECISION_INPUT_OVERRIDE,
                        PRECISION_OVERRIDE.format(dtype=np.dtype(calculation_type).name),
                        SCALE_OVERRIDE.format(scale=scale),
                    ]
                )
                expected_workers = np.array(answers, dtype=np.float64).reshape(4, 1)
                # G @ G.T = 3I, giving these explicit reconstruction equations.
                expected_decoded = np.stack(
                    [
                        (expected_workers[0] + expected_workers[2] + expected_workers[3])
                        / 3,
                        (expected_workers[1] + expected_workers[2] - expected_workers[3])
                        / 3,
                    ]
                )

                workspace = run_experiment(config)

                self.assertEqual(len(workspace[TRIAL_RESULTS_KEY]), 1)
                self._assert_trial(
                    workspace[TRIAL_RESULTS_KEY][0],
                    range(4),
                    expected_workers,
                    expected_decoded,
                    calculation_type,
                    atol=1e-7,
                )

    def test_straggler_sweep_recovers_from_every_two_worker_survivor_set(self):
        """All six survivor pairs recover the inputs across repeated configured trials."""
        config = self._compose_config([PAIR_SWEEP_OVERRIDE])

        workspace = run_experiment(config)

        trials = workspace[TRIAL_RESULTS_KEY]
        self.assertEqual(len(trials), len(SURVIVOR_PAIRS))
        for trial, survivors in zip(trials, SURVIVOR_PAIRS):
            with self.subTest(survivors=survivors):
                self._assert_trial(
                    trial,
                    survivors,
                    self._expected_worker_answers(REAL_INFORMATION),
                    REAL_INFORMATION,
                    np.float64,
                )

    def test_repeated_experiments_have_independent_results_and_worker_pools(self):
        """Repeating a configuration neither accumulates trials nor shares mutable state."""
        config = self._compose_config([PAIR_SWEEP_OVERRIDE])
        original_config = OmegaConf.to_container(config, resolve=True)

        first = run_experiment(config)
        second = run_experiment(config)

        self.assertIsNot(first, second)
        for workspace in (first, second):
            self.assertEqual(len(workspace[TRIAL_RESULTS_KEY]), len(SURVIVOR_PAIRS))
            for trial, survivors in zip(workspace[TRIAL_RESULTS_KEY], SURVIVOR_PAIRS):
                self._assert_trial(
                    trial,
                    survivors,
                    self._expected_worker_answers(REAL_INFORMATION),
                    REAL_INFORMATION,
                    np.float64,
                )
        for left, right in zip(first[TRIAL_RESULTS_KEY], second[TRIAL_RESULTS_KEY]):
            self.assertIsNot(left, right)
            self.assertIsNot(left[WORKERS_KEY], right[WORKERS_KEY])
            for worker_id in range(4):
                self.assertIsNot(
                    left[WORKERS_KEY][worker_id], right[WORKERS_KEY][worker_id]
                )
            for left_array, right_array in zip(left[DECODED_KEY], right[DECODED_KEY]):
                self.assertFalse(np.shares_memory(left_array, right_array))
        self.assertEqual(OmegaConf.to_container(config, resolve=True), original_config)

        first[TRIAL_RESULTS_KEY][0][DECODED_KEY][0][:] = -123
        first[TRIAL_RESULTS_KEY].clear()
        first[DISPATCHER_KEY].reset()

        self.assertEqual(len(second[TRIAL_RESULTS_KEY]), len(SURVIVOR_PAIRS))
        self.assertEqual(set(second[DISPATCHER_KEY].workers), set(range(4)))
        for trial in second[TRIAL_RESULTS_KEY]:
            np.testing.assert_allclose(
                np.stack(trial[DECODED_KEY]), REAL_INFORMATION, rtol=0, atol=1e-12
            )
