"""Exercise the repository's YAML recovery experiment through Hydra."""

import unittest

import numpy as np
from hydra import compose, initialize_config_module

from codedcompute.hydra_pipeline.runner import run_experiment

CONFIG_MODULE = "examples"
CONFIG_NAME = "straggler_recovery"
STRAGGLER_OVERRIDE = "straggler_ids=[{worker_id}]"
WORKER_RESULTS_KEY = "worker_results"
RECOVERED_KEY = "recovered"
EXPECTED_INPUTS = np.array([[1.0, 2.0], [3.0, 4.0]])


class TestExampleStragglerRecovery(unittest.TestCase):
    """Check recovery with each worker missing from the example experiment."""

    def test_recovers_original_arrays_with_each_worker_missing(self):
        """Every surviving pair recovers the inputs and omits the selected worker."""
        for worker_id in range(3):
            with self.subTest(straggler=worker_id):
                with initialize_config_module(version_base=None, config_module=CONFIG_MODULE):
                    config = compose(
                        config_name=CONFIG_NAME,
                        overrides=[STRAGGLER_OVERRIDE.format(worker_id=worker_id)],
                    )

                workspace = run_experiment(config)

                self.assertEqual(
                    [result.worker_id for result in workspace[WORKER_RESULTS_KEY]],
                    [survivor for survivor in range(3) if survivor != worker_id],
                )
                np.testing.assert_allclose(workspace[RECOVERED_KEY], EXPECTED_INPUTS)
