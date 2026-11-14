# Coded Compute

A Python toolkit for simulating distributed coded computation, modeling worker
stragglers, and running configurable experiments with NumPy and Hydra.

The PyPI distribution and Python import name are both `codedcompute`.

## Introduction

Coded computation introduces redundancy into worker inputs so that results can
be recovered even when some workers do not respond. Coded Compute provides
components for generating code matrices, assigning encoded inputs to workers,
executing worker operations, and decoding the surviving results.

Workers run locally and sequentially. Stragglers are modeled as missing responses,
rather than as elapsed-time delays or network failures.

## Use cases

- Prototype code families and check recovery under selected worker failures.
- Study numerical recovery with different code matrices and worker precisions.
- Run repeated experiments with seeded random straggler models.
- Describe experiment initialization and execution steps with Hydra and OmegaConf.

## Features

- **Code-family interfaces.** Extend `CodeFamily` or `RandomCodeFamily` to supply
  code matrices, with optional row or column normalization.
- **Encoding.** Apply code coefficients to information items and assign encoded
  records to identified workers.
- **Worker configuration.** Give each worker its own operation and NumPy
  calculation dtype, with optional per-worker operation arguments.
- **Straggler models.** Wrap operations with deterministic failure flags or
  probabilistic missing responses, with support for seeded NumPy generators.
- **Pseudoinverse decoding.** Recover outputs using the code rows associated with
  responding workers. Reject surviving code matrices without sufficient rank.
- **Experiment pipelines.** Instantiate objects through Hydra and execute ordered
  calls, nested pipelines, and loops against a shared workspace.

## Example: recover results with a missing worker

This example defines a Vandermonde code with three workers and two information
items. Each worker sums its coefficient-weighted inputs. Worker 0 is marked as a
straggler, and the other two responses recover the original arrays.

```python
import numpy as np

from codedcompute.codes.code_family import CodeFamily
from codedcompute.codes.code_parameters import CodeParameters
from codedcompute.decoding.decoders.pseudo_inverse import (
    PseudoInverseDecoder,
)
from codedcompute.encoding.encoder import Encoder
from codedcompute.worker_nodes.operation_wrappers import (
    with_deterministic_straggler,
)
from codedcompute.worker_nodes.worker_dispatcher import (
    WorkerDispatcher,
)
from codedcompute.worker_nodes.worker_node import WorkerNodeConfig


class VandermondeCode(CodeFamily):
    """Generate a real Vandermonde matrix for this example."""

    def _generate_new_code(self, parameters: CodeParameters) -> np.ndarray:
        return np.vander(
            np.arange(1, parameters.n + 1, dtype=float),
            N=parameters.m,
            increasing=True,
        )


def sum_information(*, information, calculation_type):
    """Sum a worker's coefficient-weighted arrays.

    Args:
        information: Weighted arrays assigned to one code row.
        calculation_type: NumPy dtype configured for the worker.

    Returns:
        The elementwise sum of the weighted arrays.
    """
    return np.sum(information, axis=0, dtype=calculation_type)


parameters = CodeParameters(m=2, n=3)
code = VandermondeCode()
code.generate_code(parameters)

inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
encoded = Encoder.encode(inputs, code)

operation = with_deterministic_straggler(sum_information)
dispatcher = WorkerDispatcher()
dispatcher.setup(
    parameters.n,
    [WorkerNodeConfig(operation, np.float64) for _ in range(parameters.n)],
)
responses = dispatcher.dispatch(
    encoded,
    worker_kwargs_by_id={0: {"straggle": True}},
)
recovered = PseudoInverseDecoder().decode(responses, code)

assert [response.worker_id for response in responses] == [1, 2]
np.testing.assert_allclose(recovered, inputs)
print(np.round(np.stack(recovered), decimals=6))
```

Expected output:

```text
[[1. 2.]
 [3. 4.]]
```

The two surviving code rows have full column rank. Recovery depends on that rank
and on numerical conditioning. Worker operations must preserve the linear
relationships represented by the code for pseudoinverse decoding to recover
their uncoded outputs. Arbitrary nonlinear operations do not generally satisfy
this requirement.

## Versioning

The current version is v0.5.0. Released Nov 14, 2026

## Installation and technical usage

### Requirements and installation

Python 3.12 or newer is required. Runtime dependencies are NumPy, Hydra Core, and
OmegaConf.

From the repository root:

```bash
python -m pip install .
```

From PyPI:

```bash
python -m pip install codedcompute
```

### Core API

1. Create `CodeParameters(m=..., n=...)`, where `m` is the number of information
   items and `n` is the number of workers, with `0 < m <= n`.
2. Subclass `CodeFamily`, implement `_generate_new_code`, and call
   `generate_code(parameters)`. The matrix must have shape `(n, m)`.
3. Call `Encoder.encode(information_set, code_family)` to create worker inputs.
4. Configure `WorkerDispatcher` with one `WorkerNodeConfig` per worker. Operations
   receive keyword arguments `information` and `calculation_type`. Returning
   `None` indicates a missing response.
5. Dispatch the inputs, then pass the responses and code family to
   `PseudoInverseDecoder.decode`.

For probabilistic stragglers, use
`with_probabilistic_straggler(operation, straggler_probability, rng=...)`.
Supply `np.random.default_rng(seed)` for reproducible draws.

### Hydra pipelines

`run_experiment` accepts an OmegaConf configuration with an `init` mapping and a
`pipeline` sequence. Hydra instantiates the initial workspace. Named steps store
their return values in that workspace, and strings beginning with `@` reference
workspace values.

```python
from omegaconf import OmegaConf

from codedcompute.hydra_pipeline.runner import run_experiment

config = OmegaConf.create({
    "init": {},
    "pipeline": [
        {"values": {"call": "numpy.arange", "args": [4]}},
        {"total": {"call": "numpy.sum", "args": ["@values"]}},
    ],
})
workspace = run_experiment(config)
print(workspace["total"])  # 6
```

Calls may use import paths or methods on workspace objects. Pipeline steps also
support `for_each` loops with `values`, an optional `as` binding, and `steps`.
Use a `run` step to execute a pipeline stored in the workspace.

### Development checks

Install the development extras and run the existing tests and lint checks:

```bash
python -m pip install -e '.[test,lint]'
python -m pytest tests
python -m ruff check src tests
```

## License

A license has not yet been specified for this repository.
