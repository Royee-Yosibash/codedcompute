# Coded Compute

A Python toolkit for simulating distributed coded computation, modeling worker
stragglers, and running configurable experiments with NumPy and Hydra.

The package and Python import name are both `codedcompute`.

## Introduction

Coded computation introduces redundancy into worker inputs so that results can
be recovered even when some workers do not respond. Coded Compute provides
components for generating code matrices, assigning encoded inputs to workers,
executing worker operations, and decoding the surviving results.

The computation has four stages:

1. **Choose a code.** A code matrix has one column per input item and one row per
   worker. Each row specifies the coefficients assigned to that worker.
2. **Encoding.** Associate each input with its coefficient for each
   worker. The dispatcher applies those coefficients before worker execution.
3. **Dispatch & Run.** Each worker applies a user-defined operation to its
   weighted inputs.
4. **Decoding.** Use the responding workers' code rows to reconstruct
   the outputs that would have been obtained from the original, uncoded inputs.

Workers run locally and sequentially. Stragglers are modeled as missing responses,
rather than as elapsed-time delays or network failures.

## Use cases

- Prototype code families and check recovery under selected worker failures.
- Study numerical recovery with different code matrices and worker precisions.
- Run repeated random seeded experiments.
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
- **Decoding.** Recover outputs using the outputs associated with
  responding workers.
- **Experiment pipelines.** Instantiate objects through Hydra and execute ordered
  calls, nested pipelines, and loops against a shared workspace.

## Example: recover results with a missing worker

Suppose the inputs are two arrays, `A = [1, 2]` and `B = [3, 4]`. Instead of
assigning each array to just one worker, use three workers with these code rows:

| Worker | Code row | Worker computes | Result |
| --- | --- | --- | --- |
| 0 | `[1, 1]` | `A + B` | `[4, 6]` |
| 1 | `[1, 2]` | `A + 2B` | `[7, 10]` |
| 2 | `[1, 3]` | `A + 3B` | `[10, 14]` |

If worker 0 does not respond, the other two answers still contain enough
information to recover both arrays. Subtracting worker 1's answer from worker 2's
gives `B`. Subtracting `2B` from worker 1's answer then gives `A`.

The library performs this reconstruction with `PseudoInverseDecoder` rather
than hand-written equations. The example's code matrix and summing operation are
implemented as `VandermondeCode` and `sum_information` in
[examples/components.py](examples/components.py).

This example recovers the input arrays because the worker operation sums the
weighted inputs. More generally, decoding recovers the uncoded operation's
outputs only when that operation preserves the code's linear relationships.
Arbitrary nonlinear operations do not generally meet this requirement. Recovery
also requires enough independent surviving code rows and depends on numerical
conditioning.

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

The classes follow the same encode, execute, and decode sequence:

1. Create `CodeParameters(m=..., n=...)`, where `m` is the number of input items
   and `n` is the number of workers, with `0 < m <= n`. In the example above,
   `m=2` and `n=3`.
2. Subclass `CodeFamily` and implement `_generate_new_code` to return a matrix of
   shape `(n, m)`. Call `generate_code(parameters)` to store that matrix in the
   code-family instance. `VandermondeCode` is an example implementation.
3. Call `Encoder.encode(information_set, code_family)`. It returns one record per
   worker containing the input items and their coefficients.
4. Create one `WorkerNodeConfig(operation, calculation_type)` per worker, then
   pass the configurations to `WorkerDispatcher.setup`. The operation is your
   callable; `calculation_type` is the NumPy dtype used for that worker's numeric
   inputs and passed to the operation.
5. Call `WorkerDispatcher.dispatch` with the encoded records. The dispatcher
   applies the coefficients, and each worker calls its operation with keyword
   arguments `information` and `calculation_type`. An operation returning `None`
   marks that worker as missing. The returned `WorkerResult` objects identify
   responding workers and contain their computed values.
6. Pass those results and the same code-family instance to
   `PseudoInverseDecoder.decode`. It uses the responding workers' code rows and
   returns one recovered output per original input item.

For probabilistic stragglers, use
`with_probabilistic_straggler(operation, straggler_probability, rng=...)`.
Supply `np.random.default_rng(seed)` for reproducible draws.

### Hydra pipelines

This repo's structure was designed to be support easy implementations as hydra pipelines (as YAML files).

An example can be found in [straggler_recovery.yaml](examples/straggler_recovery.yaml). This experiment runs
the three-worker recovery example described above and asserts that the decoded
arrays match the originals.

Each experiment has two main sections:

- `init` defines the objects and initial data. Hydra's `_target_` entries construct
  the code family, encoder, workers' operation, dispatcher, and decoder. These
  objects are stored in a shared workspace.
- `pipeline` lists the calls to execute in order. A `call` identifies an importable
  function or a method on a workspace object; `args` and `kwargs` supply its
  arguments. A named step stores its return value under that name in the workspace.

Quoted strings beginning with `@` refer to workspace values. For example,
`"@information_set"` supplies the initialized arrays, and `"@config.straggler_ids"`
reads the experiment's top-level straggler setting. Steps can use `for_each` with
`values`, an optional `as` binding, and `steps` to repeat operations. A `run` step
executes another pipeline stored in the workspace.

In the example file, change `straggler_ids` to select the missing worker or
`calculation_type` to change worker precision. The pipeline stores responding
workers' results under `worker_results` and decoded arrays under `recovered`.
The code family and worker operation remain in
[examples/components.py](examples/components.py); the experiment setup and
execution order live in YAML.

`codedcompute.hydra_pipeline.runner.run_experiment` executes the loaded or
Hydra-composed OmegaConf configuration and returns the populated workspace.

### Development checks

Install the development extras and run the existing tests and lint checks:

```bash
python -m pip install -e '.[test,lint]'
python -m pytest tests
python -m ruff check src tests
```

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Royee Yosibash.
