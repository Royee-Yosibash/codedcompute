# Coded Compute

A Python toolkit for simulating distributed coded computation, modeling worker
stragglers, and running configurable experiments with NumPy and Hydra.

[Source code and documentation](https://github.com/Royee-Yosibash/codedcompute)

## Introduction

Coded computation adds redundancy to worker inputs so that outputs can be
recovered when some workers do not respond. Coded Compute models this process
in four stages:

1. Generate a code matrix whose rows specify each worker's input coefficients.
2. Encode the inputs using those coefficients.
3. Execute user-defined worker operations and collect responding workers' results.
4. Decode those results using the corresponding code rows.

Workers execute locally and sequentially. Stragglers are modeled as missing
responses rather than elapsed-time delays or network failures.

## Use cases

- Prototype code families and study recovery under worker failures.
- Compare numerical recovery across code matrices and worker calculation bit depth.
- Run reproducible experiments with seeded probabilistic straggler models.
- Define experiment setup and execution Hydra using YAML files.

## Features

- Extensible deterministic and randomized code-family interfaces.
- Coefficient-based input encoding and worker assignment.
- Per-worker operations, calculation dtypes, and operation arguments.
- Deterministic and probabilistic straggler wrappers.
- Decoding with surviving-code rank checks.
- Configurable pipelines with shared workspace values, loops, and nested pipelines.

## Recovery example

With two input arrays, `A` and `B`, three workers can compute `A + B`, `A + 2B`,
and `A + 3B`. Any two responses are sufficient to reconstruct both arrays.

The repository includes a
[YAML recovery experiment](https://github.com/Royee-Yosibash/codedcompute/blob/main/examples/straggler_recovery.yaml)
and its
[Python components](https://github.com/Royee-Yosibash/codedcompute/blob/main/examples/components.py).
The experiment configures three workers, marks one as a straggler, decodes the
remaining responses, and checks recovery against the original arrays.

Recovery requires enough independent surviving code rows and an operation that
preserves the code's linear relationships. Arbitrary nonlinear operations are
not generally recoverable through pseudoinverse decoding. Numerical conditioning
also affects reconstruction accuracy.

## Versioning

Latest Version is v0.5.0 released on Nov 14th, 2026

## Installation and usage

Python 3.12 or newer is required.

```bash
python -m pip install codedcompute
```

The Python import name is `codedcompute`. Runtime dependencies are NumPy, Hydra
Core, and OmegaConf.

Start with the
[core API documentation](https://github.com/Royee-Yosibash/codedcompute#core-api)
to define a code family, encode inputs, configure workers, and decode results.
For YAML-based experiments, see
[Hydra pipelines](https://github.com/Royee-Yosibash/codedcompute#hydra-pipelines).

## License

Licensed under the
[MIT License](https://github.com/Royee-Yosibash/codedcompute/blob/main/LICENSE).
Copyright (c) 2026 Royee Yosibash.
