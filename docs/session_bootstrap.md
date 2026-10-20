# Reusable Experiment Session Bootstrap

## Goal

Prepare a reproducible experiment session without duplicating
environment setup code across notebooks.

The framework separates these responsibilities:

1. Repository restoration (`restore_repository.py`)
2. Runtime and dataset checks (`session_preflight.py`)
3. Python environment management (`manage_python_env.py`)
4. Unified session orchestration (`bootstrap_session.py`)
5. Selective Git changes (`scripts/git/workflow.py`)

## First-time repository restoration

The repository must be present before its scripts can run.

A minimal notebook launcher initially restores it using Git.
After cloning, reuse the framework instead of duplicating setup logic.

Repository restoration never performs implicit pulls or resets.

## CPU-only preflight

Example:

```bash
python scripts/setup/bootstrap_session.py \
  --repo /path/to/repository \
  --runtime-profile src/rtcsdd/configs/runtime_cpu_dev.json \
  --dataset-root /path/to/authorized/data
```

This runs the runtime checks without creating Python environments
or installing any packages.

## Inspect a training environment

```bash
python scripts/setup/bootstrap_session.py \
  --repo /path/to/repository \
  --runtime-profile src/rtcsdd/configs/runtime_cpu_dev.json \
  --dataset-root /path/to/authorized/data \
  --python-profile src/rtcsdd/configs/runtime_python310_training.json \
  --venv /path/to/training-venv
```

A missing environment is reported as PENDING, not silently created.

Add `--require-python-ready` for strict validation.

## Explicitly create a Python environment

Add `--create-python` to the preceding command.

This action uses uv to create the requested Python version.
It does not install project dependencies.

Pinned project dependencies are managed separately by the existing
`scripts/setup/install_training_env.py` script.

The pinned dependency installer must be separately validated before
using its installation mode in a new Kaggle runtime.

## Git workflow

Preview selected files using `scripts/git/workflow.py`.
Commit and push require explicit flags.

The workflow supports arbitrary repository roots and remotes.
It does not automatically stage every file.

## Security and reproducibility

- Never store GitHub or Hugging Face tokens in notebook source.
- Never commit private RTCFake recordings or checkpoint files.
- No automatic reset, pull, commit, or push.
- Validate package versions before GPU training.
- Record source commit and configuration for every experiment.
- Keep RTCFake Train/Dev/Progress permissions separated.
