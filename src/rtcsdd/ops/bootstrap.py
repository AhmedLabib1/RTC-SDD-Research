"""Orchestrate reusable runtime and Python environment checks.

No implicit clone, package installation, GPU usage, or Git mutations.
"""

from __future__ import annotations

from pathlib import Path

from .preflight import inspect_runtime
from .python_env import (
    create_venv,
    inspect_python,
    print_report,
    python_in_venv,
    validate_profile,
)


def bootstrap_session(
    *,
    repo: Path,
    runtime_profile: dict,
    dataset_root: Path | None = None,
    python_profile: dict | None = None,
    venv: Path | None = None,
    create_python: bool = False,
) -> dict:
    """Run runtime preflight and optional Python environment inspection."""

    repo = Path(repo).expanduser().resolve()

    if not (repo / ".git").is_dir():
        raise ValueError(
            "Repository must exist before session bootstrap"
        )

    if (python_profile is None) != (venv is None):
        raise ValueError(
            "python_profile and venv must be supplied together"
        )

    if create_python and python_profile is None:
        raise ValueError(
            "Cannot create Python without a profile and venv"
        )

    print("=" * 65, flush=True)
    print("UNIFIED EXPERIMENT SESSION BOOTSTRAP", flush=True)
    print("=" * 65, flush=True)

    # ----------------------------------------------
    # 1. Existing reusable runtime preflight
    # ----------------------------------------------

    runtime_ready = inspect_runtime(
        repo=repo,
        dataset_root=dataset_root,
        profile=runtime_profile,
    )

    if not runtime_ready:
        raise RuntimeError(
            "Runtime preflight failed; refusing to continue"
        )

    # ----------------------------------------------
    # 2. Existing reusable Python environment manager
    # ----------------------------------------------

    python_report = None
    python_ready = None

    if python_profile is not None:

        validate_profile(python_profile)

        if create_python:
            executable = create_venv(
                venv,
                python_profile,
            )
        else:
            executable = python_in_venv(venv)

        python_report = inspect_python(
            executable,
            python_profile,
        )

        print("\n[PYTHON ENVIRONMENT]", flush=True)
        print_report(python_report)

        python_ready = python_report["ready"]

        if not python_ready:
            print(
                "[PENDING] Python environment does not "
                "yet satisfy all requirements",
                flush=True,
            )

    else:
        print(
            "\n[SKIP] No Python environment profile requested",
            flush=True,
        )

    # ----------------------------------------------
    # 3. Summary
    # ----------------------------------------------

    print("\n" + "=" * 65, flush=True)

    print("[PASS] Runtime preflight complete", flush=True)
    print("[PASS] Repository untouched", flush=True)
    print("[PASS] No training executed", flush=True)

    if python_ready is True:
        print("[PASS] Python environment ready", flush=True)
    elif python_ready is False:
        print("[PENDING] Python environment setup", flush=True)

    return {
        "repo": str(repo),
        "runtime_ready": runtime_ready,
        "python_ready": python_ready,
        "python_report": python_report,
    }
