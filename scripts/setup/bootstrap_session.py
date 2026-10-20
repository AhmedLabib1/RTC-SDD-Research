"""Unified portable CLI for experiment session preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.bootstrap import bootstrap_session
from rtcsdd.ops.preflight import load_profile


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--runtime-profile",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
    )

    parser.add_argument(
        "--python-profile",
        type=Path,
    )

    parser.add_argument(
        "--venv",
        type=Path,
    )

    parser.add_argument(
        "--create-python",
        action="store_true",
        help="Explicitly create a missing Python venv",
    )

    parser.add_argument(
        "--require-python-ready",
        action="store_true",
        help="Fail if Python or required packages are missing",
    )

    args = parser.parse_args()

    if (args.python_profile is None) != (args.venv is None):
        parser.error(
            "--python-profile and --venv must be provided together"
        )

    if args.create_python and args.venv is None:
        parser.error(
            "--create-python requires --venv and --python-profile"
        )

    if args.require_python_ready and args.venv is None:
        parser.error(
            "--require-python-ready requires a Python profile and venv"
        )

    runtime_profile = load_profile(
        args.runtime_profile
    )

    python_profile = None

    if args.python_profile is not None:
        python_profile = json.loads(
            args.python_profile.read_text(encoding="utf-8")
        )

    result = bootstrap_session(
        repo=args.repo,
        runtime_profile=runtime_profile,
        dataset_root=args.dataset_root,
        python_profile=python_profile,
        venv=args.venv,
        create_python=args.create_python,
    )

    if (
        args.require_python_ready
        and not result["python_ready"]
    ):
        print(
            "[FAIL] Required Python environment is not ready"
        )
        raise SystemExit(2)

    print("\n[PASS] UNIFIED SESSION BOOTSTRAP COMPLETE")


if __name__ == "__main__":
    main()
