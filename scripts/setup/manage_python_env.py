"""Inspect or explicitly create a reusable Python environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.python_env import (
    create_venv,
    inspect_python,
    print_report,
    python_in_venv,
    validate_profile,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
    )

    target = parser.add_mutually_exclusive_group(
        required=True
    )

    target.add_argument("--python", type=Path)
    target.add_argument("--venv", type=Path)

    parser.add_argument(
        "--create",
        action="store_true",
        help="Explicitly create a missing venv",
    )

    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless all requirements are installed",
    )

    args = parser.parse_args()

    profile = validate_profile(
        json.loads(
            args.profile.read_text(encoding="utf-8")
        )
    )

    if args.create and args.venv is None:
        parser.error("--create requires --venv")

    if args.venv is not None:
        executable = python_in_venv(args.venv)

        if args.create:
            executable = create_venv(
                args.venv,
                profile,
            )
    else:
        executable = args.python

    report = inspect_python(
        executable,
        profile,
    )

    print_report(report)

    if args.require_ready and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
