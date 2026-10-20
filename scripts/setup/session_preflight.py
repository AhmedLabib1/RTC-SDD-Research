"""Run a reusable environment preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.preflight import (
    inspect_runtime,
    load_profile,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    profile = load_profile(args.profile)

    ready = inspect_runtime(
        repo=args.repo,
        dataset_root=args.dataset_root,
        profile=profile,
    )

    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
