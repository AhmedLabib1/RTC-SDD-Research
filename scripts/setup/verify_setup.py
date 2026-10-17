"""Verify RTC-SDD research infrastructure before GPU experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_check(name: str, command: list[str]) -> None:
    print(f"\n[CHECK] {name}", flush=True)

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"FAILED: {name}")

    print(f"[PASS] {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--python310",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    assert args.dataset_root.is_dir(), "Dataset missing"
    assert args.python310.is_file(), "Python 3.10 missing"

    print("=" * 65)
    print("RTC-SDD | FINAL RESEARCH SETUP VERIFICATION")
    print("=" * 65)

    # 1. Framework automated tests
    run_check(
        "Framework Unit Tests",
        [sys.executable, "-m", "pytest", "-q"],
    )

    # 2. Official source pin and clean checkout
    run_check(
        "Official Baseline Source",
        [
            sys.executable,
            str(ROOT / "scripts/setup/bootstrap_official.py"),
            "--verify-only",
        ],
    )

    # 3. Environment specification (no installation)
    run_check(
        "Training Environment Specification",
        [
            sys.executable,
            str(ROOT / "scripts/setup/install_training_env.py"),
            "--python",
            str(args.python310),
        ],
    )

    # 4. Full dataset structure
    run_check(
        "Full Dataset Preflight",
        [
            sys.executable,
            "-m", "rtcsdd",
            "preflight",
            "--dataset-root",
            str(args.dataset_root),
            "--require-audit-counts",
        ],
    )

    # 5. Real audio decoding
    run_check(
        "Audio Decoding Smoke Test",
        [
            sys.executable,
            "-m", "rtcsdd",
            "smoke-audio",
            "--dataset-root",
            str(args.dataset_root),
        ],
    )

    print("\n" + "=" * 65)
    print("RESEARCH INFRASTRUCTURE: ALL CHECKS PASSED")
    print("=" * 65)

    print("\nStill pending:")
    print("- Training dependencies installation")
    print("- Fairseq runtime compatibility")
    print("- XLS-R checkpoint loading")
    print("- GPU forward/backward smoke test")


if __name__ == "__main__":
    main()
