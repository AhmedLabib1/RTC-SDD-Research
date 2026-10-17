"""Restore or verify the pinned official RTC-SDD repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIN_FILE = PROJECT_ROOT / "third_party/official_rtc_sdd.pin.json"

RELATIVE_PATH = "third_party/official_rtc_sdd"

UPSTREAM_URL = "https://github.com/JunXue-tech/RTC-SDD.git"

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "xlsr_aasist/main_train.py",
    "xlsr_aasist/main_eval.py",
    "xlsr_aasist/model/model.py",
    "xlsr_aasist/utils/data_utils.py",
    "xlsr_nes2net/main_train.py",
    "xlsr_nes2net/main_eval.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def load_pin() -> str:
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))

    sha = pin["commit_sha"].lower()

    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid pinned commit SHA")

    if pin["relative_path"] != RELATIVE_PATH:
        raise ValueError("Invalid official repository location")

    if pin["upstream_url"] != UPSTREAM_URL:
        raise ValueError("Unexpected upstream repository")

    return sha


def bootstrap(verify_only: bool) -> None:
    sha = load_pin()
    checkout = PROJECT_ROOT / RELATIVE_PATH

    if not checkout.exists():
        if verify_only:
            raise FileNotFoundError(
                f"Official checkout missing: {checkout}"
            )

        checkout.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "git", "clone", "--quiet", "--no-checkout",
                UPSTREAM_URL, str(checkout),
            ],
            check=True,
        )

        subprocess.run(
            [
                "git", "-C", str(checkout),
                "checkout", "--quiet", "--detach", sha,
            ],
            check=True,
        )

    if not (checkout / ".git").exists():
        raise RuntimeError("Official checkout is not a Git repository")

    actual = git(
        "-C", str(checkout), "rev-parse", "HEAD"
    ).lower()

    if actual != sha:
        raise RuntimeError(
            f"Commit mismatch: expected {sha}, found {actual}"
        )

    dirty = git(
        "-C", str(checkout), "status", "--porcelain"
    )

    if dirty:
        raise RuntimeError("Official source has local modifications")

    for name in REQUIRED_FILES:
        if not (checkout / name).is_file():
            raise FileNotFoundError(f"Missing upstream file: {name}")

    print("[PASS] Official RTC-SDD source verified")
    print("[PASS] Commit:", actual)
    print("[PASS] Checkout is clean")
    print("[PASS] Required baseline files exist")
    print("[INFO] No dependencies or model weights installed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    bootstrap(verify_only=args.verify_only)


if __name__ == "__main__":
    main()
