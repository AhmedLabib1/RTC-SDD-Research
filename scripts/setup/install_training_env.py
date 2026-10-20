"""Validate or explicitly install pinned RTC-SDD dependencies.

--plan-only: CPU-safe, no installation, source checkout optional.
Default: verify existing Python 3.10 and official source.
--install: install pinned dependencies into the specified venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.training_requirements import (
    FAIRSEQ_SHA,
    convert_requirements,
    install_commands,
)

OFFICIAL = ROOT / "third_party/official_rtc_sdd"
REQUIREMENTS = OFFICIAL / "requirements.txt"
PIN_FILE = ROOT / "third_party/official_rtc_sdd.pin.json"


def command_output(*command: str) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def check_python(executable: Path) -> None:
    if not executable.is_file():
        raise FileNotFoundError(executable)

    version = command_output(
        str(executable),
        "-c",
        "import sys; print("
        "f'{sys.version_info.major}.{sys.version_info.minor}'"
        ")",
    )

    if version != "3.10":
        raise RuntimeError(
            f"Python 3.10 required, found {version}"
        )

    print("[PASS] Python 3.10")


def check_official() -> str:
    pin = json.loads(
        PIN_FILE.read_text(encoding="utf-8")
    )

    expected = pin["commit_sha"]

    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected):
        raise ValueError("Invalid official commit pin")

    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(
            "Official requirements.txt not restored"
        )

    actual = command_output(
        "git", "-C", str(OFFICIAL),
        "rev-parse", "HEAD",
    )

    if actual.lower() != expected.lower():
        raise RuntimeError(
            "Official repository commit mismatch"
        )

    dirty = command_output(
        "git", "-C", str(OFFICIAL),
        "status", "--porcelain",
    )

    if dirty:
        raise RuntimeError(
            "Official repository has local modifications"
        )

    print("[PASS] Official source commit:", actual)

    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--python",
        type=Path,
        required=True,
    )

    actions = parser.add_mutually_exclusive_group()

    actions.add_argument(
        "--plan-only",
        action="store_true",
    )

    actions.add_argument(
        "--install",
        action="store_true",
    )

    args = parser.parse_args()

    python = args.python.absolute()

    print("=" * 68)
    print("RTC-SDD PINNED TRAINING DEPENDENCIES")
    print("=" * 68)

    print("Python target:", python)
    print("Fairseq pin  :", FAIRSEQ_SHA)

    # ----------------------------------------------
    # PLAN ONLY — no requirements downloaded
    # ----------------------------------------------

    if args.plan_only:
        print("\n[PLAN ONLY]")
        print("1. Verify Python 3.10")
        print("2. Verify pinned official RTC-SDD checkout")
        print("3. Convert legacy Fairseq requirement in temp")
        print("4. Install pinned requirements using uv")
        print("5. Install project and HF checkpoint support")
        print("6. Run uv pip check")

        print("\n[PASS] No installation executed")
        print("[PASS] Official source unchanged")
        print("[PASS] No GPU used")
        return

    # ----------------------------------------------
    # Required checks for actual environment
    # ----------------------------------------------

    check_python(python)
    check_official()

    original = REQUIREMENTS.read_bytes()

    converted = convert_requirements(
        original.decode("utf-8")
    )

    print("[PASS] Fairseq pin:", FAIRSEQ_SHA)
    print("[PASS] Requirements conversion validated")
    print(
        "[PASS] Original requirements SHA256:",
        hashlib.sha256(original).hexdigest(),
    )

    # ----------------------------------------------
    # Temporary conversion — never edit official
    # ----------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="rtcsdd-requirements-"
    ) as folder:

        compatible = (
            Path(folder) / "requirements-uv.txt"
        )

        compatible.write_text(
            converted,
            encoding="utf-8",
        )

        commands = install_commands(
            python=python,
            requirements=compatible,
            project=ROOT,
        )

        if not args.install:
            print("\n[CHECK ONLY] Planned stages:")

            for index, command in enumerate(commands, 1):
                print(
                    f"{index}.",
                    command[0],
                    command[1],
                    command[2],
                )

            print("\n[PASS] No packages installed")
            return

        if shutil.which("uv") is None:
            raise RuntimeError(
                "uv must be installed before --install"
            )

        print("\n[INSTALL] Executing pinned dependency plan")

        for index, command in enumerate(commands, 1):
            print(
                f"\n[STAGE {index}/{len(commands)}]",
                flush=True,
            )

            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
            )

    # Verify that the official requirements remain unchanged.
    after = REQUIREMENTS.read_bytes()

    if after != original:
        raise RuntimeError(
            "Official requirements changed unexpectedly"
        )

    print("\n[PASS] Pinned dependency installation completed")
    print("[PASS] Official requirements unchanged")
    print("[INFO] GPU/model runtime verification still required")


if __name__ == "__main__":
    main()
