"""Check or install pinned RTC-SDD training dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]

OFFICIAL = ROOT / "third_party" / "official_rtc_sdd"
REQUIREMENTS = OFFICIAL / "requirements.txt"
PIN_FILE = ROOT / "third_party" / "official_rtc_sdd.pin.json"

TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def output(*command: str) -> str:
    return subprocess.check_output(
        command, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    # Preserve the venv path (do not resolve Python symlinks).
    python = args.python.absolute()

    if not python.is_file():
        raise FileNotFoundError(python)

    version = output(
        str(python),
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    )

    if version != "3.10":
        raise RuntimeError(f"Expected Python 3.10, got {version}")

    # Verify official source revision.
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))

    actual = output(
        "git", "-C", str(OFFICIAL), "rev-parse", "HEAD"
    )

    if actual != pin["commit_sha"]:
        raise RuntimeError("Official source commit mismatch")

    if output("git", "-C", str(OFFICIAL), "status", "--porcelain"):
        raise RuntimeError("Official source contains local modifications")

    # Verify critical dependencies.
    contents = REQUIREMENTS.read_text(encoding="utf-8")

    required_pins = (
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
    )

    for requirement in required_pins:
        if requirement not in contents.splitlines():
            raise RuntimeError(f"Missing pin: {requirement}")

    fairseq = re.findall(
        r"^-e git\+.*@([a-f0-9]{40})#egg=fairseq$",
        contents,
        re.MULTILINE,
    )

    if len(fairseq) != 1:
        raise RuntimeError("Expected one pinned Fairseq dependency")

    if shutil.which("uv") is None:
        raise RuntimeError("uv is not installed")

    # Installation commands, shared by check and install modes.
    commands = [
        [
            "uv", "pip", "install",
            "--python", str(python),
            "-r", str(REQUIREMENTS),
            "--extra-index-url", TORCH_INDEX,
        ],
        [
            "uv", "pip", "install",
            "--python", str(python),
            "-e", str(ROOT),
        ],
        [
            "uv", "pip", "check",
            "--python", str(python),
        ],
    ]

    print("[PASS] Python:", version)
    print("[PASS] Official commit:", actual)
    print("[PASS] Fairseq commit:", fairseq[0])
    print(
        "[PASS] Requirements SHA256:",
        hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
    )

    print("\nPlanned commands:")
    for command in commands:
        print("$", shlex.join(command))

    if not args.install:
        print("\n[CHECK ONLY] No packages installed.")
        print("[PENDING] Runtime compatibility verification")
        return

    for command in commands:
        subprocess.run(command, check=True)

    print("\n[PASS] Installation commands completed.")
    print("[PENDING] Model runtime verification")


if __name__ == "__main__":
    main()
