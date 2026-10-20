"""Reusable Python environment inspection and explicit creation.

Does not install project dependencies or select GPU devices.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROBE_CODE = """
import importlib.metadata as metadata
import json
import sys

packages = json.loads(sys.argv[1])
installed = {}

for name in packages:
    try:
        installed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed[name] = None

print(json.dumps({
    "python_version": (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    ),
    "executable": sys.executable,
    "packages": installed,
}))
"""


def validate_profile(profile: dict) -> dict:
    """Validate a portable environment specification."""

    if not isinstance(profile, dict):
        raise ValueError("Profile must be an object")

    version = profile.get("python_version")

    if not isinstance(version, str):
        raise ValueError("Missing python_version")

    if not re.fullmatch(r"\d+\.\d+", version):
        raise ValueError("Invalid Python version")

    packages = profile.get("packages", {})

    if not isinstance(packages, dict):
        raise ValueError("packages must be an object")

    for name, requirement in packages.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise ValueError("Invalid package name")

        if not isinstance(requirement, str) or not requirement:
            raise ValueError("Invalid package requirement")

    return profile


def python_in_venv(venv: Path) -> Path:
    """Return platform-appropriate venv Python path."""

    venv = Path(venv).expanduser().absolute()

    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"

    return venv / "bin" / "python"


def inspect_python(
    executable: Path,
    profile: dict,
) -> dict:
    """Check Python and installed versions without importing ML packages."""

    validate_profile(profile)

    executable = Path(executable).expanduser().absolute()

    if not executable.is_file():
        return {
            "exists": False,
            "ready": False,
            "executable": str(executable),
            "reason": "Python executable missing",
            "packages": {},
        }

    result = subprocess.run(
        [
            str(executable),
            "-c",
            PROBE_CODE,
            json.dumps(profile.get("packages", {})),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Python environment inspection failed"
        )

    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid environment probe response"
        ) from exc

    expected_python = profile["python_version"]
    actual_python = observed["python_version"]

    package_results = {}
    packages_ok = True

    for name, expected in profile.get("packages", {}).items():
        actual = observed["packages"].get(name)

        ok = (
            actual is not None
            and (expected == "*" or actual == expected)
        )

        package_results[name] = {
            "expected": expected,
            "installed": actual,
            "ok": ok,
        }

        packages_ok = packages_ok and ok

    python_ok = actual_python == expected_python

    return {
        "exists": True,
        "ready": python_ok and packages_ok,
        "executable": str(executable),
        "python_expected": expected_python,
        "python_actual": actual_python,
        "python_ok": python_ok,
        "packages": package_results,
    }


def create_venv(
    venv: Path,
    profile: dict,
) -> Path:
    """Explicitly create an environment using uv.

    Existing environments are never replaced.
    Dependency installation is a separate operation.
    """

    validate_profile(profile)

    venv = Path(venv).expanduser().absolute()
    executable = python_in_venv(venv)

    if executable.is_file():
        report = inspect_python(executable, {
            "python_version": profile["python_version"],
            "packages": {},
        })

        if not report["python_ok"]:
            raise RuntimeError(
                "Existing environment uses the wrong Python version"
            )

        return executable

    if venv.exists() and any(venv.iterdir()):
        raise RuntimeError(
            "Nonempty target directory is not a valid environment"
        )

    uv = shutil.which("uv")

    if not uv:
        raise RuntimeError(
            "uv is required to create Python environments"
        )

    subprocess.run(
        [
            uv,
            "venv",
            str(venv),
            "--python",
            profile["python_version"],
        ],
        check=True,
    )

    if not executable.is_file():
        raise RuntimeError(
            "Environment creation did not produce Python"
        )

    return executable


def print_report(report: dict) -> None:
    print("=" * 65)
    print("PYTHON ENVIRONMENT REPORT")
    print("=" * 65)

    print("Executable:", report["executable"])

    if not report["exists"]:
        print("[PENDING]", report["reason"])
        return

    print(
        "Python:",
        report["python_actual"],
        "(expected", report["python_expected"] + ")",
    )

    print(
        "[PASS]" if report["python_ok"] else "[FAIL]",
        "Python version",
    )

    for name, item in report["packages"].items():
        print(
            "[PASS]" if item["ok"] else "[PENDING]",
            name,
            "installed=", item["installed"],
            "expected=", item["expected"],
        )

    print(
        "[READY]" if report["ready"] else "[NOT READY]",
        "Environment status",
    )
