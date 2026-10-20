"""CPU-only tests for Python environment management."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.ops.python_env import (
    inspect_python,
    python_in_venv,
    validate_profile,
)


class TestPythonEnvironment(unittest.TestCase):

    def setUp(self):
        self.version = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}"
        )

    def test_valid_profile(self):
        profile = {
            "python_version": self.version,
            "packages": {},
        }
        self.assertEqual(
            validate_profile(profile),
            profile,
        )

    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            validate_profile({
                "python_version": "latest",
                "packages": {},
            })

    def test_current_python_ready(self):
        report = inspect_python(
            Path(sys.executable),
            {
                "python_version": self.version,
                "packages": {},
            },
        )

        self.assertTrue(report["ready"])

    def test_wrong_python_version(self):
        report = inspect_python(
            Path(sys.executable),
            {
                "python_version": "99.99",
                "packages": {},
            },
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["python_ok"])

    def test_missing_python(self):
        with TemporaryDirectory() as temp:
            report = inspect_python(
                Path(temp) / "missing-python",
                {
                    "python_version": self.version,
                    "packages": {},
                },
            )

            self.assertFalse(report["exists"])
            self.assertFalse(report["ready"])

    def test_missing_package(self):
        report = inspect_python(
            Path(sys.executable),
            {
                "python_version": self.version,
                "packages": {
                    "rtcsdd-definitely-not-installed-xyz": "*",
                },
            },
        )

        self.assertFalse(report["ready"])

    def test_venv_path(self):
        path = python_in_venv(Path("/tmp/test-venv"))

        self.assertEqual(
            path.name,
            "python.exe" if sys.platform == "win32"
            else "python",
        )


if __name__ == "__main__":
    unittest.main()
