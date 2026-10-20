"""Offline tests for the unified session bootstrap CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/setup/bootstrap_session.py"


class TestSessionBootstrap(unittest.TestCase):

    def setUp(self):

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()

        subprocess.run(
            [
                "git", "init", "-q",
                "--initial-branch=main",
                str(self.repo),
            ],
            check=True,
        )

        subprocess.run(
            [
                "git", "-C", str(self.repo),
                "-c", "user.name=Test",
                "-c", "user.email=test@example.test",
                "commit", "--allow-empty",
                "-qm", "initial",
            ],
            check=True,
        )

        (self.repo / "README.md").write_text(
            "Test research repository\n",
            encoding="utf-8",
        )

        self.runtime_profile = self.root / "runtime.json"

        self.runtime_profile.write_text(
            json.dumps({
                "branch": "main",
                "min_free_gib": 0,
                "repo_files": ["README.md"],
                "dataset_files": [],
                "secrets": [],
            }),
            encoding="utf-8",
        )

        self.python_profile = self.root / "python.json"

        self.python_profile.write_text(
            json.dumps({
                "python_version": "3.10",
                "packages": {},
            }),
            encoding="utf-8",
        )

    def run_cli(self, *extra):

        return subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--repo", str(self.repo),
                "--runtime-profile",
                str(self.runtime_profile),
                *map(str, extra),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_runtime_only(self):

        result = self.run_cli()

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

        self.assertIn(
            "UNIFIED SESSION BOOTSTRAP COMPLETE",
            result.stdout,
        )

    def test_missing_python_is_pending(self):

        result = self.run_cli(
            "--python-profile",
            self.python_profile,
            "--venv",
            self.root / "missing-venv",
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

        self.assertIn(
            "[PENDING]",
            result.stdout,
        )

    def test_strict_mode_rejects_missing_python(self):

        result = self.run_cli(
            "--python-profile",
            self.python_profile,
            "--venv",
            self.root / "missing-venv",
            "--require-python-ready",
        )

        self.assertEqual(result.returncode, 2)

        self.assertIn(
            "[FAIL] Required Python environment",
            result.stdout,
        )

    def test_no_repository_mutations(self):

        before = subprocess.check_output(
            [
                "git", "-C", str(self.repo),
                "rev-parse", "HEAD",
            ],
            text=True,
        ).strip()

        result = self.run_cli()

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

        after = subprocess.check_output(
            [
                "git", "-C", str(self.repo),
                "rev-parse", "HEAD",
            ],
            text=True,
        ).strip()

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
