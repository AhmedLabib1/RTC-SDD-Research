"""End-to-end CPU-only tests of reusable experiment operations.

Uses temporary local repositories. Never contacts GitHub.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    command = ["git"]

    if cwd is not None:
        command.extend(["-C", str(cwd)])

    command.extend(str(arg) for arg in args)

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def cli(script, *args):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            *(str(arg) for arg in args),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if result.returncode:
        raise AssertionError(
            f"CLI failed: {script}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return result.stdout


class TestOperationsIntegration(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.base = Path(temp.name)
        self.remote = self.base / "remote.git"
        self.seed = self.base / "seed"
        self.clone = self.base / "clone"

        git(
            "init",
            "-q",
            "--bare",
            "--initial-branch=main",
            self.remote,
        )

        git(
            "init",
            "-q",
            "--initial-branch=main",
            self.seed,
        )

        git(
            "config",
            "user.name",
            "Integration Test",
            cwd=self.seed,
        )

        git(
            "config",
            "user.email",
            "test@example.test",
            cwd=self.seed,
        )

        (self.seed / "README.md").write_text(
            "Initial project\n",
            encoding="utf-8",
        )

        git("add", "README.md", cwd=self.seed)
        git("commit", "-qm", "Initial", cwd=self.seed)

        git(
            "remote",
            "add",
            "origin",
            self.remote,
            cwd=self.seed,
        )

        git(
            "push",
            "-q",
            "-u",
            "origin",
            "main",
            cwd=self.seed,
        )

    def restore(self):
        return cli(
            "scripts/setup/restore_repository.py",
            "--url",
            self.remote,
            "--destination",
            self.clone,
            "--branch",
            "main",
        )

    def test_complete_workflow(self):

        # 1. Clone through reusable manager.
        output = self.restore()
        self.assertIn("CLONED", output)

        git(
            "config",
            "user.name",
            "Integration Test",
            cwd=self.clone,
        )

        git(
            "config",
            "user.email",
            "test@example.test",
            cwd=self.clone,
        )

        # 2. Run reusable preflight.
        profile = self.base / "profile.json"

        profile.write_text(
            json.dumps({
                "branch": "main",
                "min_free_gib": 0,
                "repo_files": ["README.md"],
                "dataset_files": [],
                "secrets": [],
            }),
            encoding="utf-8",
        )

        output = cli(
            "scripts/setup/session_preflight.py",
            "--repo",
            self.clone,
            "--profile",
            profile,
        )

        self.assertIn(
            "ENVIRONMENT PREFLIGHT COMPLETE",
            output,
        )

        # 3. Create a project source file.
        filename = "analysis/result.txt"
        destination = self.clone / filename
        destination.parent.mkdir()

        destination.write_text(
            "Reproducible experiment result\n",
            encoding="utf-8",
        )

        # 4. Preview must not stage or commit.
        output = cli(
            "scripts/git/workflow.py",
            "--repo",
            self.clone,
            "--file",
            filename,
        )

        self.assertIn("PREVIEW ONLY", output)

        self.assertEqual(
            git(
                "diff",
                "--cached",
                "--name-only",
                cwd=self.clone,
            ),
            "",
        )

        # 5. Commit and push to LOCAL remote only.
        output = cli(
            "scripts/git/workflow.py",
            "--repo",
            self.clone,
            "--file",
            filename,
            "--message",
            "test: integration workflow",
            "--commit",
            "--push",
        )

        self.assertIn("PUSH COMPLETE", output)

        # 6. Remote must have the exact commit.
        local_head = git(
            "rev-parse",
            "HEAD",
            cwd=self.clone,
        )

        remote_head = git(
            "--git-dir",
            self.remote,
            "rev-parse",
            "refs/heads/main",
        )

        self.assertEqual(local_head, remote_head)

        committed_files = git(
            "show",
            "--pretty=format:",
            "--name-only",
            "HEAD",
            cwd=self.clone,
        )

        self.assertEqual(
            committed_files,
            filename,
        )

    def test_existing_work_is_preserved(self):

        self.restore()

        readme = self.clone / "README.md"

        readme.write_text(
            "My uncommitted research work\n",
            encoding="utf-8",
        )

        before = git(
            "rev-parse",
            "HEAD",
            cwd=self.clone,
        )

        output = self.restore()

        after = git(
            "rev-parse",
            "HEAD",
            cwd=self.clone,
        )

        self.assertIn(
            "EXISTING",
            output,
        )

        self.assertEqual(before, after)

        self.assertEqual(
            readme.read_text(encoding="utf-8"),
            "My uncommitted research work\n",
        )


if __name__ == "__main__":
    unittest.main()
