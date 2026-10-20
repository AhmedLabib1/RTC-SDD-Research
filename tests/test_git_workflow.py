"""Offline tests; pushes only to temporary local repositories."""

import subprocess
import sys
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.ops.git_workflow import (
    preview,
    commit,
    push,
)


def git(*args, cwd=None):
    command = [
        "git",
        *(
            ["-C", str(cwd)]
            if cwd else []
        ),
        *map(str, args),
    ]

    return subprocess.check_output(
        command,
        text=True,
    ).strip()


class TestGitWorkflow(unittest.TestCase):

    def setUp(self):

        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        root = Path(temp.name)

        self.remote = root / "remote.git"
        self.repo = root / "repo"

        git(
            "init", "-q", "--bare",
            self.remote,
        )

        git(
            "init", "-q",
            "--initial-branch", "main",
            self.repo,
        )

        git(
            "config",
            "user.email",
            "tester@example.com",
            cwd=self.repo,
        )

        git(
            "config",
            "user.name",
            "Tester",
            cwd=self.repo,
        )

        git(
            "remote", "add",
            "origin", self.remote,
            cwd=self.repo,
        )

        for name in ["a.txt", "b.txt"]:
            (self.repo / name).write_text(
                "initial\n",
                encoding="utf-8",
            )

        git(
            "add", "a.txt", "b.txt",
            cwd=self.repo,
        )

        git(
            "commit", "-qm", "initial",
            cwd=self.repo,
        )

        git(
            "push", "-q", "-u",
            "origin", "main",
            cwd=self.repo,
        )

    def test_preview_does_not_stage(self):

        (self.repo / "a.txt").write_text("changed")

        plan = preview(
            self.repo,
            paths=["a.txt"],
        )

        self.assertEqual(
            plan["files"],
            ["a.txt"],
        )

        staged = git(
            "diff",
            "--cached",
            "--name-only",
            cwd=self.repo,
        )

        self.assertEqual(staged, "")

    def test_commit_only_selected(self):

        (self.repo / "a.txt").write_text("changed")
        (self.repo / "b.txt").write_text("unselected")

        commit(
            self.repo,
            paths=["a.txt"],
            message="selected only",
        )

        committed_files = git(
            "show",
            "--pretty=format:",
            "--name-only",
            "HEAD",
            cwd=self.repo,
        )

        self.assertEqual(committed_files, "a.txt")

        self.assertIn(
            "b.txt",
            git(
                "status", "--porcelain",
                cwd=self.repo,
            ),
        )

    def test_reject_existing_staged_files(self):

        (self.repo / "b.txt").write_text("staged")

        git(
            "add", "b.txt",
            cwd=self.repo,
        )

        (self.repo / "a.txt").write_text("changed")

        with self.assertRaisesRegex(
            RuntimeError,
            "Staging area",
        ):
            preview(
                self.repo,
                paths=["a.txt"],
            )

    def test_reject_sensitive_files(self):

        (self.repo / ".env").write_text(
            "SECRET=private"
        )

        with self.assertRaises(ValueError):
            preview(
                self.repo,
                paths=[".env"],
            )

        (self.repo / "a.txt").write_text(
            "ghp_" + "A" * 40
        )

        with self.assertRaisesRegex(
            ValueError,
            "credential",
        ):
            preview(
                self.repo,
                paths=["a.txt"],
            )

    def test_push_requires_confirmation(self):

        with self.assertRaisesRegex(
            ValueError,
            "explicit",
        ):
            push(self.repo)

    def test_push_one_commit(self):

        (self.repo / "a.txt").write_text("changed")

        commit(
            self.repo,
            paths=["a.txt"],
            message="one commit",
        )

        result = push(
            self.repo,
            confirm=True,
        )

        self.assertEqual(
            result["branch"],
            "main",
        )

        self.assertEqual(
            git("rev-parse", "HEAD", cwd=self.repo),
            git(
                "rev-parse",
                "refs/remotes/origin/main",
                cwd=self.repo,
            ),
        )

    def test_reject_multiple_outgoing_commits(self):

        for name in ["v1", "v2"]:
            (self.repo / "a.txt").write_text(name)

            git(
                "add", "a.txt",
                cwd=self.repo,
            )

            git(
                "commit", "-qm", name,
                cwd=self.repo,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "exactly one",
        ):
            push(
                self.repo,
                confirm=True,
            )

    def test_reject_unchanged_file(self):

        with self.assertRaisesRegex(
            ValueError,
            "No changes",
        ):
            preview(
                self.repo,
                paths=["a.txt"],
            )


if __name__ == "__main__":
    unittest.main()
