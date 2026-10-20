"""Offline tests for generic repository restoration."""

import subprocess
import sys
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.ops.repository import restore_repository


def git(*args, cwd=None):

    command = [
        "git",
        *(["-C", str(cwd)] if cwd else []),
        *map(str, args),
    ]

    return subprocess.check_output(
        command,
        text=True,
    ).strip()


class TestRepositoryOps(unittest.TestCase):

    def setUp(self):

        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        root = Path(self.temp.name)

        self.source = root / "origin"
        self.destination = root / "checked_out"

        git(
            "init",
            "-q",
            "--initial-branch",
            "main",
            self.source,
        )

        git(
            "config",
            "user.email",
            "test@example.test",
            cwd=self.source,
        )

        git(
            "config",
            "user.name",
            "Test",
            cwd=self.source,
        )

        (self.source / "README.md").write_text(
            "initial",
            encoding="utf-8",
        )

        git("add", "README.md", cwd=self.source)
        git("commit", "-qm", "initial", cwd=self.source)

    def test_restore_and_idempotence(self):

        first = restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        self.assertTrue(first["cloned"])
        self.assertTrue(first["clean"])

        second = restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        self.assertFalse(second["cloned"])

        self.assertEqual(
            first["commit"],
            second["commit"],
        )

    def test_preserve_uncommitted_changes(self):

        restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        (self.destination / "README.md").write_text(
            "user edit",
            encoding="utf-8",
        )

        second = restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        self.assertFalse(second["clean"])
        self.assertEqual(
            (self.destination / "README.md").read_text(),
            "user edit",
        )

    def test_reject_different_remote(self):

        restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        with self.assertRaisesRegex(
            ValueError, "origin"
        ):
            restore_repository(
                url=str(self.source.parent / "other"),
                destination=self.destination,
            )

    def test_reject_different_branch(self):

        restore_repository(
            url=str(self.source),
            destination=self.destination,
        )

        with self.assertRaisesRegex(
            ValueError, "branch"
        ):
            restore_repository(
                url=str(self.source),
                destination=self.destination,
                branch="dev",
            )

    def test_reject_existing_nonrepo_directory(self):

        self.destination.mkdir()

        with self.assertRaises(ValueError):
            restore_repository(
                url=str(self.source),
                destination=self.destination,
            )


if __name__ == "__main__":
    unittest.main()
