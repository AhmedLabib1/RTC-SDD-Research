"""CPU-only tests for the reusable runtime preflight."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.ops.preflight import (
    load_profile,
    safe_relative,
    secret_available,
)


class TestRuntimePreflight(unittest.TestCase):

    def test_safe_relative_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.txt").write_text("test")

            path = safe_relative(root, "example.txt")

            self.assertEqual(
                path,
                (root / "example.txt").resolve(),
            )

    def test_path_traversal_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_relative(
                    Path(directory),
                    "../secret.txt",
                )

    def test_absolute_path_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_relative(
                    Path(directory),
                    "/tmp/secret.txt",
                )

    def test_profile_load(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"min_free_gib": 2}')

            profile = load_profile(path)

            self.assertEqual(profile["min_free_gib"], 2)

    def test_secret_from_environment(self):
        with patch.dict(
            "os.environ",
            {"TEST_RUNTIME_SECRET": "present"},
        ):
            self.assertTrue(
                secret_available("TEST_RUNTIME_SECRET")
            )


if __name__ == "__main__":
    unittest.main()
