"""Offline tests for pinned requirement conversion and planning."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.ops.training_requirements import (
    FAIRSEQ_SHA,
    convert_requirements,
    install_commands,
)


def fixture():
    return "\n".join([
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        (
            "-e git+https://github.com/pytorch/fairseq@"
            + FAIRSEQ_SHA
            + "#egg=fairseq"
        ),
        "numpy==1.26.4",
        "",
    ])


class TestTrainingRequirements(unittest.TestCase):

    def test_convert_pinned_fairseq(self):
        converted = convert_requirements(fixture())

        self.assertIn(
            "fairseq @ git+https://github.com/"
            "pytorch/fairseq.git@" + FAIRSEQ_SHA,
            converted,
        )

        self.assertNotIn("-e git+", converted)

    def test_preserves_other_pins(self):
        converted = convert_requirements(fixture())

        self.assertIn(
            "torch==2.5.1+cu121", converted
        )

        self.assertIn(
            "numpy==1.26.4", converted
        )

    def test_reject_wrong_fairseq_sha(self):
        source = fixture().replace(
            FAIRSEQ_SHA, "0" * 40
        )

        with self.assertRaises(ValueError):
            convert_requirements(source)

    def test_reject_missing_torch_pin(self):
        source = fixture().replace(
            "torch==2.5.1+cu121",
            "torch==2.6.0",
        )

        with self.assertRaises(ValueError):
            convert_requirements(source)

    def test_reject_missing_fairseq(self):
        source = fixture().replace(
            next(
                line for line in fixture().splitlines()
                if line.startswith("-e git+")
            ),
            "",
        )

        with self.assertRaises(ValueError):
            convert_requirements(source)

    def test_install_plan_is_explicit(self):
        plan = install_commands(
            python=Path("/tmp/venv/bin/python"),
            requirements=Path("/tmp/requirements.txt"),
            project=Path("/tmp/project"),
        )

        self.assertEqual(len(plan), 3)

        self.assertIn(
            "unsafe-first-match", plan[0]
        )

        self.assertIn(
            "-e", plan[1]
        )

        self.assertEqual(
            plan[2][1:3],
            ["pip", "check"],
        )


if __name__ == "__main__":
    unittest.main()
