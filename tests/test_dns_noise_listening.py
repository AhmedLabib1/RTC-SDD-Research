"""Offline tests for DNS listening sample selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.data.dns_noise_listening import (
    select_samples,
)


class TestDNSListening(unittest.TestCase):

    def test_deterministic_selection(self):
        rows = [
            {
                "name": f"fan_{i}.wav",
                "source_id": str(i),
                "category": "fan",
                "tier": "candidate_listening",
                "quality_flags": "",
            }
            for i in range(8)
        ]

        first = select_samples(rows)
        second = select_samples(rows)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_invalid_record_excluded(self):
        rows = [{
            "name": "silent.wav",
            "source_id": "123",
            "category": "fan",
            "tier": "exclude_invalid",
            "quality_flags": "digital_silence",
        }]

        self.assertEqual(
            select_samples(rows), []
        )

    def test_source_deduplication(self):
        rows = [
            {
                "name": f"fan_{i}.wav",
                "source_id": "same_source",
                "category": "fan",
                "tier": "candidate_listening",
                "quality_flags": "",
            }
            for i in range(4)
        ]

        selected = select_samples(rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["approved_for_training"],
            "false",
        )


if __name__ == "__main__":
    unittest.main()
