"""CPU-only DNS candidate-screening tests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.data.dns_noise_screen import (
    classify,
    identify_source,
)


class TestDNSNoiseScreen(unittest.TestCase):

    def record(self, name, flags=None):
        return {
            "name": name,
            "valid": True,
            "quality_flags": flags or [],
            "duration_s": 10,
            "rms_dbfs": -30,
        }

    def test_freesound_source(self):
        category, source = identify_source(
            "noise/fan_Freesound_validated_123_2.wav"
        )
        self.assertEqual(category, "fan")
        self.assertEqual(source, "123")

    def test_extended_filename(self):
        category, source = identify_source(
            "munching_Freesound_validated_153524"
            "__someone__eating_2.wav"
        )
        self.assertEqual(category, "munching")
        self.assertEqual(source, "153524")

    def test_normal_candidate_needs_review(self):
        result = classify(
            self.record(
                "fan_Freesound_validated_123_0.wav"
            )
        )
        self.assertEqual(
            result["tier"],
            "candidate_listening",
        )
        self.assertFalse(result["approved_for_training"])

    def test_silence_is_excluded(self):
        result = classify(
            self.record(
                "fan_Freesound_validated_123_0.wav",
                ["digital_silence"],
            )
        )
        self.assertEqual(
            result["tier"],
            "exclude_invalid",
        )

    def test_human_risk(self):
        result = classify(
            self.record(
                "munching_Freesound_validated_555_0.wav"
            )
        )
        self.assertEqual(
            result["tier"],
            "review_human",
        )

    def test_quality_review(self):
        result = classify(
            self.record(
                "door_Freesound_validated_123_0.wav",
                ["mostly_quiet"],
            )
        )
        self.assertEqual(
            result["tier"],
            "review_quality",
        )

    def test_unknown_source_review(self):
        result = classify(
            self.record(
                "unknown_noise_0.wav"
            )
        )
        self.assertEqual(
            result["tier"],
            "review_provenance",
        )


if __name__ == "__main__":
    unittest.main()
