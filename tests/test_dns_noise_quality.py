"""CPU-only unit tests for noise signal quality."""

import io
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.data.dns_noise_quality import measure_pcm16_wav


def make_wav(samples):
    stream = io.BytesIO()

    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(
            np.asarray(samples, dtype="<i2").tobytes()
        )

    return stream.getvalue()


class TestDNSNoiseQuality(unittest.TestCase):

    def test_normal_signal(self):
        samples = np.tile(
            np.array([5000, -5000], dtype=np.int16),
            8000,
        )

        result = measure_pcm16_wav(
            make_wav(samples)
        )

        self.assertEqual(result["sample_rate"], 16000)
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["quality_flags"], [])
        self.assertLess(result["rms_dbfs"], -10)

    def test_digital_silence(self):
        result = measure_pcm16_wav(
            make_wav(np.zeros(16000, dtype=np.int16))
        )

        self.assertIn(
            "digital_silence",
            result["quality_flags"],
        )
        self.assertIn(
            "mostly_quiet",
            result["quality_flags"],
        )

    def test_clipped_signal(self):
        samples = np.tile(
            np.array([-32768, 32767], dtype=np.int16),
            8000,
        )

        result = measure_pcm16_wav(
            make_wav(samples)
        )

        self.assertIn(
            "high_clipping_fraction",
            result["quality_flags"],
        )

    def test_truncated_wav(self):
        payload = make_wav(
            np.zeros(16000, dtype=np.int16)
        )

        with self.assertRaises(
            (ValueError, wave.Error, EOFError)
        ):
            measure_pcm16_wav(payload[:-100])


if __name__ == "__main__":
    unittest.main()
