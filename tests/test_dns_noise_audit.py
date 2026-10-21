"""CPU-only smoke test for DNS archive metadata audit."""

import io
import sys
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from rtcsdd.data.dns_noise_audit import scan_archive


class TestDNSNoiseAudit(unittest.TestCase):

    def test_multiple_segments_from_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "test.tar.bz2"

            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 16000)

            content = audio_buffer.getvalue()

            with tarfile.open(archive, "w:bz2") as tar:
                for index in range(2):
                    name = (
                        "noise/fan_Freesound_validated_123_"
                        f"{index}.wav"
                    )

                    info = tarfile.TarInfo(name)
                    info.size = len(content)

                    tar.addfile(info, io.BytesIO(content))

            summary = scan_archive(
                archive,
                root / "audit",
            )

            self.assertEqual(
                summary["files_scanned"], 2
            )
            self.assertEqual(
                summary["valid_metadata"], 2
            )
            self.assertEqual(
                summary["metadata_errors"], 0
            )
            self.assertEqual(
                summary["sample_rates"]["16000"], 2
            )
            self.assertEqual(
                summary["unique_freesound_source_ids"], 1
            )
            self.assertEqual(
                summary["source_groups_with_multiple_segments"], 1
            )


if __name__ == "__main__":
    unittest.main()
