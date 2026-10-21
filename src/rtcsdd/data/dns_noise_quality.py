"""CPU-only signal-quality inspection for DNS PCM16 WAV recordings.

Quality flags are review indicators, not automatic exclusions.
No speech detection is performed.
"""

from __future__ import annotations

import io
import json
import tarfile
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from rtcsdd.data.dns_noise_audit import FREESOUND_PATTERN

MAX_MEMBER_BYTES = 32 * 1024 * 1024
SILENCE_DBFS = -50.0
LOW_RMS_DBFS = -50.0
CLIPPING_FRACTION_LIMIT = 0.001
DC_OFFSET_LIMIT = 0.05


def measure_pcm16_wav(payload: bytes) -> dict:
    """Calculate signal statistics without resampling or augmentation."""

    with wave.open(io.BytesIO(payload), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()

        if (
            rate <= 0 or channels <= 0 or frames <= 0
            or width != 2 or wav.getcomptype() != "NONE"
        ):
            raise ValueError("Expected nonempty PCM16 WAV")

        pcm_bytes = wav.readframes(frames)

    expected_bytes = frames * channels * 2

    if len(pcm_bytes) != expected_bytes:
        raise ValueError("Truncated PCM payload")

    pcm = np.frombuffer(pcm_bytes, dtype="<i2")

    samples = pcm.astype(np.float32) / 32768.0
    mono = samples.reshape(-1, channels).mean(axis=1)

    rms = float(
        np.sqrt(np.mean(mono.astype(np.float64) ** 2))
    )

    rms_dbfs = (
        round(20 * np.log10(rms), 2)
        if rms > 0 else None
    )

    peak = float(np.max(np.abs(samples)))

    clipped_fraction = float(
        np.mean(np.abs(pcm.astype(np.int32)) >= 32767)
    )

    dc_offset = float(np.mean(mono))

    # 100 ms windows for identifying mostly quiet recordings.
    window_size = max(1, round(rate * 0.1))
    full_windows = len(mono) // window_size

    if full_windows:
        windowed = mono[
            :full_windows * window_size
        ].reshape(full_windows, window_size)

        window_rms = np.sqrt(
            np.mean(
                windowed.astype(np.float64) ** 2,
                axis=1,
            )
        )

        threshold = 10 ** (SILENCE_DBFS / 20)

        silent_ratio = float(
            np.mean(window_rms < threshold)
        )
    else:
        silent_ratio = float(
            rms < 10 ** (SILENCE_DBFS / 20)
        )

    flags = []

    if rms_dbfs is None:
        flags.append("digital_silence")
    elif rms_dbfs < LOW_RMS_DBFS:
        flags.append("very_low_rms")

    if clipped_fraction >= CLIPPING_FRACTION_LIMIT:
        flags.append("high_clipping_fraction")

    if abs(dc_offset) > DC_OFFSET_LIMIT:
        flags.append("high_dc_offset")

    if silent_ratio > 0.5:
        flags.append("mostly_quiet")

    return {
        "sample_rate": rate,
        "channels": channels,
        "duration_s": round(frames / rate, 4),
        "rms_dbfs": rms_dbfs,
        "peak": round(peak, 6),
        "clipped_fraction": round(clipped_fraction, 6),
        "dc_offset": round(dc_offset, 6),
        "silent_window_ratio": round(silent_ratio, 4),
        "quality_flags": flags,
    }


def audit_archive(archive_path: Path, output_dir: Path) -> dict:
    """Audit every WAV in a streaming archive without extracting it."""

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "dns_audio_quality.jsonl"
    summary_path = output_dir / "dns_quality_summary.json"
    temporary = output_dir / "dns_audio_quality.jsonl.part"

    total = 0
    valid = 0
    flag_counts = Counter()
    categories = Counter()
    errors = []
    unmatched = []
    rms_values = []

    with (
        tarfile.open(archive_path, "r|bz2") as archive,
        temporary.open("w", encoding="utf-8") as output,
    ):
        for member in archive:
            if not member.isfile():
                continue

            total += 1
            name = member.name

            record = {
                "name": name,
                "valid": False,
                "quality_flags": [],
            }

            try:
                parts = name.replace("\\", "/").split("/")

                if (
                    name.startswith(("/", "\\"))
                    or "\\" in name
                    or ".." in parts
                    or ":" in name[:3]
                ):
                    raise ValueError("Unsafe archive path")

                if not name.lower().endswith(".wav"):
                    raise ValueError("Unexpected non-WAV file")

                if member.size > MAX_MEMBER_BYTES:
                    raise ValueError("WAV exceeds audit size limit")

                stream = archive.extractfile(member)

                if stream is None:
                    raise ValueError("Unreadable TAR member")

                with stream:
                    payload = stream.read(
                        MAX_MEMBER_BYTES + 1
                    )

                if len(payload) != member.size:
                    raise ValueError("Truncated TAR member")

                metrics = measure_pcm16_wav(payload)

                record.update(metrics)
                record["valid"] = True
                valid += 1

                if metrics["rms_dbfs"] is not None:
                    rms_values.append(metrics["rms_dbfs"])

                for flag in metrics["quality_flags"]:
                    flag_counts[flag] += 1

                match = FREESOUND_PATTERN.fullmatch(
                    Path(name).name
                )

                if match:
                    category = match.group("category")
                    record["category"] = category
                    record["source_id"] = match.group(
                        "source_id"
                    )
                    categories[category] += 1

                elif len(unmatched) < 56:
                    unmatched.append(name)

            except (ValueError, wave.Error, EOFError) as exc:
                record["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append({
                    "name": name,
                    "error": record["error"],
                })

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            if total % 500 == 0:
                print(
                    f"[SCAN] {total} / {valid} valid",
                    flush=True,
                )

    temporary.replace(records_path)

    summary = {
        "archive": archive_path.name,
        "files_scanned": total,
        "valid_signal_reads": valid,
        "signal_read_errors": len(errors),
        "quality_flags": dict(flag_counts),
        "median_rms_dbfs": (
            round(float(np.median(rms_values)), 2)
            if rms_values else None
        ),
        "min_rms_dbfs": (
            min(rms_values) if rms_values else None
        ),
        "max_rms_dbfs": (
            max(rms_values) if rms_values else None
        ),
        "categories": dict(categories),
        "unmatched_names_preview": unmatched[:30],
        "errors_preview": errors[:20],
        "quality_audit": "completed",
        "speech_audit": "pending",
        "approved_for_training": False,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary
