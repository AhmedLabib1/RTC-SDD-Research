"""CPU-only, non-extracting metadata audit for DNS WAV archives."""

from __future__ import annotations

import io
import json
import re
import statistics
import tarfile
import wave
from collections import Counter
from pathlib import Path

MAX_MEMBER_BYTES = 32 * 1024 * 1024

FREESOUND_PATTERN = re.compile(
    r"(?P<category>.+)_Freesound_validated_"
    r"(?P<source_id>\d+)_(?P<segment>\d+)\.wav$",
    re.IGNORECASE,
)


def scan_archive(archive_path: Path, output_dir: Path) -> dict:

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "dns_audio_metadata.jsonl"
    summary_path = output_dir / "dns_audio_summary.json"

    rates = Counter()
    channels = Counter()
    widths = Counter()
    categories = Counter()
    sources = Counter()

    durations = []
    errors = []
    total = 0
    valid = 0
    unmatched_names = 0

    temp_records = records_path.with_suffix(".jsonl.part")

    with (
        tarfile.open(archive_path, mode="r|bz2") as archive,
        temp_records.open("w", encoding="utf-8") as output,
    ):
        for member in archive:

            if not member.isfile():
                continue

            total += 1
            name = member.name

            record = {
                "name": name,
                "size_bytes": member.size,
                "valid": False,
            }

            parts = name.replace("\\", "/").split("/")

            if (
                name.startswith(("/", "\\"))
                or "\\" in name
                or ".." in parts
                or ":" in name[:3]
            ):
                record["error"] = "unsafe_path"

            elif not name.lower().endswith(".wav"):
                record["error"] = "not_wav"

            elif member.size > MAX_MEMBER_BYTES:
                record["error"] = "member_too_large"

            else:
                stream = archive.extractfile(member)

                if stream is None:
                    record["error"] = "unreadable_member"
                else:
                    # Hold only one WAV in memory at a time.
                    with stream:
                        payload = stream.read(
                            MAX_MEMBER_BYTES + 1
                        )

                    try:
                        with wave.open(
                            io.BytesIO(payload), "rb"
                        ) as audio:

                            rate = audio.getframerate()
                            channel_count = audio.getnchannels()
                            width = audio.getsampwidth()
                            frames = audio.getnframes()
                            compression = audio.getcomptype()

                        if (
                            rate <= 0
                            or channel_count <= 0
                            or width <= 0
                            or frames <= 0
                            or compression != "NONE"
                        ):
                            raise ValueError(
                                "Invalid PCM audio metadata"
                            )

                        duration = frames / rate

                        record.update({
                            "valid": True,
                            "sample_rate": rate,
                            "channels": channel_count,
                            "sample_width_bytes": width,
                            "frames": frames,
                            "duration_s": round(duration, 6),
                        })

                        valid += 1
                        rates[str(rate)] += 1
                        channels[str(channel_count)] += 1
                        widths[str(width)] += 1
                        durations.append(duration)

                        match = FREESOUND_PATTERN.fullmatch(
                            Path(name).name
                        )

                        if match:
                            category = match.group("category")
                            source_id = match.group("source_id")

                            record["category"] = category
                            record["source_id"] = source_id
                            record["segment"] = int(
                                match.group("segment")
                            )

                            categories[category] += 1
                            sources[source_id] += 1
                        else:
                            unmatched_names += 1

                    except (
                        wave.Error,
                        EOFError,
                        ValueError,
                    ) as exc:
                        record["error"] = (
                            "wav_metadata_error:"
                            + type(exc).__name__
                        )

            if not record["valid"]:
                errors.append({
                    "name": name,
                    "error": record.get("error", "unknown"),
                })

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            if total % 500 == 0:
                print(
                    f"[SCAN] {total} files | "
                    f"{valid} valid",
                    flush=True,
                )

    temp_records.replace(records_path)

    summary = {
        "archive": archive_path.name,
        "files_scanned": total,
        "valid_metadata": valid,
        "metadata_errors": len(errors),
        "sample_rates": dict(rates),
        "channels": dict(channels),
        "sample_width_bytes": dict(widths),
        "unique_freesound_source_ids": len(sources),
        "unmatched_name_patterns": unmatched_names,
        "categories": dict(categories),
        "top_categories": categories.most_common(20),
        "source_groups_with_multiple_segments": sum(
            count > 1 for count in sources.values()
        ),
        "total_audio_hours": (
            round(sum(durations) / 3600, 3)
            if durations else 0
        ),
        "duration_min_s": (
            round(min(durations), 3)
            if durations else None
        ),
        "duration_median_s": (
            round(statistics.median(durations), 3)
            if durations else None
        ),
        "duration_max_s": (
            round(max(durations), 3)
            if durations else None
        ),
        "errors_preview": errors[:20],
        "speech_audit": "pending",
        "approved_for_training": False,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary
