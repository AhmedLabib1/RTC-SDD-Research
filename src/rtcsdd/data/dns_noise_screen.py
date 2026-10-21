"""Conservative DNS noise candidate screening.

Uses prior signal-quality reports without loading audio.
Never automatically approves recordings for training.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

HUMAN_TERMS = (
    "speech", "voice", "talk", "whisper",
    "sing", "laugh", "cough", "sneeze",
    "breath", "spit", "munch", "eating",
    "chew", "crying", "snoring",
)

QUALITY_FLAGS = {
    "mostly_quiet",
    "very_low_rms",
    "high_clipping_fraction",
    "high_dc_offset",
}


def identify_source(name: str) -> tuple[str, str | None]:
    """Extract category and Freesound ID where identifiable."""

    filename = Path(name).name
    marker = "_Freesound_validated_"

    if marker not in filename:
        return "unknown", None

    category, remainder = filename.split(marker, 1)

    match = re.match(r"(\d+)(?:_|$)", remainder)

    source_id = match.group(1) if match else None

    return category.lower(), source_id


def classify(record: dict) -> dict:
    """Assign a review tier, never a training approval."""

    name = record["name"]
    flags = set(record.get("quality_flags") or [])

    category, source_id = identify_source(name)

    lower = Path(name).name.lower()

    human_risk = any(
        term in lower for term in HUMAN_TERMS
    )

    if not record.get("valid", False):
        tier = "exclude_invalid"
        reason = "signal_read_failed"

    elif "digital_silence" in flags:
        tier = "exclude_invalid"
        reason = "digital_silence"

    elif human_risk:
        tier = "review_human"
        reason = "possible_human_sound"

    elif source_id is None:
        tier = "review_provenance"
        reason = "source_id_unresolved"

    elif flags & QUALITY_FLAGS:
        tier = "review_quality"
        reason = ",".join(sorted(flags & QUALITY_FLAGS))

    else:
        tier = "candidate_listening"
        reason = "requires_content_review"

    return {
        "name": name,
        "category": category,
        "source_id": source_id,
        "tier": tier,
        "reason": reason,
        "duration_s": record.get("duration_s"),
        "rms_dbfs": record.get("rms_dbfs"),
        "silent_window_ratio": record.get(
            "silent_window_ratio"
        ),
        "quality_flags": ",".join(sorted(flags)),
        "approved_for_training": False,
    }


def screen_report(
    quality_jsonl: Path,
    output_dir: Path,
) -> dict:
    """Generate an auditable CSV and summary JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dns_screening.csv"
    summary_path = output_dir / "dns_screening_summary.json"

    tiers = Counter()
    categories = Counter()
    sources = set()

    seen = set()
    total = 0

    columns = [
        "name",
        "category",
        "source_id",
        "tier",
        "reason",
        "duration_s",
        "rms_dbfs",
        "silent_window_ratio",
        "quality_flags",
        "approved_for_training",
    ]

    temporary = csv_path.with_suffix(".csv.part")

    try:
        with (
            Path(quality_jsonl).open(
                encoding="utf-8"
            ) as input_stream,
            temporary.open(
                "w", encoding="utf-8", newline=""
            ) as output_stream,
        ):
            writer = csv.DictWriter(
                output_stream,
                fieldnames=columns,
            )
            writer.writeheader()

            for line in input_stream:
                if not line.strip():
                    continue

                record = json.loads(line)
                result = classify(record)

                name = result["name"]

                if name in seen:
                    raise ValueError(
                        f"Duplicate recording: {name}"
                    )

                seen.add(name)
                total += 1

                tiers[result["tier"]] += 1
                categories[result["category"]] += 1

                if result["source_id"] is not None:
                    sources.add(result["source_id"])

                writer.writerow(result)

        temporary.replace(csv_path)

    finally:
        if temporary.exists():
            temporary.unlink()

    summary = {
        "total_recordings": total,
        "tiers": dict(tiers),
        "unique_identified_source_ids": len(sources),
        "top_categories": categories.most_common(20),
        "screening_method": "metadata_and_filename_heuristics",
        "speech_recognition_performed": False,
        "human_listening_completed": False,
        "approved_for_training": False,
        "records_file": csv_path.name,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary
