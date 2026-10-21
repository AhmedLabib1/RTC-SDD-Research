"""Reproducible, source-aware DNS listening review.

Extracts selected WAVs only. Never approves training audio.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import wave
from collections import defaultdict
from pathlib import Path

MAX_WAV_BYTES = 32 * 1024 * 1024
SEED = "p03_dns_listening_v1"


def _ordered(rows):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (SEED + "|" + row["name"]).encode()
        ).hexdigest(),
    )


def select_samples(rows, max_samples=40):
    """Balance categories, review tiers and quality concerns."""

    rows = list(rows)

    selected = []
    seen_names = set()
    seen_sources = set()

    def add(row, reason):
        if len(selected) >= max_samples:
            return False

        if row["tier"] == "exclude_invalid":
            return False

        name = row["name"]
        source = row.get("source_id") or ("file:" + name)

        if name in seen_names or source in seen_sources:
            return False

        seen_names.add(name)
        seen_sources.add(source)

        selected.append({
            **row,
            "selection_reason": reason,
        })

        return True

    # Human-risk categories first, followed by usable
    # candidates and quality-review categories.
    budgets = [
        ("review_human", 2),
        ("candidate_listening", 2),
        ("review_quality", 1),
        ("review_provenance", 2),
    ]

    for tier, per_category in budgets:
        grouped = defaultdict(list)

        for row in rows:
            if row["tier"] == tier:
                grouped[row["category"]].append(row)

        for category in sorted(grouped):
            count = 0

            for row in _ordered(grouped[category]):
                if add(row, f"{tier}:{category}"):
                    count += 1

                if count >= per_category:
                    break

    # Explicitly include important quality failure modes.
    flags = [
        "high_clipping_fraction",
        "high_dc_offset",
        "very_low_rms",
        "mostly_quiet",
    ]

    quality_rows = [
        row for row in rows
        if row["tier"] == "review_quality"
    ]

    for flag in flags:
        count = 0

        for row in _ordered(quality_rows):
            current_flags = set(
                (row.get("quality_flags") or "").split(",")
            )

            if flag not in current_flags:
                continue

            if add(row, "quality_flag:" + flag):
                count += 1

            if count >= 2:
                break

    for index, row in enumerate(selected, 1):
        row["clip_id"] = f"clip_{index:03d}"
        row["reviewed_by_human"] = "no"
        row["speech_present"] = "unknown"
        row["listening_decision"] = "pending"
        row["notes"] = ""
        row["approved_for_training"] = "false"

    return selected


def prepare_listening_audit(
    screening_csv: Path,
    archive_path: Path,
    output_dir: Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(screening_csv).open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    selected = select_samples(rows)

    if not selected:
        raise RuntimeError("No review samples selected")

    review_csv = output_dir / "listening_review.csv"

    # Reruns must not destroy human annotations.
    if review_csv.exists():
        with review_csv.open(
            encoding="utf-8", newline=""
        ) as stream:
            existing = list(csv.DictReader(stream))

        if [r["name"] for r in existing] != [
            r["name"] for r in selected
        ]:
            raise RuntimeError(
                "Existing review manifest differs; "
                "refusing to overwrite annotations"
            )
    else:
        with review_csv.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(selected[0]),
            )
            writer.writeheader()
            writer.writerows(selected)

    chosen = {
        row["name"]: row for row in selected
    }

    found = set()

    # Stream the archive; never extract arbitrary paths.
    with tarfile.open(
        archive_path, mode="r|bz2"
    ) as archive:

        for member in archive:
            if member.name not in chosen:
                continue

            if not member.isfile():
                raise RuntimeError(
                    "Selected archive entry is not a file"
                )

            if (
                member.size > MAX_WAV_BYTES
                or member.size <= 44
            ):
                raise RuntimeError(
                    "Unexpected selected WAV size"
                )

            stream = archive.extractfile(member)

            if stream is None:
                raise RuntimeError(
                    "Selected WAV could not be opened"
                )

            with stream:
                payload = stream.read(
                    MAX_WAV_BYTES + 1
                )

            if len(payload) != member.size:
                raise RuntimeError(
                    "Selected WAV is incomplete"
                )

            with wave.open(
                io.BytesIO(payload), "rb"
            ) as wav:
                if (
                    wav.getsampwidth() != 2
                    or wav.getframerate() <= 0
                    or wav.getnframes() <= 0
                    or wav.getcomptype() != "NONE"
                ):
                    raise RuntimeError(
                        "Unsupported WAV format"
                    )

                expected_pcm_bytes = (
                    wav.getnframes()
                    * wav.getnchannels()
                    * wav.getsampwidth()
                )

                if len(wav.readframes(
                    wav.getnframes()
                )) != expected_pcm_bytes:
                    raise RuntimeError(
                        "Incomplete WAV audio frames"
                    )

            row = chosen[member.name]

            target = output_dir / (
                row["clip_id"] + ".wav"
            )

            if target.exists():
                if target.read_bytes() != payload:
                    raise RuntimeError(
                        "Existing review WAV differs"
                    )
            else:
                target.write_bytes(payload)

            found.add(member.name)

            if len(found) % 10 == 0:
                print(
                    f"[EXTRACT] {len(found)} samples",
                    flush=True,
                )

    missing = set(chosen) - found

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} selected WAVs"
        )

    summary = {
        "archive": Path(archive_path).name,
        "total_source_records": len(rows),
        "review_samples": len(selected),
        "selection_seed": SEED,
        "source_aware_sampling": True,
        "speech_review_completed": False,
        "approved_for_training": False,
        "review_csv": review_csv.name,
    }

    (output_dir / "listening_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return selected
