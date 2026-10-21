"""Run a CPU-only DNS noise metadata audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.data.dns_noise_audit import scan_archive


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    summary = scan_archive(
        args.archive,
        args.output_dir,
    )

    print("\n" + "=" * 68)
    print("DNS AUDIO METADATA SUMMARY")
    print("=" * 68)

    for key in [
        "files_scanned",
        "valid_metadata",
        "metadata_errors",
        "sample_rates",
        "channels",
        "sample_width_bytes",
        "unique_freesound_source_ids",
        "unmatched_name_patterns",
        "source_groups_with_multiple_segments",
        "total_audio_hours",
        "duration_min_s",
        "duration_median_s",
        "duration_max_s",
    ]:
        print(f"{key}: {summary[key]}")

    print("\nTop 20 noise categories:")

    for name, count in summary["top_categories"]:
        print(f"  {name:<28} {count}")

    if summary["metadata_errors"]:
        print("\n[REVIEW] Metadata errors:")
        for item in summary["errors_preview"]:
            print(" -", item)

    print("\n[PASS] Metadata report saved")
    print("[PENDING] Audio signal quality audit")
    print("[PENDING] Speech-content audit")
    print("[INFO] Not approved for training")


if __name__ == "__main__":
    main()
