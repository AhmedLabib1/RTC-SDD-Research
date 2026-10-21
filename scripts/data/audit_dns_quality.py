"""Run signal-level DNS archive audit on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.data.dns_noise_quality import audit_archive


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive", type=Path, required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )

    args = parser.parse_args()

    result = audit_archive(
        args.archive, args.output_dir
    )

    print("\n" + "=" * 68)
    print("DNS SIGNAL QUALITY SUMMARY")
    print("=" * 68)

    for key in (
        "files_scanned",
        "valid_signal_reads",
        "signal_read_errors",
        "median_rms_dbfs",
        "min_rms_dbfs",
        "max_rms_dbfs",
    ):
        print(f"{key}: {result[key]}")

    print("\nQuality review flags:")

    if result["quality_flags"]:
        for flag, count in sorted(
            result["quality_flags"].items()
        ):
            print(f"  {flag:<30} {count}")
    else:
        print("  No threshold flags detected")

    print("\nUnmatched filename examples:")

    for name in result["unmatched_names_preview"][:12]:
        print(" -", name)

    if result["signal_read_errors"]:
        print("\n[REVIEW] Audio decoding errors:")
        for item in result["errors_preview"]:
            print(" -", item)

    print("\n[PASS] Signal audit results saved")
    print("[PENDING] Speech-content review")
    print("[INFO] No recordings automatically approved")


if __name__ == "__main__":
    main()
