"""Screen DNS noise quality reports without reading audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.data.dns_noise_screen import screen_report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quality-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    summary = screen_report(
        args.quality_jsonl,
        args.output_dir,
    )

    print("=" * 68)
    print("DNS NOISE SCREENING SUMMARY")
    print("=" * 68)

    print(
        "Total recordings:",
        summary["total_recordings"],
    )

    print(
        "Identified original sources:",
        summary["unique_identified_source_ids"],
    )

    print("\nReview tiers:")

    for tier, count in sorted(
        summary["tiers"].items()
    ):
        print(f"  {tier:<25} {count}")

    print("\nTop categories:")

    for category, count in summary["top_categories"]:
        print(f"  {category:<25} {count}")

    print("\n[PASS] Screening CSV generated")
    print("[PASS] Summary JSON generated")
    print("[PENDING] Human listening and speech review")
    print("[INFO] Zero recordings automatically approved")


if __name__ == "__main__":
    main()
