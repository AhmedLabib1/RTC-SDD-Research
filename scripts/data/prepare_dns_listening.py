"""Prepare source-aware WAV samples for human listening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.data.dns_noise_listening import (
    prepare_listening_audit,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = prepare_listening_audit(
        args.screening,
        args.archive,
        args.output_dir,
    )

    print("\n" + "=" * 68)
    print("STRATIFIED LISTENING AUDIT")
    print("=" * 68)

    print("Review clips:", len(selected))
    print("Unique selected source groups:", len(selected))

    for row in selected:
        print(
            f"{row['clip_id']}: "
            f"{row['tier']} | "
            f"{row['category']} | "
            f"{row['selection_reason']}"
        )

    print("\n[PENDING] Human listening review")
    print("[INFO] Zero clips approved for training")


if __name__ == "__main__":
    main()
