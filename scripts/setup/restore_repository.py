"""Restore or verify a Git repository without overwriting changes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.repository import restore_repository


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--secret", default=None)
    parser.add_argument(
        "--username",
        default="x-access-token",
    )

    args = parser.parse_args()

    info = restore_repository(
        url=args.url,
        destination=args.destination,
        branch=args.branch,
        secret_name=args.secret,
        username=args.username,
    )

    print("[PASS] REPOSITORY VERIFIED")
    print(
        "Action:",
        "CLONED" if info["cloned"]
        else "EXISTING — UNCHANGED",
    )
    print("Branch:", info["branch"])
    print("Commit:", info["commit"])
    print(
        "Working tree:",
        "CLEAN" if info["clean"]
        else "HAS LOCAL CHANGES",
    )
    print("Path:", info["destination"])


if __name__ == "__main__":
    main()
