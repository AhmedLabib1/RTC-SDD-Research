"""Preview or explicitly commit/push selected source files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from rtcsdd.ops.git_workflow import (
    commit,
    preview,
    push,
)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
    )

    parser.add_argument(
        "--branch",
        default="main",
    )

    parser.add_argument(
        "--remote",
        default="origin",
    )

    parser.add_argument("--message")
    parser.add_argument(
        "--commit",
        action="store_true",
    )

    parser.add_argument(
        "--push",
        action="store_true",
    )

    parser.add_argument(
        "--secret",
        default=None,
    )

    args = parser.parse_args()

    if args.push and not args.commit:
        parser.error(
            "--push requires --commit"
        )

    if args.commit and not args.message:
        parser.error(
            "--commit requires --message"
        )

    plan = preview(
        args.repo,
        paths=args.files,
        branch=args.branch,
        remote=args.remote,
    )

    print("=" * 65)
    print("GENERIC GIT WORKFLOW")
    print("=" * 65)

    print("Repository:", plan["repo"])
    print("Branch    :", plan["branch"])
    print("Remote    :", plan["remote"])

    print("\nSelected files:")

    for filename in plan["files"]:
        print(" +", filename)

    print(
        "\nTotal selected size:",
        plan["size_bytes"],
        "bytes",
    )

    print(
        "Existing outgoing commits:",
        plan["outgoing_commits_before_commit"],
    )

    if not args.commit:
        print("\n[PREVIEW ONLY]")
        print("[PASS] Nothing staged")
        print("[PASS] Nothing committed")
        print("[PASS] Nothing pushed")
        return

    result = commit(
        args.repo,
        paths=args.files,
        message=args.message,
        branch=args.branch,
        remote=args.remote,
    )

    print("\n[COMMIT CREATED]", result["commit"])

    if args.push:
        pushed = push(
            args.repo,
            branch=args.branch,
            remote=args.remote,
            secret_name=args.secret,
            confirm=True,
        )

        print(
            "[PUSH COMPLETE]",
            pushed["commit"],
        )
    else:
        print("[INFO] Push not requested")


if __name__ == "__main__":
    main()
