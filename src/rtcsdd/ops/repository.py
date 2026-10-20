"""Conservative, reusable Git repository restoration.

Never implicitly fetch, pull, reset, stage, commit, or push.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from .git_auth import git_environment


def _git(
    *args: str,
    cwd: Path | None = None,
    env: dict | None = None,
) -> str:

    command = [
        "git",
        *(["-C", str(cwd)] if cwd else []),
        *args,
    ]

    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
    )

    if result.returncode:
        # Git stderr can include credential-bearing URLs.
        raise RuntimeError(
            f"Git command failed "
            f"({args[0]}, exit={result.returncode}); "
            "details suppressed"
        )

    return result.stdout.strip()


def _canonical_remote(url: str) -> str:

    if "://" in url:
        parts = urlsplit(url)

        hostname = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""

        path = parts.path.rstrip("/")

        if path.endswith(".git"):
            path = path[:-4]

        return (
            f"{parts.scheme.lower()}://"
            f"{hostname.lower()}{port}{path}"
        )

    candidate = Path(url).expanduser()

    if candidate.exists():
        return str(candidate.resolve())

    return url.rstrip("/").removesuffix(".git")


def restore_repository(
    *,
    url: str,
    destination: Path,
    branch: str = "main",
    secret_name: str | None = None,
    username: str = "x-access-token",
) -> dict:
    """Clone if missing; otherwise verify without changing files."""

    if not url.strip() or not branch.strip():
        raise ValueError(
            "url and branch must be nonempty"
        )

    destination = (
        Path(destination).expanduser().absolute()
    )

    if destination.exists():

        if not (destination / ".git").is_dir():
            raise ValueError(
                "Destination exists but is not a Git working tree"
            )

        origin = _git(
            "remote",
            "get-url",
            "origin",
            cwd=destination,
        )

        if _canonical_remote(origin) != _canonical_remote(url):
            raise ValueError(
                "Existing repository origin does not match requested URL"
            )

        current_branch = _git(
            "branch",
            "--show-current",
            cwd=destination,
        )

        if current_branch != branch:
            raise ValueError(
                "Existing repository is not on the requested branch"
            )

        cloned = False

    else:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with git_environment(
            secret_name=secret_name,
            username=username,
        ) as env:

            _git(
                "-c",
                "credential.helper=",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                "--",
                url,
                str(destination),
                env=env,
            )

        cloned = True

    return {
        "destination": str(destination),
        "cloned": cloned,
        "branch": _git(
            "branch",
            "--show-current",
            cwd=destination,
        ),
        "commit": _git(
            "rev-parse",
            "--short",
            "HEAD",
            cwd=destination,
        ),
        "clean": not bool(
            _git(
                "status",
                "--porcelain",
                cwd=destination,
            )
        ),
    }
