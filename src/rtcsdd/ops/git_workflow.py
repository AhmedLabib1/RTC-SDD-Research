"""Selective Git workflow: preview, commit, and guarded push."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

from .git_auth import git_environment


MAX_FILE_BYTES = 5 * 1024 * 1024

BLOCKED_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx",
    ".pt", ".pth", ".ckpt", ".pack",
    ".wav", ".zip", ".tar", ".gz",
    ".sqlite", ".db",
}

BLOCKED_PARTS = {
    ".git", ".ssh", ".kaggle",
    "checkpoints", "secrets",
}

SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"hf_[A-Za-z0-9]{32,}"),
    re.compile(
        r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"
    ),
)


def _git(
    repo: Path,
    *args: str,
    env: dict | None = None,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode:
        raise RuntimeError(
            f"Git {args[0]} failed "
            f"(exit={result.returncode}); "
            "command output suppressed"
        )

    return result.stdout.strip()


def _ref_exists(repo: Path, ref: str) -> bool:
    return subprocess.run(
        [
            "git", "-C", str(repo),
            "show-ref", "--verify", "--quiet", ref,
        ],
        capture_output=True,
    ).returncode == 0


def _verify_repo(
    repo: Path,
    branch: str,
    remote: str,
) -> Path:

    repo = Path(repo).expanduser().resolve()

    root = _git(
        repo, "rev-parse", "--show-toplevel"
    )

    if root != str(repo):
        raise ValueError(
            "Specify the Git repository root"
        )

    current = _git(
        repo, "branch", "--show-current"
    )

    if current != branch:
        raise ValueError(
            "Current branch does not match "
            "the requested branch"
        )

    if (
        not branch
        or branch.startswith("-")
        or not remote
        or remote.startswith("-")
    ):
        raise ValueError("Invalid branch or remote")

    _git(repo, "check-ref-format", "--branch", branch)

    remotes = _git(repo, "remote").splitlines()

    if remote not in remotes:
        raise ValueError(
            "Requested remote does not exist"
        )

    return repo


def _validate_files(
    repo: Path,
    paths: list[str],
) -> list[str]:

    if not paths:
        raise ValueError(
            "Select at least one file"
        )

    selected = []

    for raw in paths:

        if (
            not isinstance(raw, str)
            or not raw
            or "\x00" in raw
        ):
            raise ValueError("Invalid filename")

        normalized = raw.replace("\\", "/")
        relative = PurePosixPath(normalized)

        if (
            relative.is_absolute()
            or normalized.startswith("~")
            or ":" in normalized
            or any(
                part in {"", ".", ".."}
                for part in normalized.split("/")
            )
        ):
            raise ValueError(
                f"Unsafe path: {raw}"
            )

        if any(
            part.lower() in BLOCKED_PARTS
            for part in relative.parts
        ):
            raise ValueError(
                f"Blocked path: {raw}"
            )

        target = repo.joinpath(*relative.parts)

        if target.is_symlink() or not target.is_file():
            raise ValueError(
                f"Expected regular file: {raw}"
            )

        if not target.resolve().is_relative_to(repo):
            raise ValueError(
                "Selected file escapes repository"
            )

        if (
            target.name.lower().startswith(".env")
            or target.suffix.lower() in BLOCKED_SUFFIXES
        ):
            raise ValueError(
                f"Blocked sensitive/artifact file: {raw}"
            )

        if target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds source-size limit: {raw}"
            )

        try:
            content = target.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Only UTF-8 text files allowed: {raw}"
            ) from exc

        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                raise ValueError(
                    f"Potential credential in file: {raw}"
                )

        name = relative.as_posix()

        if name in selected:
            raise ValueError(
                f"Duplicate selection: {name}"
            )

        changed = _git(
            repo,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            name,
        )

        if not changed:
            raise ValueError(
                f"No changes detected: {name}"
            )

        selected.append(name)

    return selected


def preview(
    repo: Path,
    *,
    paths: list[str],
    branch: str = "main",
    remote: str = "origin",
) -> dict:

    repo = _verify_repo(repo, branch, remote)

    staged = _git(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "-z",
    )

    if staged:
        raise RuntimeError(
            "Staging area is not empty. "
            "Resolve existing staged changes first."
        )

    selected = _validate_files(repo, paths)

    tracking = (
        f"refs/remotes/{remote}/{branch}"
    )

    ahead = None

    if _ref_exists(repo, tracking):
        ahead = int(
            _git(
                repo,
                "rev-list",
                "--count",
                f"{tracking}..HEAD",
            )
        )

    return {
        "repo": str(repo),
        "branch": branch,
        "remote": remote,
        "files": selected,
        "size_bytes": sum(
            (repo / name).stat().st_size
            for name in selected
        ),
        "outgoing_commits_before_commit": ahead,
    }


def commit(
    repo: Path,
    *,
    paths: list[str],
    message: str,
    branch: str = "main",
    remote: str = "origin",
) -> dict:

    if (
        not message.strip()
        or "\n" in message
        or "\r" in message
    ):
        raise ValueError(
            "A one-line commit message is required"
        )

    plan = preview(
        repo,
        paths=paths,
        branch=branch,
        remote=remote,
    )

    root = Path(plan["repo"])
    selected = plan["files"]

    try:
        _git(
            root,
            "add",
            "--",
            *selected,
        )

        staged = set(
            _git(
                root,
                "diff",
                "--cached",
                "--name-only",
                "-z",
            ).split("\x00")
        ) - {""}

        if staged != set(selected):
            raise RuntimeError(
                "Staged files differ from selection"
            )

        _git(
            root,
            "diff",
            "--cached",
            "--check",
        )

        _git(
            root,
            "commit",
            "-m",
            message,
        )

    except Exception:
        # No staged files existed before this function.
        # Undo staging, without discarding working files.
        _git(
            root,
            "restore",
            "--staged",
            "--",
            *selected,
        )
        raise

    return {
        "commit": _git(
            root,
            "rev-parse",
            "--short",
            "HEAD",
        ),
        "files": selected,
        "branch": branch,
    }


def push(
    repo: Path,
    *,
    branch: str = "main",
    remote: str = "origin",
    secret_name: str | None = None,
    confirm: bool = False,
) -> dict:

    if not confirm:
        raise ValueError(
            "Push requires explicit confirmation"
        )

    root = _verify_repo(
        repo,
        branch,
        remote,
    )

    tracking = (
        f"refs/remotes/{remote}/{branch}"
    )

    if not _ref_exists(root, tracking):
        raise RuntimeError(
            "Remote tracking branch is unavailable"
        )

    outgoing = int(
        _git(
            root,
            "rev-list",
            "--count",
            f"{tracking}..HEAD",
        )
    )

    if outgoing != 1:
        raise RuntimeError(
            f"Expected exactly one outgoing "
            f"commit, found {outgoing}"
        )

    with git_environment(
        secret_name=secret_name
    ) as env:

        _git(
            root,
            "-c",
            "credential.helper=",
            "push",
            remote,
            f"HEAD:refs/heads/{branch}",
            env=env,
        )

    return {
        "commit": _git(
            root,
            "rev-parse",
            "--short",
            "HEAD",
        ),
        "branch": branch,
        "remote": remote,
    }
