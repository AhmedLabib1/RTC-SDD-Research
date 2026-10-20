"""Portable, standard-library-only runtime preflight."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def secret_available(name: str) -> bool:
    """Check secret availability without revealing its value."""

    if os.environ.get(name):
        return True

    try:
        from kaggle_secrets import UserSecretsClient
        return bool(UserSecretsClient().get_secret(name))
    except (ImportError, Exception):
        return False


def safe_relative(root: Path, filename: str) -> Path:
    """Resolve only paths contained inside the given root."""

    candidate = Path(filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative path: {filename}")

    root = root.resolve()
    resolved = (root / candidate).resolve()

    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes root: {filename}")

    return resolved


def git_information(repo: Path) -> dict:
    """Read Git metadata without modifying the repository."""

    def run(*args):
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "--short", "HEAD"),
        "status": run("status", "--porcelain"),
    }


def inspect_runtime(
    *,
    repo: Path,
    dataset_root: Path | None,
    profile: dict,
) -> bool:
    """Inspect requirements and report readiness."""

    repo = Path(repo).resolve()

    print("=" * 65)
    print("REUSABLE RUNTIME PREFLIGHT")
    print("=" * 65)

    print("\n[1] ENVIRONMENT")

    print("Python:", sys.version.split()[0])
    print("Platform:", sys.platform)
    print("Repository:", repo)

    free_gib = shutil.disk_usage(
        repo.parent
    ).free / 1024**3

    print("Free disk:", f"{free_gib:.2f} GiB")

    minimum = float(profile.get("min_free_gib", 0))
    disk_ok = free_gib >= minimum

    print(
        "[PASS]" if disk_ok else "[FAIL]",
        "Minimum free disk:",
        f"{minimum:.2f} GiB",
    )

    print("\n[2] GIT")

    info = git_information(repo)

    print("Branch:", info["branch"])
    print("Commit:", info["commit"])
    print("Status:", info["status"] or "CLEAN")

    expected_branch = profile.get("branch")
    branch_ok = (
        not expected_branch
        or info["branch"] == expected_branch
    )

    print(
        "[PASS]" if branch_ok else "[FAIL]",
        "Branch requirement",
    )

    print("\n[3] REPOSITORY FILES")

    repo_ok = True

    for name in profile.get("repo_files", []):
        exists = safe_relative(repo, name).is_file()
        repo_ok &= exists

        print(
            "[PASS]" if exists else "[FAIL]",
            name,
        )

    print("\n[4] DATASET")

    dataset_ok = True

    if dataset_root is None:
        if profile.get("dataset_files"):
            dataset_ok = False
            print("[FAIL] Dataset root not provided")
        else:
            print("[SKIP] No dataset required")
    else:
        dataset_root = Path(dataset_root).resolve()

        for name in profile.get("dataset_files", []):
            exists = safe_relative(
                dataset_root, name
            ).is_file()

            dataset_ok &= exists

            print(
                "[PASS]" if exists else "[FAIL]",
                name,
            )

    print("\n[5] REQUIRED SECRETS")

    secrets_ok = True

    for name in profile.get("secrets", []):
        ready = secret_available(name)
        secrets_ok &= ready

        print(
            name,
            "READY" if ready else "MISSING",
        )

    ready = all([
        disk_ok,
        branch_ok,
        repo_ok,
        dataset_ok,
        secrets_ok,
    ])

    print("\n" + "=" * 65)

    if ready:
        print("[PASS] ENVIRONMENT PREFLIGHT COMPLETE")
    else:
        print("[FAIL] ENVIRONMENT NOT READY")

    return ready


def load_profile(path: Path) -> dict:
    profile = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(profile, dict):
        raise ValueError("Environment profile must be JSON object")

    return profile
