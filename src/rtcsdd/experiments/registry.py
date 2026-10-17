"""Immutable run manifests, independent of notebook runtime state."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def register_experiment(config_path: str | Path, runs_root: str | Path) -> Path:
    path = Path(config_path)
    config_text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict) or not config.get("name"):
        raise ValueError("Experiment config requires a nonempty name")
    if not isinstance(config["name"], str) or not all(
        c.isalnum() or c in {"_", "-"} for c in config["name"]
    ):
        raise ValueError("Experiment name must be alphanumeric, underscores, or dashes")
    digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    git_sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{config['name']}_{digest[:8]}"
    directory = Path(runs_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(config_text, encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at_utc": timestamp,
        "config_sha256": digest,
        "git_sha": git_sha,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "status": "registered_not_trained",
        "checkpoint_sha256": None,
        "dataset_manifest_sha256": None,
        "note": "Registration does not mean training or validation succeeded.",
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return directory
