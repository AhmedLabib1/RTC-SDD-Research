"""Load and validate reproducible, bundled XLS-R/AASIST experiment settings."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from rtcsdd.storage.backup import validate_backup_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "p00_xlsr_aasist_official_like.json"


def read_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path is not None else DEFAULT_CONFIG
    settings = json.loads(source.read_text(encoding="utf-8"))
    validate(settings)
    return settings


def with_overrides(config: dict, **overrides) -> dict:
    result = copy.deepcopy(config)
    for name, value in overrides.items():
        if value is not None:
            if name not in result["training"]:
                raise KeyError(name)
            result["training"][name] = value
    validate(result)
    return result


def validate(config: dict) -> None:
    train, data, model, evaluation = (
        config["training"], config["data"], config["model"], config["evaluation"]
    )
    if model["architecture"] != "xlsr2_300m_aasist":
        raise ValueError("Only pinned XLS-R 300M + AASIST is supported")
    if not model["xlsr_trainable"] or not model["aasist_trainable"]:
        raise ValueError("P00 requires BOTH XLS-R and AASIST fully trainable")
    for name in ("epochs", "batch_size", "gradient_accumulation", "eval_batch_size", "save_every_optimizer_steps", "log_every_optimizer_steps"):
        if not isinstance(train[name], int) or train[name] <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    if not isinstance(train["workers"], int) or train["workers"] < 0:
        raise ValueError("Invalid DataLoader worker count")
    if train["early_stopping_patience"] < 1 or train["max_minutes"] < 0:
        raise ValueError("Invalid early stop / max_minutes")
    if train["optimizer"] != "adam" or train["lr_scheduler"] != "none":
        raise ValueError("P00 is official-like Adam without scheduler")
    if train["learning_rate"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if train["precision"] not in ("amp_fp16", "fp32") or train["device"] != "cuda:0":
        raise ValueError("Supported precision: amp_fp16/fp32; device: cuda:0")
    if train["gradient_clip_norm"] <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if len(train["class_weights"]) != 2 or any(w <= 0 for w in train["class_weights"]):
        raise ValueError("Expected positive [spoof, bonafide] weights")
    if data["sample_rate"] != 16_000 or data["audio_length"] != 64_600:
        raise ValueError("Official XLS-R baseline uses exactly 16 kHz and 64600 samples")
    if data["crop"] != "left" or data["pad"] != "repeat":
        raise ValueError("Official preprocessing: first 64600 samples, repeated padding")
    if not data["rawboost"]["enabled"] or data["rawboost"]["algorithm"] != 5:
        raise ValueError("P00 must use official RawBoost algorithm 5")
    if evaluation["threshold"] != 0.5 or not evaluation["evaluate_full_dev_every_epoch"]:
        raise ValueError("P00 requires full Dev / threshold 0.5")
    if evaluation["noisy_proxy_generated"]:
        raise ValueError("P00 cannot claim unavailable hidden noisy evaluation")
    validate_backup_config(config.get("checkpoint_backup"))


def locked_config(config: dict) -> dict:
    """Only settings that would change training dynamics or data selection.

    Operational controls (save cadence, logs, workers, max_minutes, total epochs)
    can be changed on resume without invalidating the already trained weights.
    """
    result = copy.deepcopy(config)
    for name in (
        "epochs", "workers", "eval_batch_size", "save_every_optimizer_steps",
        "log_every_optimizer_steps", "max_minutes", "early_stopping_patience",
    ):
        result["training"].pop(name, None)
    result.pop("experiment", None)
    # Purely operational cloud settings can be enabled AFTER P01 has started.
    result.pop("checkpoint_backup", None)
    return result


def fingerprint(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(locked_config(config), sort_keys=True).encode("utf-8")
    ).hexdigest()
