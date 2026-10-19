"""Crash-resumable rolling checkpoints + two model-only best snapshots.

`last.pt` stores FP32 master model and full Adam state (needed for real resume).
Best snapshots store model weights as FP16 to conserve Kaggle disk; reloading
into an FP32 model incurs a small quantization difference. Saved metadata makes
that transparent. Atomic replacement protects previous checkpoints when a save
is interrupted, at the cost of temporary disk space.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


def tensor_bytes(item) -> int:
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    if isinstance(item, dict):
        return sum(tensor_bytes(x) for x in item.values())
    if isinstance(item, (list, tuple)):
        return sum(tensor_bytes(x) for x in item)
    return 0


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    if tmp.exists():
        tmp.unlink()  # stale interrupted temporary, NOT the last good checkpoint
    bytes_needed = tensor_bytes(payload)
    free = shutil.disk_usage(destination.parent).free
    # torch serialization overhead + safety margin
    margin = 256 * (1024 ** 2)
    if free < bytes_needed * 1.05 + margin:
        raise OSError(
            f"Insufficient disk for safe atomic save of {destination.name}: "
            f"need ~{(bytes_needed*1.05+margin)/1024**3:.2f} GiB temporarily, "
            f"free {free/1024**3:.2f} GiB. Offload checkpoints or free storage."
        )
    try:
        torch.save(payload, tmp)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_resume(destination: Path, *, model, optimizer, scaler, config_hash: str,
                epoch: int, next_batch: int, global_step: int,
                best_dev_loss: float, best_online_macro_f1: float,
                no_improve_epochs: int, rng_state: dict, accumulated_samples: int):
    atomic_torch_save({
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config_fingerprint": config_hash,
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_dev_loss": best_dev_loss,
        "best_online_macro_f1": best_online_macro_f1,
        "no_improve_epochs": no_improve_epochs,
        "accumulated_samples": accumulated_samples,
        "rng": rng_state,
    }, destination)


def read_resume(path: Path, *, expected_fingerprint: str):
    # Only load a checkpoint CREATED BY THIS PROJECT and trusted by the user.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported or legacy checkpoint format: start a new v0.5 run")
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("Resume config mismatch: batch size/augmentation/LR/etc changed")
    return checkpoint


def save_best_model(destination: Path, *, model, epoch: int, global_step: int,
                    dev_loss: float, online_macro_f1: float, config_hash: str,
                    selection: str) -> None:
    state = {
        key: (value.detach().cpu().to(torch.float16).contiguous()
              if torch.is_floating_point(value) else value.detach().cpu().contiguous())
        for key, value in model.state_dict().items()
    }
    atomic_torch_save({
        "format_version": 2,
        "model_state_dict": state,
        "stored_weight_dtype": "float16_to_save_disk",
        "epoch": epoch,
        "global_step": global_step,
        "dev_loss": dev_loss,
        "dev_online_macro_f1": online_macro_f1,
        "config_fingerprint": config_hash,
        "selection_metric": selection,
    }, destination)


def write_json_atomic(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
