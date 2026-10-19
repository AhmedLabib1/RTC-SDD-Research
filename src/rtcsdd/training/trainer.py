"""Orchestrate one single-GPU, packed RTCFake XLS-R/AASIST experiment.

The optimizer/signal augmentations match the published baseline, whereas
accumulation, 3 epochs, optional AMP, deterministic per-clip RawBoost seeds,
and local F1 checkpoint selection are intentional Kaggle-friendly changes.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rtcsdd.data.dataset import RTCFakePackedDataset
from rtcsdd.data.protocol import read_protocol
from rtcsdd.eval.competition import dev_online_macro_f1
from rtcsdd.eval.dev_evaluator import evaluate_full_dev
from rtcsdd.models.xlsr_aasist import load_xlsr_aasist
from rtcsdd.training.checkpointing import (
    read_resume, save_best_model, save_resume, write_json_atomic,
)
from rtcsdd.training.config import fingerprint
from rtcsdd.training.weighted_loss import normalize_gradients, weighted_ce_sum
from rtcsdd.storage.backup import CheckpointBackupManager


def epoch_permutation(total: int, seed: int, epoch: int) -> list[int]:
    indices = list(range(total))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def optimizer_steps_per_epoch(samples: int, batch_size: int, accumulation: int) -> int:
    return math.ceil(math.ceil(samples / batch_size) / accumulation)


def rng_snapshot() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(data: dict) -> None:
    torch.set_rng_state(data["torch"])
    if data["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["cuda"])
    np.random.set_state(data["numpy"])
    random.setstate(data["python"])


def _loader(dataset, *, batch_size: int, workers: int, seed: int):
    # Use an independent RNG so DataLoader worker initialization cannot consume
    # the main model/dropout RNG unexpectedly. Recreated on each epoch/resume.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def model_trainability(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    ssl = getattr(model, "ssl_model", None)
    if ssl is None or not hasattr(ssl, "model"):
        raise RuntimeError("Official XLS-R backbone was not found")
    ssl_params = list(ssl.model.parameters())
    if not ssl_params or not all(p.requires_grad for p in ssl_params):
        raise RuntimeError("P00 requires ALL XLS-R parameters to be trainable")
    ssl_ids = {id(p) for p in ssl_params}
    head_params = [p for p in model.parameters() if id(p) not in ssl_ids]
    if not head_params or not all(p.requires_grad for p in head_params):
        raise RuntimeError("P00 requires ALL AASIST parameters to be trainable")
    return ssl_params, head_params


def _gradients_present(params) -> bool:
    """Graph connectivity check, not an FP16 finite-value check.

    A gradient containing Inf/NaN is still connected to the model; AMP's
    GradScaler handles a *numerical* overflow by skipping the optimizer update
    and lowering its scale. Treating it as a missing graph gradient prevents
    AMP from recovering and produces a misleading error.
    """
    return any(p.grad is not None for p in params)


def _gradient_diagnostics(params) -> tuple[int, int]:
    """Count grad tensors present, and those whose values are all finite."""
    present = 0
    finite = 0
    for param in params:
        if param.grad is not None:
            present += 1
            finite += int(bool(torch.isfinite(param.grad).all().item()))
    return present, finite


def apply_optimizer_update(*, optimizer, scaler, amp: bool, grad_norm: float) -> tuple[bool, float, float]:
    """Update weights only for valid FP32 gradients, or via AMP overflow guard.

    FP32 must check finite gradients *before* calling Adam.step(), otherwise
    Inf/NaN can permanently corrupt both weights and optimizer moments.
    When AMP is enabled, GradScaler owns that decision and may skip an update.
    Returns (skipped, old_scale, new_scale).
    """
    if not amp:
        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                "FP32 gradient norm is Inf/NaN; optimizer update NOT executed"
            )
        optimizer.step()
        return False, 1.0, 1.0
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after = scaler.get_scale()
    return after < before, before, after


def _append_history(output: Path, metrics: dict) -> None:
    history = output / "history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")


def _gpu_out_of_memory(exc: RuntimeError, batch_size: int) -> RuntimeError:
    if "out of memory" not in str(exc).lower():
        return exc
    return RuntimeError(
        f"CUDA OOM for micro-batch={batch_size}. XLS-R 300M trainable on one "
        "T4 may not fit batch=8; restart a NEW run with --batch-size 4 "
        "--accumulate 4 (or --batch-size 2 --accumulate 8). "
        "Do not resume a checkpoint with a changed effective batch setup."
    )


def train(config: dict, *, project_root: Path, dataset_root: Path,
          pretrained_checkpoint: Path, output: Path, resume: bool = False) -> None:
    train_cfg, data_cfg = config["training"], config["data"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(pretrained_checkpoint)
    official_root = project_root / "third_party" / "official_rtc_sdd"
    if not (official_root / "xlsr_aasist" / "utils" / "RawBoost.py").is_file():
        raise FileNotFoundError("Pinned upstream RawBoost.py is missing")
    if not (dataset_root / "train_label.txt").is_file():
        raise FileNotFoundError("Missing RTCFake packed protocols")
    output.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = output / "last.pt"
    if resume and not existing_checkpoint.is_file():
        raise FileNotFoundError("Cannot resume: last.pt not found")
    if not resume and (existing_checkpoint.exists() or (output / "config.json").exists()):
        raise FileExistsError("This experiment already exists; use --resume or a new --out")
    device = train_cfg["device"]
    amp = train_cfg["precision"] == "amp_fp16"
    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_records = read_protocol(dataset_root / "train_label.txt", "train")
    dev_records = read_protocol(dataset_root / "dev_label.txt", "dev")
    train_data = RTCFakePackedDataset(
        train_records, dataset_root,
        length=data_cfg["audio_length"], crop=data_cfg["crop"], pad=data_cfg["pad"],
        rawboost=data_cfg["rawboost"], upstream_root=official_root,
        seed=seed,
    )
    dev_data = RTCFakePackedDataset(
        dev_records, dataset_root,
        length=data_cfg["audio_length"], crop=data_cfg["crop"], pad=data_cfg["pad"],
        rawboost={"enabled": False}, seed=seed,
    )
    dev_loader = _loader(dev_data, batch_size=train_cfg["eval_batch_size"],
                         workers=train_cfg["workers"], seed=seed + 1_000_000)

    print("[MODEL] Loading pretrained XLS-R 300M + pinned AASIST ...", flush=True)
    model = load_xlsr_aasist(official_root, pretrained_checkpoint, device)
    # Explicitly unfreeze ALL parts, even if the loaded checkpoint has flags off.
    previously_frozen = sum(not p.requires_grad for p in model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if previously_frozen:
        print(f"[UNFROZEN] Reset requires_grad=True for {previously_frozen} tensors", flush=True)
    ssl_params, head_params = model_trainability(model)
    parameters = list(model.parameters())
    print(f"[TRAINABLE] XLS-R={sum(p.numel() for p in ssl_params):,}; "
          f"AASIST={sum(p.numel() for p in head_params):,}", flush=True)

    optimizer = torch.optim.Adam(
        parameters,
        lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weights = torch.tensor(train_cfg["class_weights"], device=device, dtype=torch.float32)
    config_hash = fingerprint(config)
    backup = CheckpointBackupManager(config, output)
    settings_manifest = {
        "config": config,
        "config_fingerprint": config_hash,
        "dataset_root": str(dataset_root),
        "pretrained": str(pretrained_checkpoint),
        "official_root": str(official_root),
        "notes": (
            "Local official-like XLS-R+AASIST (RawBoost5 train-only, "
            f"precision={train_cfg['precision']}, 3 epochs, gradient accumulation)."
        ),
    }
    if not resume:
        write_json_atomic(settings_manifest, output / "config.json")
    else:
        manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if manifest["config_fingerprint"] != config_hash:
            raise ValueError("Training configuration differs from original run")
        if manifest["dataset_root"] != str(dataset_root) or manifest["pretrained"] != str(pretrained_checkpoint):
            raise ValueError("Dataset or XLS-R pretrained checkpoint location changed")

    epoch = 0
    next_batch = 0
    global_step = 0
    best_dev_loss = math.inf
    best_online_f1 = -1.0
    no_improve_epochs = 0
    cached_rng = None
    if resume:
        checkpoint = read_resume(existing_checkpoint, expected_fingerprint=config_hash)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch = checkpoint["epoch"]
        next_batch = checkpoint["next_batch"]
        global_step = checkpoint["global_step"]
        best_dev_loss = checkpoint["best_dev_loss"]
        best_online_f1 = checkpoint["best_online_macro_f1"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        cached_rng = checkpoint["rng"]
        del checkpoint
        print(f"[RESUME] epoch={epoch+1} next_batch={next_batch} step={global_step}", flush=True)

    micro_batch_size = train_cfg["batch_size"]
    accumulation = train_cfg["gradient_accumulation"]
    batches_per_epoch = math.ceil(len(train_data) / micro_batch_size)
    print(f"[DATA] train={len(train_data):,} dev={len(dev_data):,}; "
          f"micro_batch={micro_batch_size} x accum={accumulation} "
          f"-> effective={micro_batch_size * accumulation}; "
          f"steps/epoch={optimizer_steps_per_epoch(len(train_data), micro_batch_size, accumulation):,}", flush=True)
    print("[AUGMENT] TRAIN RawBoost5 ON (official functions); DEV RawBoost OFF", flush=True)
    print(f"[EVAL] Clean ONLINE Dev Macro-F1; official 0.3/0.7 hidden score unavailable", flush=True)
    print(f"[CHECKPOINT] last.pt every {train_cfg['save_every_optimizer_steps']} steps, "
          "best_dev_loss.pt and best_online_f1.pt after epoch", flush=True)
    print(f"[PRECISION] {train_cfg['precision']}; "
          f"AMP={'ON' if amp else 'OFF'}, GradScaler={'ON' if amp else 'OFF'}", flush=True)
    print("[RESOURCE] Only cuda:0 is used; T4 x2 does NOT automatically combine VRAM", flush=True)

    started_at = time.monotonic()

    def snapshot(ep: int, batch: int, *, reason: str = "periodic"):
        save_resume(
            output / "last.pt", model=model, optimizer=optimizer, scaler=scaler,
            config_hash=config_hash, epoch=ep, next_batch=batch,
            global_step=global_step, best_dev_loss=best_dev_loss,
            best_online_macro_f1=best_online_f1,
            no_improve_epochs=no_improve_epochs, rng_state=rng_snapshot(),
            accumulated_samples=0,
        )
        print(f"[SAVED] last.pt epoch={ep+1} next_batch={batch} step={global_step}", flush=True)
        backup.saved_last(epoch=ep+1, global_step=global_step, reason=reason)

    model.train()
    for ep in range(epoch, train_cfg["epochs"]):
        train_data.set_epoch(ep)
        start_batch = next_batch if ep == epoch else 0
        if start_batch < 0 or start_batch > batches_per_epoch:
            raise ValueError("Invalid resume batch cursor")
        if start_batch != batches_per_epoch:
            order = epoch_permutation(len(train_data), seed, ep)
            partial = Subset(train_data, order[start_batch * micro_batch_size:])
            train_loader = _loader(
                partial, batch_size=micro_batch_size,
                workers=train_cfg["workers"], seed=seed + ep,
            )
            iterator = iter(train_loader)
            # Iterator worker creation consumes random state; restore dropout RNG.
            if cached_rng is not None:
                restore_rng(cached_rng)
                cached_rng = None
            optimizer.zero_grad(set_to_none=True)
            micro_count = 0
            weighted_sum = 0.0
            numerator_sum = 0.0
            current_batch = start_batch
            consecutive_amp_skips = 0
            for audio, labels, _ in iterator:
                current_batch += 1
                audio = audio.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        logits = model(audio)
                    numerator, denominator = weighted_ce_sum(logits, labels, class_weights)
                    if not bool(torch.isfinite(numerator)):
                        raise FloatingPointError("Non-finite training CE")
                    if amp:
                        scaler.scale(numerator).backward()
                    else:
                        numerator.backward()
                except RuntimeError as exc:
                    raise _gpu_out_of_memory(exc, micro_batch_size) from exc
                weighted_sum += denominator.item()
                numerator_sum += numerator.detach().item()
                micro_count += 1
                last_batch = current_batch == batches_per_epoch
                if micro_count < accumulation and not last_batch:
                    continue
                # First actual optimizer window: validate connectivity separately
                # from the numerical finite-gradient check handled by GradScaler.
                if global_step == 0:
                    ssl_count, ssl_finite = _gradient_diagnostics(ssl_params)
                    head_count, head_finite = _gradient_diagnostics(head_params)
                    print(
                        f"[GRAD] XLS-R present={ssl_count} finite={ssl_finite}; "
                        f"AASIST present={head_count} finite={head_finite}", flush=True,
                    )
                    if not ssl_count or not head_count:
                        missing = "XLS-R" if not ssl_count else "AASIST"
                        raise RuntimeError(
                            f"No backward gradient tensors for {missing}; "
                            "check graph connectivity (not an AMP overflow)."
                        )
                if amp:
                    scaler.unscale_(optimizer)
                grad_norm = normalize_gradients(
                    parameters, weight_sum=weighted_sum,
                    max_norm=train_cfg["gradient_clip_norm"],
                )
                # FP32 refuses non-finite gradients BEFORE any weight update.
                # In FP16, GradScaler handles overflow/skipped optimizer steps.
                try:
                    skipped, before_scale, after_scale = apply_optimizer_update(
                        optimizer=optimizer, scaler=scaler, amp=amp,
                        grad_norm=grad_norm,
                    )
                except RuntimeError as exc:
                    raise _gpu_out_of_memory(exc, micro_batch_size) from exc
                optimizer.zero_grad(set_to_none=True)
                loss_val = numerator_sum / weighted_sum
                micro_count, weighted_sum, numerator_sum = 0, 0.0, 0.0
                if skipped:
                    consecutive_amp_skips += 1
                    print(
                        f"[AMP OVERFLOW] optimizer update skipped; "
                        f"scale={before_scale:g}->{after_scale:g}; "
                        f"grad_norm={grad_norm:.3f}; "
                        f"consecutive={consecutive_amp_skips}", flush=True,
                    )
                    if consecutive_amp_skips >= 12:
                        raise FloatingPointError(
                            "12 consecutive AMP overflow skips. "
                            "Investigate numerical instability / try FP32."
                        )
                    continue
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        "Non-finite gradient norm was not handled by GradScaler"
                    )
                consecutive_amp_skips = 0
                global_step += 1  # only count executed optimizer updates
                if (global_step == 1 or global_step % train_cfg["log_every_optimizer_steps"] == 0):
                    print(
                        f"[TRAIN] epoch={ep+1}/{train_cfg['epochs']} "
                        f"batch={current_batch}/{batches_per_epoch} step={global_step} "
                        f"weighted_ce={loss_val:.5f} grad_norm={grad_norm:.3f} "
                        f"precision={train_cfg['precision']} "
                        f"elapsed_min={(time.monotonic()-started_at)/60:.1f}",
                        flush=True,
                    )
                stop_due_to_time = (
                    train_cfg["max_minutes"] > 0 and
                    time.monotonic() - started_at >= train_cfg["max_minutes"] * 60
                )
                if global_step % train_cfg["save_every_optimizer_steps"] == 0 or stop_due_to_time:
                    snapshot(ep, current_batch, reason="pause" if stop_due_to_time else "periodic")
                if stop_due_to_time:
                    print("[PAUSED] Time budget reached. Restart with --resume.", flush=True)
                    return
            # Snapshot right AFTER completed train epoch before evaluation; robust to eval error.
            snapshot(ep, batches_per_epoch, reason="pre_eval")
        else:
            # A previous run finished training but did not finish this epoch's Dev.
            # Recover the dropout RNG before moving into the next epoch.
            if cached_rng is not None:
                restore_rng(cached_rng)
                cached_rng = None
            print(f"[RESUME] epoch={ep+1} training finished, starting Dev evaluation", flush=True)

        print(f"[DEV] Evaluating all {len(dev_records):,} recordings, no augmentation", flush=True)
        metrics, loss = evaluate_full_dev(
            model, dev_loader, dev_records, device=device, amp=amp,
            class_weights=train_cfg["class_weights"],
        )
        online_f1 = dev_online_macro_f1(metrics)
        epoch_metrics = {
            "epoch": ep + 1,
            "step": global_step,
            "dev_weighted_loss": loss,
            "dev_online_macro_f1": online_f1,
            "official_hidden_weighted_f1": None,
            "metric_kind": "clean_online_dev_only_not_competition_score",
            "subgroups": metrics,
        }
        write_json_atomic(epoch_metrics, output / f"dev_epoch_{ep+1:03d}.json")
        write_json_atomic(epoch_metrics, output / "dev_metrics.json")
        # History is convenience; per-epoch files are authoritative on restart.
        _append_history(output, {k: v for k, v in epoch_metrics.items() if k != "subgroups"})
        print(f"[DEV] epoch={ep+1} loss={loss:.5f} "
              f"online_macro_f1={online_f1:.5f} all_macro_f1={metrics['all_dev']['macro_f1']:.5f}",
              flush=True)

        if loss < best_dev_loss:
            best_dev_loss = loss
            no_improve_epochs = 0
            save_best_model(
                output / "best_dev_loss.pt", model=model, epoch=ep+1,
                global_step=global_step, dev_loss=loss,
                online_macro_f1=online_f1, config_hash=config_hash,
                selection="min_dev_weighted_loss_official_reference",
            )
            print("[BEST] best_dev_loss.pt updated", flush=True)
            backup.saved_best("best_dev_loss.pt", epoch=ep+1, global_step=global_step)
        else:
            no_improve_epochs += 1
        if (online_f1 > best_online_f1):
            best_online_f1 = online_f1
            save_best_model(
                output / "best_online_f1.pt", model=model, epoch=ep+1,
                global_step=global_step, dev_loss=loss,
                online_macro_f1=online_f1, config_hash=config_hash,
                selection="max_clean_online_dev_macro_f1_not_official",
            )
            print("[BEST] best_online_f1.pt updated", flush=True)
            backup.saved_best("best_online_f1.pt", epoch=ep+1, global_step=global_step)

        # Mark start of NEXT epoch: correct resume even if interrupted now.
        snapshot(ep+1, 0, reason="epoch_end")
        next_batch = 0
        if no_improve_epochs >= train_cfg["early_stopping_patience"]:
            print(f"[EARLY STOP] Dev loss has not improved for {no_improve_epochs} epochs", flush=True)
            break

    print("[DONE] Requested fine-tuning epochs completed", flush=True)
    print(f"[OUTPUT] {output}", flush=True)
