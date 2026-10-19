"""Full, unaugmented development evaluation on pinned RTCFake protocols."""

from __future__ import annotations

import torch

from rtcsdd.eval.metrics import evaluate_subgroups
from rtcsdd.training.weighted_loss import weighted_ce_sum


def evaluate_full_dev(model, loader, examples, *, device: str, amp: bool,
                      class_weights: list[float]) -> tuple[dict, float]:
    model.eval()
    weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    scores: dict[str, float] = {}
    ce_sum = 0.0
    weight_sum = 0.0
    try:
        with torch.inference_mode():
            for audio, labels, ids in loader:
                audio = audio.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = model(audio)
                numerator, denominator = weighted_ce_sum(logits, labels, weights)
                ce_sum += numerator.item()
                weight_sum += denominator.item()
                for uid, p in zip(ids, logits.float().softmax(dim=-1)[:, 0].cpu().tolist()):
                    if uid in scores:
                        raise ValueError(f"Duplicate dev prediction: {uid}")
                    scores[uid] = p
        if len(scores) != len(examples):
            raise RuntimeError(f"Dev incomplete: {len(scores)} != {len(examples)}")
        metrics = evaluate_subgroups(examples, scores)
        return metrics, ce_sum / weight_sum
    finally:
        model.train()
