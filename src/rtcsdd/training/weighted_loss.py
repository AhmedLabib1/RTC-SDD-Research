"""Weighted CE for gradient accumulation; preserves class weights with batch=1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_ce_sum(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.ndim != 1:
        raise ValueError("Expected [batch,2] logits and [batch] integer labels")
    return (
        F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum"),
        weights[labels].sum().detach(),
    )


def normalize_gradients(parameters, *, weight_sum: float, max_norm: float) -> float:
    """Call AFTER GradScaler.unscale_. Accumulated CE gradient is numerator only."""
    if weight_sum <= 0:
        raise ValueError("weight_sum must be positive")
    for param in parameters:
        if param.grad is not None:
            param.grad.div_(weight_sum)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(norm)
