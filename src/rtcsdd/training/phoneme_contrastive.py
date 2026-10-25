
"""E02 experimental phoneme-matched symmetric contrastive loss.

Cache matches format: [token_id, offline_start, offline_end, online_start, online_end].
Frame span ends are exclusive. Strict feature/frame agreement avoids silent misalignment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def phoneme_contrastive_loss(
    offline_features: torch.Tensor,
    online_features: torch.Tensor,
    matches: Sequence[Sequence[int]],
    *,
    offline_frames: int,
    online_frames: int,
    temperature: float = 0.1,
    max_matches: int = 96,
) -> torch.Tensor:
    """Differentiable, symmetric in-pair InfoNCE over matching phoneme spans.

    Each 5-tuple is [phoneme_id, offline_start, offline_end,
    online_start, online_end]. Spans are [start, end) in CTC frames.
    Negative pairs with the same phoneme ID are masked to prevent false negatives.
    No projection parameters are added, to keep the initial loss interface simple.

    For empty/degenerate matching sets, return zero attached to both graphs.
    """
    if offline_features.ndim != 2 or online_features.ndim != 2:
        raise ValueError("Expected [T, D] tensors for both recordings")
    if offline_features.shape[1] != online_features.shape[1]:
        raise ValueError("Feature dimensions do not match")
    if offline_features.device != online_features.device:
        raise ValueError("Both feature tensors must be on the same device")
    if offline_features.shape[0] != offline_frames or online_features.shape[0] != online_frames:
        raise ValueError(
            "CTC/XLS-R frame-count mismatch: "
            f"offline {offline_frames}->{offline_features.shape[0]}, "
            f"online {online_frames}->{online_features.shape[0]}. "
            "Do not silently rescale phoneme alignment indices."
        )
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not isinstance(max_matches, int) or max_matches < 2:
        raise ValueError("max_matches must be an integer >= 2")
    if not isinstance(matches, (list, tuple)):
        raise ValueError("matches must be a list or tuple")

    zero = (offline_features.sum() + online_features.sum()) * 0.0
    if len(matches) == 0:
        return zero

    # Deterministic uniform subsampling limits peak memory.
    if len(matches) > max_matches:
        indices = [
            (i * (len(matches) - 1)) // (max_matches - 1)
            for i in range(max_matches)
        ]
        selected = [matches[i] for i in indices]
    else:
        selected = matches

    offline_vecs = []
    online_vecs = []
    ids = []

    for match in selected:
        if len(match) != 5 or any(type(v) is not int for v in match):
            raise ValueError(f"Invalid PCL match tuple: {match!r}")
        token, a, b, c, d = match
        if not (
            token >= 0 and 0 <= a < b <= offline_frames
            and 0 <= c < d <= online_frames
        ):
            raise ValueError(f"PCL span outside valid frame range: {match!r}")

        ids.append(token)
        offline_vecs.append(offline_features[a:b].mean(dim=0))
        online_vecs.append(online_features[c:d].mean(dim=0))

    if len(set(ids)) < 2:
        return zero

    off = F.normalize(torch.stack(offline_vecs).float(), dim=-1, eps=1e-8)
    on = F.normalize(torch.stack(online_vecs).float(), dim=-1, eps=1e-8)

    # FP32 logits even if the training regime later changes precision.
    logits = (off @ on.T) / temperature
    ids_t = torch.tensor(ids, device=logits.device, dtype=torch.long)
    same_token = ids_t[:, None].eq(ids_t[None, :])
    diag = torch.eye(len(ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_token & ~diag, -torch.inf)
    target = torch.arange(len(ids), device=logits.device)

    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
