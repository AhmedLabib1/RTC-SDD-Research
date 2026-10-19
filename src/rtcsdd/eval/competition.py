"""RTC-SDD evaluation helpers: clean online Dev and *optional* noisy proxy.

Official score uses separately labeled clean/noisy ONLINE evaluation sets,
which are hidden from participants. Dev contains no noisy evaluation samples.
Do NOT label a Dev metric or locally simulated noise result as official score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rtcsdd.data.protocol import Example
from rtcsdd.eval.metrics import evaluate

CLEAN_WEIGHT = 0.3
NOISY_WEIGHT = 0.7


def dev_online_macro_f1(subgroup_metrics: Mapping) -> float:
    """Primary local checkpoint selection metric (fixed threshold = 0.5)."""
    score = float(subgroup_metrics["view/online"]["macro_f1"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Online Dev Macro-F1 must be finite in [0,1]")
    return score


def choose_better_checkpoint(
    online_macro_f1: float,
    dev_loss: float,
    best_online_macro_f1: float,
    best_dev_loss: float,
) -> bool:
    """Maximize online Macro-F1, breaking exact ties by lower dev loss."""
    if not math.isfinite(online_macro_f1) or not 0 <= online_macro_f1 <= 1:
        raise ValueError("Invalid candidate online Macro-F1")
    if not math.isfinite(dev_loss):
        raise ValueError("Invalid candidate dev loss")
    return online_macro_f1 > best_online_macro_f1 or (
        online_macro_f1 == best_online_macro_f1 and dev_loss < best_dev_loss
    )


def competition_proxy(
    clean_online_dev: Sequence[Example],
    clean_scores: dict[str, float],
    synthetic_noisy_online_dev: Sequence[Example],
    noisy_scores: dict[str, float],
) -> dict:
    """Optional 0.3/0.7 proxy using *locally simulated* noisy Dev.

    This helper DOES NOT synthesize noise. Both labeled cohorts must be passed
    explicitly and must represent ONLINE development utterances only.
    Never apply it to progress/final evaluation data (unlabeled/restricted).
    """
    for cohort in (clean_online_dev, synthetic_noisy_online_dev):
        if not cohort or any(e.split != "dev" or e.view != "online" for e in cohort):
            raise ValueError("Competition proxy accepts labeled online DEV only")
    clean = evaluate(list(clean_online_dev), clean_scores)["macro_f1"]
    noisy = evaluate(list(synthetic_noisy_online_dev), noisy_scores)["macro_f1"]
    return {
        "metric": "synthetic_noisy_online_dev_proxy_NOT_official_score",
        "threshold": 0.5,
        "clean_online_macro_f1": clean,
        "synthetic_noisy_online_macro_f1": noisy,
        "proxy_weighted_macro_f1": CLEAN_WEIGHT * clean + NOISY_WEIGHT * noisy,
    }
