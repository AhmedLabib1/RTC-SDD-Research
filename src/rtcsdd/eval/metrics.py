"""Strict, threshold-locked binary and subgroup metrics for dev only.

Class encoding from official baseline: spoof=0, bonafide=1.
The score is *P(spoof)* and is thresholded at >=0.5.
"""

from __future__ import annotations

import math
from collections import Counter

from rtcsdd.data.protocol import Example

THRESHOLD = 0.5


def class_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(examples: list[Example], scores: dict[str, float]) -> dict:
    if not examples or any(e.label is None or e.split != "dev" for e in examples):
        raise ValueError("Only labeled dev examples are eligible for local evaluation")
    ids = {e.utterance_id for e in examples}
    if ids != set(scores):
        raise ValueError(
            f"Score IDs mismatch: missing={len(ids - set(scores))}, "
            f"unexpected={len(set(scores) - ids)}"
        )
    counts = Counter()
    for e in examples:
        p_spoof = scores[e.utterance_id]
        if not math.isfinite(p_spoof) or not 0 <= p_spoof <= 1:
            raise ValueError(f"Invalid spoof score: {e.utterance_id}={p_spoof}")
        pred = 0 if p_spoof >= THRESHOLD else 1
        counts[(e.label, pred)] += 1
    spoof = class_metrics(counts[(0, 0)], counts[(1, 0)], counts[(0, 1)])
    bonafide = class_metrics(counts[(1, 1)], counts[(0, 1)], counts[(1, 0)])
    total = len(examples)
    return {
        "count": total,
        "threshold": THRESHOLD,
        "accuracy": (counts[(0, 0)] + counts[(1, 1)]) / total,
        "macro_f1": (spoof["f1"] + bonafide["f1"]) / 2,
        "spoof": spoof,
        "bonafide": bonafide,
        "confusion": {
            "true_spoof_pred_spoof": counts[(0, 0)],
            "true_spoof_pred_bonafide": counts[(0, 1)],
            "true_bonafide_pred_spoof": counts[(1, 0)],
            "true_bonafide_pred_bonafide": counts[(1, 1)],
        },
    }


def evaluate_subgroups(examples: list[Example], scores: dict[str, float]) -> dict:
    results = {"all_dev": evaluate(examples, scores)}
    for attr in ["view", "language"]:
        for value in sorted({getattr(e, attr) for e in examples}):
            group = [e for e in examples if getattr(e, attr) == value]
            results[f"{attr}/{value}"] = evaluate(
                group, {e.utterance_id: scores[e.utterance_id] for e in group}
            )
    for view in ("offline", "online"):
        for language in ("en", "zh"):
            group = [e for e in examples if e.view == view and e.language == language]
            if group:
                results[f"view_language/{view}/{language}"] = evaluate(
                    group, {e.utterance_id: scores[e.utterance_id] for e in group}
                )
    # No weighted official clean/noisy metric on dev: dev has NO noisy subset labels.
    return results
