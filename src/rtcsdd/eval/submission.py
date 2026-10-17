"""Read and write validated Codabench two-column score protocols."""

from __future__ import annotations

import math
from pathlib import Path

from rtcsdd.data.protocol import Example


def read_score_file(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Score file line {line_no}: expected exactly two columns")
            uid, raw_score = parts
            if uid in scores:
                raise ValueError(f"Duplicate score ID: {uid}")
            try:
                score = float(raw_score)
            except ValueError as exc:
                raise ValueError(f"Nonnumeric score: {uid}") from exc
            if not math.isfinite(score) or not (0 <= score <= 1):
                raise ValueError(f"Score must be finite in [0,1]: {uid}={raw_score}")
            scores[uid] = score
    if not scores:
        raise ValueError("Empty score file")
    return scores


def write_submission(
    examples: list[Example], scores: dict[str, float], out_path: str | Path
) -> None:
    if not examples or any(e.split not in {"progress", "eval"} for e in examples):
        raise ValueError("Submissions require an unlabeled progress/eval protocol")
    if any(e.label is not None for e in examples):
        raise ValueError("Submissions cannot use labeled protocols")
    ids = {e.utterance_id for e in examples}
    missing = ids - set(scores)
    extra = set(scores) - ids
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, unexpected={len(extra)}")
    for uid, score in scores.items():
        if not math.isfinite(score) or not (0 <= score <= 1):
            raise ValueError(f"Invalid score for {uid}: {score}")
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for item in examples:  # Strictly preserve official protocol order.
            f.write(f"{item.utterance_id} {scores[item.utterance_id]:.10f}\n")
