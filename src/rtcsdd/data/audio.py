"""Deterministic audio windowing; preserve exact baseline settings as one named policy."""

from __future__ import annotations

import numpy as np


BASELINE_SAMPLES = 64_600  # Official train/dev/inference: 4.0375s at 16000 Hz


def crop_or_pad(
    audio: np.ndarray,
    *,
    length: int = BASELINE_SAMPLES,
    crop: str = "left",
    pad: str = "repeat",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("Audio must be a nonempty mono waveform")
    if length <= 0:
        raise ValueError("Length must be positive")
    if crop not in {"left", "center", "random"}:
        raise ValueError(f"Unknown crop policy: {crop}")
    if pad not in {"repeat", "zero"}:
        raise ValueError(f"Unknown padding policy: {pad}")
    n = len(audio)
    if n >= length:
        if crop == "left":
            start = 0
        elif crop == "center":
            start = (n - length) // 2
        else:
            start = int((rng or np.random.default_rng()).integers(0, n - length + 1))
        output = audio[start : start + length]
    elif pad == "repeat":
        output = np.tile(audio, (length + n - 1) // n)[:length]
    else:
        output = np.pad(audio, (0, length - n), mode="constant")
    return np.ascontiguousarray(output, dtype=np.float32)
