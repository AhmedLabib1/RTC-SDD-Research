"""Exact upstream RawBoost Algo 5 (LnL then ISD), for training WAVs only.

No reimplementation of the two signal algorithms: this imports the pinned
organizers' RawBoost.py. Seeding per utterance/epoch makes restarts deterministic
independent of DataLoader worker counts; that is a deliberate local change.
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

import numpy as np


@lru_cache(maxsize=4)
def _official_module(upstream_root: str):
    source = Path(upstream_root) / "xlsr_aasist" / "utils" / "RawBoost.py"
    if not source.is_file():
        raise FileNotFoundError(f"Pinned official RawBoost file missing: {source}")
    spec = importlib.util.spec_from_file_location("rtcsdd_pinned_rawboost", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pinned RawBoost: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_seed(base_seed: int, epoch: int, utterance_id: str) -> int:
    payload = f"{base_seed}:{epoch}:{utterance_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def algorithm5(
    waveform: np.ndarray,
    *,
    upstream_root: str | Path,
    settings: Mapping,
    seed: int,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Official Algo 5: LnL convolutive distortion, followed by ISD impulse noise.

    Runs on untrimmed 16 kHz waveform, *before* crop/repeat pad, matching
    the organizers' train dataset order. Does not alter global NumPy RNG state.
    """
    if settings.get("algorithm") != 5:
        raise ValueError("This adapter supports pinned RawBoost algorithm 5 only")
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected a nonempty mono waveform")
    upstream = _official_module(str(Path(upstream_root).absolute()))
    # The pinned official code uses global np.random.*; isolate reproducibly.
    prior_np = np.random.get_state()
    prior_py = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        augmented = upstream.LnL_convolutive_noise(
            waveform,
            settings["N_f"], settings["nBands"], settings["minF"],
            settings["maxF"], settings["minBW"], settings["maxBW"],
            settings["minCoeff"], settings["maxCoeff"], settings["minG"],
            settings["maxG"], settings["minBiasLinNonLin"],
            settings["maxBiasLinNonLin"], sample_rate,
        )
        augmented = upstream.ISD_additive_noise(
            augmented, settings["P"], settings["g_sd"]
        )
    finally:
        np.random.set_state(prior_np)
        random.setstate(prior_py)
    augmented = np.asarray(augmented, dtype=np.float32)
    if augmented.shape != waveform.shape or not np.all(np.isfinite(augmented)):
        raise ValueError("Official RawBoost returned an invalid waveform")
    return np.ascontiguousarray(augmented)
