"""RTCFake ZIP-packed dataset; official RawBoost is an explicit train-only option."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .audio import BASELINE_SAMPLES, crop_or_pad
from .packed import PackedAudioStore
from .protocol import Example
from .rawboost import algorithm5, sample_seed


class RTCFakePackedDataset:
    """Returns (waveform tensor, 0-spoof/1-bonafide label, utterance ID).

    * train: optional official Algo 5 BEFORE windowing
    * dev: always untouched (no leakage/augmentation)
    """

    def __init__(
        self,
        examples: list[Example],
        data_root: str | Path,
        *,
        crop: str = "left",
        pad: str = "repeat",
        length: int = BASELINE_SAMPLES,
        rawboost: Mapping | None = None,
        upstream_root: str | Path | None = None,
        seed: int = 1234,
    ):
        if not examples or any(e.split not in {"train", "dev"} for e in examples):
            raise ValueError("Nonempty train/dev examples required")
        if len({e.split for e in examples}) != 1:
            raise ValueError("Cannot mix train and dev examples")
        if rawboost and rawboost.get("enabled") and examples[0].split != "train":
            raise ValueError("RawBoost must be disabled for Dev")
        if rawboost and rawboost.get("enabled") and upstream_root is None:
            raise ValueError("Official upstream_root is required for RawBoost")
        self.examples = examples
        self.store = PackedAudioStore(data_root)
        self.crop, self.pad, self.length = crop, pad, length
        self.rawboost = dict(rawboost or {"enabled": False})
        self.upstream_root = str(upstream_root) if upstream_root is not None else None
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch index must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        item = self.examples[idx]
        samples, sr = self.store.read(item)
        if self.rawboost.get("enabled", False):
            samples = algorithm5(
                samples,
                upstream_root=self.upstream_root,
                settings=self.rawboost,
                seed=sample_seed(self.seed, self.epoch, item.utterance_id),
                sample_rate=sr,
            )
        fixed = crop_or_pad(samples, length=self.length, crop=self.crop, pad=self.pad)
        return torch.from_numpy(fixed), int(item.label), item.utterance_id
