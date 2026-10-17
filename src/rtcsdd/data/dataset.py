"""Optional PyTorch dataset using the packed archives directly."""

from __future__ import annotations

from .audio import BASELINE_SAMPLES, crop_or_pad
from .packed import PackedAudioStore
from .protocol import Example


class RTCFakePackedDataset:
    """Returns waveform, numeric label, original utterance ID.

    No automatic augmentation: official RawBoost behavior must be implemented in
    the official-baseline adapter and explicitly configured there. The dataset
    does not silently change the organizers' training distribution.
    """

    def __init__(
        self,
        examples: list[Example],
        data_root: str,
        *,
        crop: str = "left",
        pad: str = "repeat",
        length: int = BASELINE_SAMPLES,
    ):
        if not examples or any(e.split not in {"train", "dev"} for e in examples):
            raise ValueError("Training dataset accepts nonempty train/dev records only")
        if len({e.split for e in examples}) != 1:
            raise ValueError("A dataset instance cannot mix train and dev examples")
        self.examples = examples
        self.store = PackedAudioStore(data_root)
        self.crop = crop
        self.pad = pad
        self.length = length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        item = self.examples[idx]
        samples, _ = self.store.read(item)
        fixed = crop_or_pad(samples, length=self.length, crop=self.crop, pad=self.pad)
        return torch.from_numpy(fixed), int(item.label), item.utterance_id
