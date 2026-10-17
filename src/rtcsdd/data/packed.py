"""Read gated RTCFake ZIP-backed .pack archives without extracting them.

The team's .pack files were created with zipfile.ZipFile(... ZIP_DEFLATED ...).
Each archive member is a path like offline/en/Train_000000.wav.
This class opens an independent ZipFile lazily per process/worker.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .protocol import Example


class PackedAudioStore:
    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        self._handles: dict[tuple[str, str], zipfile.ZipFile] = {}
        self._pid = os.getpid()

    def archive_path(self, example: Example) -> Path:
        if example.split not in {"train", "dev"}:
            raise ValueError("Held-out audio is not available through the training pack store")
        return self.root / f"{example.split}_{example.view}.pack"

    def _archive(self, example: Example) -> zipfile.ZipFile:
        if self._pid != os.getpid():
            # Workers created by fork must not share an inherited ZipFile handle.
            # Close the child copies before opening fresh worker-local handles.
            for inherited in self._handles.values():
                inherited.close()
            self._handles = {}
            self._pid = os.getpid()
        key = (example.split, example.view)
        if key not in self._handles:
            archive = self.archive_path(example)
            if not archive.is_file():
                raise FileNotFoundError(f"Missing archive: {archive}")
            if not zipfile.is_zipfile(archive):
                raise ValueError(f"Expected a ZIP-backed .pack file: {archive}")
            self._handles[key] = zipfile.ZipFile(archive, "r")
        return self._handles[key]

    def has_member(self, example: Example) -> bool:
        try:
            self._archive(example).getinfo(example.utterance_id)
        except KeyError:
            return False
        return True

    def read(self, example: Example) -> tuple[np.ndarray, int]:
        archive = self._archive(example)
        # ZIP central-directory member lookup; no full-dataset extraction required.
        audio_bytes = archive.read(example.utterance_id)
        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if samples.ndim != 1:
            raise ValueError(f"Expected mono waveform: {example.utterance_id}")
        if sr != 16_000:
            raise ValueError(f"Expected 16000 Hz: {example.utterance_id} -> {sr}")
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            raise ValueError(f"Invalid/empty waveform: {example.utterance_id}")
        return samples, sr

    def member_count(self, split: str, view: str) -> int:
        pseudo = Example(f"{view}/en/dummy.wav", split, view, "en", 0)
        handle = self._archive(pseudo)
        return sum(1 for name in handle.namelist() if name.endswith(".wav"))

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_pid"] = None
        return state

    def __del__(self):
        if getattr(self, "_pid", None) == os.getpid():
            self.close()
