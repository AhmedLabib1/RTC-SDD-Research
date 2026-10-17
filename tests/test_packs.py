import io
import wave
import zipfile

import numpy as np

from rtcsdd.data.audio import crop_or_pad
from rtcsdd.data.packed import PackedAudioStore
from rtcsdd.data.protocol import Example


def _wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.ones(1600) * 1000).astype("<i2").tobytes())
    return buffer.getvalue()


def test_pack_read(tmp_path):
    with zipfile.ZipFile(tmp_path / "train_offline.pack", "w") as z:
        z.writestr("offline/en/Train_000000.wav", _wav_bytes())
    item = Example("offline/en/Train_000000.wav", "train", "offline", "en", 0)
    with_store = PackedAudioStore(tmp_path)
    assert with_store.has_member(item)
    wav, sr = with_store.read(item)
    assert sr == 16000
    assert wav.shape == (1600,)
    assert with_store.member_count("train", "offline") == 1
    with_store.close()


def test_padding_and_crops():
    audio = np.array([1, 2, 3], dtype=np.float32)
    assert crop_or_pad(audio, length=8, pad="repeat").tolist() == [1, 2, 3, 1, 2, 3, 1, 2]
    assert crop_or_pad(audio, length=5, pad="zero").tolist() == [1, 2, 3, 0, 0]
    assert crop_or_pad(audio, length=2, crop="center").tolist() == [1, 2]
