# RTCFake Data, Archive Formats and Protocol Contracts

## 1. Data inventory from the completed run

| Split / view | Labeled? | Recordings | Pack format | Usage |
|---|---|---:|---|---|
| Train — Offline | Yes | 38,660 | ZIP-backed `.pack` | Fine-tuning |
| Train — Online | Yes | 37,125 | ZIP-backed `.pack` | Fine-tuning |
| **Train total** | **Yes** | **75,785** | Two ZIP packs | Model updates |
| Dev — Offline | Yes | 7,739 | ZIP-backed `.pack` | Local diagnostics |
| Dev — Online | Yes | 7,465 | ZIP-backed `.pack` | Primary local model selection |
| **Dev total** | **Yes** | **15,204** | Two ZIP packs | Full validation |
| Progress | **No** | **30,279** | TAR.GZ named `progress.pack` | Submit scores only |

Train plus Dev comprises **90,989** labeled recordings. Audio decode checks use **16,000 Hz, mono**, non-empty, finite float32 values. The Progress IDs were checked for **30,279 distinct paths**.

## 2. Expected private Kaggle input layout

```text
rtcfake-packed-private/
├── train_label.txt
├── train_online.pack
├── train_offline.pack
├── dev_label.txt
├── dev_online.pack
├── dev_offline.pack
├── progress.txt
└── progress.pack
```

The dataset is gated. The path `/kaggle/input/datasets/a7medlabib/rtcfake-packed-private` is **specific to the existing account**, not a general public URL. Never copy data/labels into this Git repository or a public artifact.

## 3. Train and Dev protocols

The first-party parser `rtcsdd.data.protocol.read_protocol` expects **two whitespace-delimited fields** per labeled record:

```text
online/en/Train_000000.wav spoof
```

This is illustrative of the format, **not a quoted dataset row**. The parser expects a relative ID `view/language/file.wav`, where `view ∈ {online, offline}` and `language ∈ {en, zh}`. Class encoding is:

```text
spoof    -> 0
bonafide -> 1
```

The `PackedAudioStore` locates a file through the combined split/view archive name (e.g. `train_online.pack`) and reads the member under its `view/language/filename.wav` ID. Each worker opens a separate ZIP handle lazily; the reader checks the audio is valid and has the expected sample rate.

## 4. Progress is a different format

The actual **unlabeled Progress protocol** uses **four-part** IDs:

```text
clean/online/zh/Progress_000000.wav
clean/online/en/Progress_000004.wav
noisy/online/zh/Progress_000003.wav
```

Archive signature is **GZIP** (`1f8b08`); the GZIP payload is a **TAR** (`ustar`). Example archive member:

```text
progress/clean/online/en/Progress_017403.wav
```

The inference reader strips the one-time `progress/` archive prefix and looks up the result in the `progress.txt` set. The archive order is **not** protocol order; the writer must reassemble in protocol order. Avoid extracting the whole ~5.3 GB compressed archive when streaming is sufficient.

Do **not** pass the Progress pack to `PackedAudioStore` or unmodified `read_protocol` from the v0.5 source: the former rejects held-out splits and expects ZIP; the latter enforces three-part IDs and rejects the clean/noisy prefix. Use the dedicated Progress script or create a separately tested adapter.

## 5. Preprocessing contract

- Input waveform: mono float32 at **16 kHz**.
- Model input: **64,600 samples**, which equals **4.0375 seconds**.
- Longer: choose the **left/first** 64,600 samples in P01.
- Shorter: **repeat** the waveform to reach 64,600, then truncate.
- Train: apply official RawBoost Algorithm 5 (when enabled) **before** windowing.
- Dev and Progress: **no RawBoost**, no label exposure for Progress.
- Model outputs two logits ordered `[spoof, bonafide]`; select softmax column 0 for scoring.

Equivalent preprocessing is implemented in `src/rtcsdd/data/audio.py::crop_or_pad`. Random crop/other lengths can be proposed as **future controlled experiments**, not silently inserted into the P01 pipeline.

## 6. Data validation before training

Check: file presence, proper number of IDs, no duplicates per protocol, no Train/Dev leakage, ZIP membership for each labeled ID, WAV decoding, sample rate/channel/finite values, and alignment between `split`, `view`, `language`, labels and audio.

You can run the existing CPU verification once the project's package and upstream source are ready:

```bash
python scripts/setup/verify_setup.py \
  --dataset-root /path/to/your/authorized/rtcfake-packed-private \
  --python310 /path/to/your/python3.10
```

This is an infrastructure audit and **does not certify every WAV file** or validate model accuracy.

## 7. Privacy and allowed uses

- Treat all RTCFake audio, labels, IDs and checkpoint files under challenge access rules; do not redistribute them here.
- Progress/Eval are held out: **no training, pseudo-labeling, or tuning based on hidden labels or leaderboard feedback**.
- Review organizer terms before using any external audio. Restricted speech examples/anti-spoof checkpoints cannot be presumed permissible.
- Prefer aggregate reports in Git; do not commit per-utterance hidden scores unless explicitly permitted and reviewed.

## 8. Debugging pack problems

| Symptom | Probable cause | Action |
|---|---|---|
| `is_zipfile(progress.pack) == False` | Progress is TAR.GZ | Use streaming `tarfile.open(..., 'r|gz')` |
| `Expected view/language/filename.wav` | Progress has `clean/noisy` prefix | Use held-out-specific protocol logic |
| `Missing archive: progress_online.pack` | ZIP reader used for Progress | Use `progress.pack` |
| `Unknown audio: ...` | Archive name prefix or mismatched IDs | Strip `progress/` exactly once, verify protocol |
| Incorrect score row order | TAR order differs from protocol | Write rows in `progress.txt` order |
| Audio decode mismatch | Corrupt member / unsupported sampling | Inspect member via `soundfile`; no silent resampling |
