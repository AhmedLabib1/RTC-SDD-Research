# Dataset and pack-file contract

## Private Kaggle dataset (user's screenshot, 17 Oct 2026)

```text
rtcfake-packed-private/
  train_offline.pack         ZIP_DEFLATED, compression level 1, ZIP64
  train_online.pack          ZIP_DEFLATED, compression level 1, ZIP64
  dev_offline.pack           ZIP_DEFLATED, compression level 1, ZIP64
  dev_online.pack            ZIP_DEFLATED, compression level 1, ZIP64
  train_label.txt
  dev_label.txt
  train_offline_online_pairs.csv
  dev_offline_online_pairs.csv
  progress.txt
```

Packed member paths are the same as protocol IDs, for example:
`offline/en/Train_000000.wav`. No separate index: ZIP central directory handles
member lookup. The `.pack` extension is intentionally not `.zip` but the format is ZIP.

## Audit report (15 Oct 2026)

| split | offline | online | all |
|---|---:|---:|---:|
| train | 38,660 | 37,125 | 75,785 |
| dev | 7,739 | 7,465 | 15,204 |
| all | 46,399 | 44,590 | 90,989 |

Train: bonafide 14,353, spoof 61,432. Dev: bonafide 2,849,
spoof 12,355. All audio reportedly 16 kHz mono PCM16; 29 recordings
under 1s; no empty WAV, no decoder error, no byte-identical cross-split hashes.
Audit does **not** rule out near-duplicates, speaker/content overlap, or label
provenance issues.

The `train_*/dev_*_offline_online_pairs.csv` files are metadata only; pairing
is not proof of sample-perfect time alignment. 1,809 offline files have no online
counterpart and should stay in a standard classification training setup.

## Hard interface

- `Example.utterance_id`: always the exact protocol relative path.
- `Example.label`: 0=spoof, 1=bonafide; no labels for progress/eval.
- Input audio is a `float32` 1D waveform in [-1,1] (typical PCM16 conversion),
  16000 Hz, post decoder. No automatic train/eval audio resampling.
- V0.1 baseline window: first 64,600 samples, shorter clips tiled/repeated.
- `PackedAudioStore` reads only train/dev `.pack`, and lazily opens ZIP handles.
- `RTCFakePackedDataset` returns `(torch.FloatTensor[64600], int_label, id)`.

## Preconditions for training

1. Run `rtcsdd preflight --dataset-root ... --require-audit-counts`.
2. Run `rtcsdd smoke-audio --dataset-root ...`.
3. Compare decoded inputs from new pack reader with official librosa loader.
   NumPy/SoundFile can differ in normalization/dtype from librosa;
   check numerical equivalence and label consistency for representative audio.
4. Benchmark archive random access with 0/2/4 workers and file sizes;
   ZIP is not guaranteed faster than extracted files.
5. Do not export/generate a public README listing gated filenames or data metadata
   beyond organizer-authorized counts unless data use permits it.
