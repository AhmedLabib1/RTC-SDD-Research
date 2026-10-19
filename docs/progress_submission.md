# Progress Inference and `submission.zip`

This page documents the **actual tested P01 Progress pipeline**. Its objective is simple: load the already fine-tuned model, produce one probability for each **30,279** unlabeled utterances, and package those scores for submission. **No training**, and **no automatic Hugging Face upload**.

## 1. Required inputs

| Item | Example reference path |
|---|---|
| Project root | `/kaggle/working/RTC-SDD-Research` |
| Progress IDs | `/kaggle/input/datasets/a7medlabib/rtcfake-packed-private/progress.txt` |
| Progress audio | Same directory, `progress.pack` |
| Original XLS-R checkpoint | `/kaggle/working/checkpoints/pretrained/xlsr2_300m.pt` |
| Fine-tuned model | `outputs/experiments/p01_xlsr_aasist_official_like_fp32/best_online_f1.pt` |
| Active P01 configuration | `src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json` |
| Inference script | `scripts/progress_inference_p01.py` (developed in Kaggle; verify committed state) |

The Progress pack is **TAR.GZ**, not the ZIP format used by Train/Dev. Each TAR member has the leading `progress/` path segment; its relative ID is the `progress.txt` entry, e.g. `clean/online/en/Progress_017403.wav`. Archive order is not protocol order.

## 2. Verified smoke-test milestone

A real three-sample smoke test loaded:

```text
clean/online/en/Progress_017403.wav -> P(spoof) 0.454088
clean/online/en/Progress_024306.wav -> P(spoof) 0.456552
clean/online/en/Progress_015701.wav -> P(spoof) 0.501852
```

It loaded checkpoint **epoch 1 / step 4737**, verified strict model weight loading, finite probability scores in `[0,1]`, and a sum of class probabilities near 1. These unlabeled predictions say **nothing** about their correctness.

## 3. Inference algorithm

1. Read `progress.txt` into an ordered unique list (expected **30,279** entries).
2. Optionally load a **local** `progress_scores_partial.txt` to skip completed IDs; reject unknown IDs.
3. Initialize the pinned official model using pretrained XLS-R, then load the **fine-tuned best-model weights** with `strict=True`.
4. `model.eval()` and `torch.inference_mode()`; **no backward or optimizer**.
5. Stream `progress.pack` with `tarfile.open(PACK, mode='r|gz')`.
6. For each `.wav`, strip the `progress/` prefix, match an expected ID, decode mono 16 kHz audio, left-crop/repeat-pad to **64,600** samples, accumulate a small GPU batch (batch size **2** was used).
7. Convert logits to `softmax[:,0]` **P(spoof)**. Save the ID and score locally after each batch.
8. Validate that every expected ID is present exactly once and no unexpected IDs exist.
9. Sort/reassemble into the exact order of `progress.txt`, then create `submission_progress.txt` (two columns).
10. Build a ZIP containing **exactly one root-level file named `scores.txt`**.

**Runtime caveat:** local partial scores survive errors or cell interruption only while that `/kaggle/working` state survives. They do **not** automatically survive a Kaggle session reset. GZIP streaming must advance through the archive from the beginning on restart, though already scored records can skip GPU inference.

## 4. Run the current source (only after verifying it exists)

In a Kaggle cell:

```python
import subprocess
from pathlib import Path

ROOT = Path('/kaggle/working/RTC-SDD-Research')
PY310 = Path('/kaggle/working/rtcsdd-baseline-py310/bin/python')
SCRIPT = ROOT / 'scripts/progress_inference_p01.py'
assert SCRIPT.is_file(), 'Inference script not yet restored/committed'

subprocess.run([str(PY310), '-u', str(SCRIPT)], cwd=ROOT, check=True)
```

No HF token is needed in this step, provided the model was restored earlier. **Do not run it again** if a validated completed score file already exists, unless a new, deliberately labeled inference experiment is intended.

## 5. Exact submission file contract

The project writer in `src/rtcsdd/eval/submission.py` validates:

- Two **whitespace-separated** columns per nonempty line.
- Column 1 is the **exact utterance ID**, preserving directory components.
- Column 2 is a **finite float** in `[0,1]`, the **spoof probability**.
- IDs must be complete, unique, and match the official Progress protocol.
- Rows must be written in the same order as `progress.txt`.

Example (**numbers invented for format illustration**):

```text
clean/online/zh/Progress_000000.wav 0.8234567890
clean/online/zh/Progress_000001.wav 0.1245678900
```

The submitted ZIP layout is:

```text
submission.zip
└── scores.txt
```

No parent folder, README, CSV header, labels, checkpoints or audio inside the ZIP.

## 6. Validate and package the finished inference text

Run **only after** `submission_progress.txt` contains all predictions:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import math

ROOT = Path('/kaggle/working/RTC-SDD-Research')
OUT = ROOT / 'outputs/experiments/p01_xlsr_aasist_official_like_fp32'
SOURCE = OUT / 'submission_progress.txt'
ZIP_PATH = OUT / 'submission.zip'
PROTOCOL = Path('/kaggle/input/datasets/a7medlabib/rtcfake-packed-private/progress.txt')

assert SOURCE.is_file(), 'Run inference first'
expected = [x.strip() for x in PROTOCOL.read_text().splitlines() if x.strip()]
rows = [x.strip() for x in SOURCE.read_text().splitlines() if x.strip()]
assert len(expected) == len(rows) == 30279
assert len(expected) == len(set(expected))

for uid, row in zip(expected, rows):
    columns = row.split()
    assert len(columns) == 2 and columns[0] == uid
    value = float(columns[1])
    assert math.isfinite(value) and 0 <= value <= 1

with ZipFile(ZIP_PATH, 'w', ZIP_DEFLATED) as archive:
    archive.write(SOURCE, arcname='scores.txt')
with ZipFile(ZIP_PATH) as archive:
    assert archive.namelist() == ['scores.txt']
    assert archive.testzip() is None

print('READY:', ZIP_PATH)
```

This local package check verifies shape and integrity; it **cannot** certify model accuracy or guarantee that the server's live submission rules have not changed. Confirm the organizer's current submission instructions before upload.

## 7. Interpreting finished submissions

Two finished entries were observed on the competition interface with scores **32.2484** and **45.6397**. The screenshot alone does not identify which source checkpoint or inference config belongs to each entry. Keep a private manifest mapping submission ID → `submission.zip` hash → model/config/score file hashes before making experiment comparisons.

No automatic cloud backups of Progress score files are required; the decision for this pipeline was to prioritize fast inference and submission packaging.
