# Environment Setup and Reproduction Guide


> **Updated handoff (19 October 2026):** The source checkout at `08a723467157f220bf4340d52206282bbc79a3e1` was inspected with a clean Git working tree and tracked P01 implementation; the new completed Epoch 2 audit notebook is being added separately. Do not repeat Epoch 1/2 setup/training to reproduce reported metrics if using the verified private checkpoints; use [the two-epoch audit](p01_epoch1_epoch2_audit.md) and intentional checkpoint restoration instead. Historical patch notes below describe earlier development stages.

This is an operational guide to rebuilding the Python and GPU environment after a fresh Kaggle session. It replaces the **obsolete CPU-only / no-training-yet** version of this document. Its intended use is to set up the **existing** experiment, not to rerun its completed epoch.

## 1. Prerequisites and immutable inputs

- A clone of the **latest committed** `RTC-SDD-Research` repository, with all Kaggle-developed patches subsequently committed. If the Git checkout is still at v0.2, it is **not** the complete P01 runtime by itself.
- Permission to access the gated RTCFake files. The private Kaggle packed dataset is a user-specific example, not public distribution.
- Kaggle with an accelerator for GPU operations; single **NVIDIA T4 at `cuda:0`** was used for the completed run. A second available T4 is not used by this implementation.
- Python **3.10** virtual environment, compatible CUDA-enabled PyTorch and Fairseq, Git, `uv` and sufficient storage.
- Original pretrained XLS-R checkpoint, obtained from the permitted source and kept outside Git.
- Pinned official code revision, recorded at `third_party/official_rtc_sdd.pin.json`.

### Paths used in the completed Kaggle environment

| Variable | Value in the reference run |
|---|---|
| `ROOT` | `/kaggle/working/RTC-SDD-Research` |
| `PY310` | `/kaggle/working/rtcsdd-baseline-py310/bin/python` |
| `DATA` | `/kaggle/input/datasets/a7medlabib/rtcfake-packed-private` |
| `PRETRAINED` | `/kaggle/working/checkpoints/pretrained/xlsr2_300m.pt` |
| `OFFICIAL` | `$ROOT/third_party/official_rtc_sdd` |
| `CONFIG` | `$ROOT/src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json` |
| `OUT` | `$ROOT/outputs/experiments/p01_xlsr_aasist_official_like_fp32` |

These are **reference paths**, not hardcoded requirements for other accounts. Substitute your own authorized mount paths.

## 2. Install the lightweight project

For source inspection and CPU infrastructure testing on a developer machine:

```bash
cd /path/to/RTC-SDD-Research
python -m pip install -e '.[dev]'
python -m pytest -q
```

This command installs the project's basic infrastructure dependencies; it is **not enough** for the model's pinned GPU/Fairseq environment.

## 3. Create Python 3.10 environment on Kaggle

The completed Kaggle run used an isolated virtual environment managed by `uv`:

```bash
# Example commands to execute in a Kaggle notebook cell with %%bash
python -m pip install -q uv
uv venv --python 3.10 /kaggle/working/rtcsdd-baseline-py310
```

Python 3.10 must actually be available to `uv`; an environment-creation error means the interpreter must first be made available. **Do not silently switch** to the notebook kernel's newer Python if Fairseq compatibility is not verified.

## 4. Restore and verify pinned official source

```bash
cd /kaggle/working/RTC-SDD-Research
python scripts/setup/bootstrap_official.py
python scripts/setup/bootstrap_official.py --verify-only
```

The checkout is required to be clean and at commit:

```text
bb68fcb57672439180b791c43f4496e924efc3c1
```

The upstream repository is [`JunXue-tech/RTC-SDD`](https://github.com/JunXue-tech/RTC-SDD). Do not edit it in place; the first-party adapter should carry compatibility fixes.

## 5. Install pinned training dependencies

```bash
cd /kaggle/working/RTC-SDD-Research
python scripts/setup/install_training_env.py \
  --python /kaggle/working/rtcsdd-baseline-py310/bin/python \
  --install
```

The installer validates the clean upstream checkout and pinned requirements, and installs them into the specified environment. The successfully tested runtime used Python 3.10, PyTorch **2.5.1+cu121**, and the pinned Fairseq dependency. Package compatibility can change: validate, do not assume every new Kaggle image behaves identically.

Storage support can be installed if the checkpoint CLI is required:

```bash
uv pip install \
  --python /kaggle/working/rtcsdd-baseline-py310/bin/python \
  'huggingface_hub>=0.30,<1.0'
```

**NumPy / Fairseq compatibility:** `src/rtcsdd/models/xlsr_aasist.py` adds missing NumPy aliases in-memory before Fairseq is imported. Keep this adapter in the first-party code; no untracked third-party source edits.

## 6. Restore pretrained weights

The original pretrained SSL checkpoint is `xlsr2_300m.pt`; in the tested Kaggle runtime it was stored under:

```text
/kaggle/working/checkpoints/pretrained/xlsr2_300m.pt
```

Its Fairseq format is distinct from this project's **fine-tuned** `best_online_f1.pt` and `last.pt`. Reconstructing the model through the current adapter still requires the original pretrained checkpoint before loading the project's fine-tuned `model_state_dict`. Record a SHA256 digest when transferring the original file. Do not commit any `.pt` file to Git.

## 7. Validate paths, data, and GPU before computation

In a Kaggle Python cell:

```python
from pathlib import Path
import subprocess

root = Path('/kaggle/working/RTC-SDD-Research')
py310 = Path('/kaggle/working/rtcsdd-baseline-py310/bin/python')
data = Path('/kaggle/input/datasets/a7medlabib/rtcfake-packed-private')
pretrained = Path('/kaggle/working/checkpoints/pretrained/xlsr2_300m.pt')
config = root / 'src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json'

for item in [py310, data / 'train_label.txt', data / 'dev_label.txt',
             data / 'progress.txt', data / 'progress.pack', pretrained, config]:
    assert item.is_file(), f'Missing: {item}'

subprocess.run([str(py310), '-c',
    'import torch; print(torch.__version__); print(torch.cuda.is_available()); '
    'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'],
    check=True)
```

When validating CPU data packs, the `scripts/setup/verify_setup.py` utility may be used with `--dataset-root` and `--python310`, provided its upstream files are already restored.

## 8. Restore source patches if Git is still incomplete

The completed P01 code was developed over historical source patches (v0.4 → v0.5 → v0.5.1 → v0.6 → v0.7) and a separate Kaggle Progress inference script. Until these are merged into the Git repository, merely cloning the historical v0.2 Git branch will be insufficient.

Prefer **a reviewed, clean commit** of the final source. Treat the old patch archives as historical recovery aids, not a recommended routine install method: patch ordering can overwrite later fixes.

### Runtime integrity checkpoints

Verify these files exist before inference:

```text
src/rtcsdd/models/xlsr_aasist.py
src/rtcsdd/training/xlsr_aasist_finetune.py
src/rtcsdd/training/trainer.py
src/rtcsdd/training/config.py
src/rtcsdd/training/weighted_loss.py
src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json
src/rtcsdd/eval/submission.py
src/rtcsdd/storage/backup.py
scripts/progress_inference_p01.py
```

Some of these files currently exist only in the Kaggle working copy, so do not claim Git completeness until verified.

## 9. Session persistence and secret handling

- `/kaggle/input` is read-only; datasets must be attached and available to the notebook. `/kaggle/working` is mutable and may be lost on a session reset.
- Training checkpoints can be large (the full resume snapshot was approximately **3.8 GB**); keep approved, private backups. Prediction score text is small and can be downloaded locally after completion.
- Use `UserSecretsClient().get_secret('HF_TOKEN')` for authorized Hugging Face operations; do not print the value. **Progress submission inference does not require HF_TOKEN** after restoring its checkpoint.
- A displayed **“Failed to save draft”** warning requires attention. Verify the notebook was actually saved or download/export an `.ipynb` before closing the page.

## 10. What setup does not mean

Passing dependency installation or model-loading smoke tests does **not** mean that training should restart or that baseline performance was reproduced. For the completed run, restore `best_online_f1.pt` for inference and inspect the Dev report. Only launch `xlsr_aasist_finetune` when intentionally starting or resuming an experiment. See [training](training.md), [checkpoints](checkpoints.md) and [troubleshooting](troubleshooting.md).
