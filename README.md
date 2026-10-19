# RTC-SDD Research

**Speech deepfake detection research with XLS-R 300M + AASIST.**

This repository contains our reproducible workflow for the [RTC-SDD Challenge](https://www.junxue.tech/rtc-sdd-challenge/): RTCFake data validation, model fine-tuning, development-set evaluation, checkpoint management, and Progress-set inference. We use a pinned copy of the [official RTC-SDD baseline](https://github.com/JunXue-tech/RTC-SDD), with our experiment code maintained separately.

> **Project status (19 October 2026):** The P01 XLS-R 300M + AASIST experiment has completed **2 FP32 epochs (9,474 optimizer steps)**. Epoch 2 improved labeled-Dev ROC-AUC but reduced Macro-F1 at the fixed 0.5 threshold. The original P01 Progress submission score is **32.2484**; a different teammate experiment scored **45.6397** (its training provenance remains unaudited). See [P01 two-epoch audit](docs/p01_epoch1_epoch2_audit.md).

## Model and pipeline

```text
Authorized RTCFake Train / Dev
        |
   Audio validation
        |
 16 kHz, 64,600 samples
        |
 [Train only: RawBoost 5]
        |
   XLS-R 300M backbone
        |
   AASIST classifier
        |
  Logits: [spoof, bonafide]
        |
  +-----+----------------+
  |                      |
Train: weighted CE    Inference: P(spoof)
Adam + FP32           softmax(logits)[:, 0]
  |                      |
Checkpoints           Dev metrics / Progress scores
                         |
                   submission.zip
```

Both XLS-R and AASIST were trainable during P01. RawBoost Algorithm 5 was used for training only; Dev and Progress inference were not augmented.

## Repository structure

```text
RTC-SDD-Research/
|-- README.md
|-- pyproject.toml
|-- src/rtcsdd/
|   |-- configs/             # Active experiment JSONs
|   |-- data/                # Protocols, packed audio, augmentation
|   |-- models/              # XLS-R + AASIST adapter
|   |-- training/            # FP32 fine-tuning and checkpoints
|   |-- eval/                # Metrics and score validation
|   |-- storage/             # Optional private checkpoint backup
|   `-- experiments/         # Experiment registry
|-- scripts/
|   |-- setup/               # Pinned source and runtime setup
|   |-- checkpoint_storage.py
|   `-- progress_inference_p01.py
|-- notebooks/
|   |-- 00_research_setup.ipynb
|   |-- 01_xlsr_aasist_p01_pipeline.ipynb
|   `-- 02_p01_xlsr_aasist_audit_and_epoch2_continuation.ipynb
|-- tests/
|-- docs/                    # Detailed technical documentation
|-- third_party/
|   `-- official_rtc_sdd.pin.json
`-- .github/
```

The official upstream checkout, large checkpoints, private data, and generated outputs are **not committed**.

## Data and preprocessing

We use the access-controlled [RTCFake dataset](https://huggingface.co/datasets/JunXueTech/RTCFake), obtained through authorized challenge access. The repository does not include audio, labels, or held-out data.

| Split | Recordings | Format | Role |
|---|---:|---|---|
| Train | 75,785 | ZIP-backed `.pack` files | Fine-tuning |
| Dev | 15,204 | ZIP-backed `.pack` files | Labeled evaluation |
| Progress | 30,279 | TAR.GZ archive named `progress.pack` | Unlabeled submission inference |

Audio is decoded as **mono, 16 kHz**, and cropped or repeat-padded to **64,600 samples**. Class indices are **`spoof = 0`** and **`bonafide = 1`**. The prediction submitted to the competition is the continuous **spoof probability**, not a hard class label.

Progress IDs include a `clean/` or `noisy/` prefix; archive entries additionally begin with `progress/`. The inference script streams the archive and writes predictions in the exact order of `progress.txt`, rather than archive order.

See [Data & protocols](docs/data.md) for format details and validation rules.

## Getting started

### Lightweight checks

Use **Python 3.10** to install the package and run unit tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The latest verified project test run passed **34 tests**. These checks do not load the full XLS-R model or reproduce GPU training.

### GPU training / inference environment

A full run also requires a CUDA-capable GPU, the authorized RTCFake packs, the original pretrained XLS-R checkpoint, the pinned official source checkout, and the compatible PyTorch/Fairseq environment. The completed P01 run used **Python 3.10**, **PyTorch 2.5.1+cu121**, and one **NVIDIA T4**.

Follow [Environment setup](docs/setup.md) and the scripts in `scripts/setup/`. Do not assume `pip install -e '.[dev]'` alone installs the pinned GPU stack. The original pretrained XLS-R weights and fine-tuned model checkpoints are separate files.

For the Kaggle workflow, see [`notebooks/01_xlsr_aasist_p01_pipeline.ipynb`](notebooks/01_xlsr_aasist_p01_pipeline.ipynb). Training and full inference are **opt-in** so opening the notebook does not start an expensive run. The [Epoch 2 audit notebook](notebooks/02_p01_xlsr_aasist_audit_and_epoch2_continuation.ipynb) is an execution-disabled research archive; all historical code cells are opt-in, and training/cloud writes require additional flags.

## P01 — First completed experiment

**Experiment:** `p01_xlsr_aasist_official_like_fp32`  
**Active configuration:** [`src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json`](src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json)

| Setting | Recorded value |
|---|---|
| Model | XLS-R 300M + AASIST |
| Training completed | 1 epoch; 4,737 optimizer steps |
| Precision | FP32 (AMP disabled) |
| Micro-batch / accumulation | 2 / 8 (effective batch 16) |
| Optimizer | Adam; learning rate `1e-6` |
| Class weights `[spoof, bonafide]` | `[0.1, 0.9]` |
| Gradient clipping | Max norm `1.0` |
| Augmentation | Official RawBoost Algorithm 5 on Train |

We switched to FP32 after FP16/AMP overflow issues during development. The project stores best-model snapshots compactly; their storage dtype must not be confused with the FP32 training precision.

### Development results — Epoch 1

| Metric | Value |
|---|---:|
| Weighted Dev loss | 0.670940 |
| Online Dev Macro-F1 | **0.362177** |
| Offline Dev Macro-F1 | **0.322469** |
| All Dev Macro-F1 | **0.342422** |
| All Dev Spoof recall | **0.241684** |

These values were measured on labeled Dev at a **0.5 decision threshold**; they are **not** the official hidden competition score. The low Spoof recall is a major limitation. Class weighting, calibration, and robustness need controlled investigation rather than assumptions.

Official Progress submission `1001718` came from **P01 Epoch 1** and scored **32.2484** (reported by the submitting team member). A different teammate submission `1001603` scored **45.6397**; its code, checkpoint lineage, and training strategy are **not yet verified**. See [Results](docs/results.md) and [Evaluation](docs/evaluation.md).

## P01 — Epoch 2 continuation (completed)

P01 was resumed from its original `last.pt` with optimizer/RNG state intact; the continuation used the same loss `[0.1,0.9]`, FP32, Adam, micro-batch 2, accumulation 8, RawBoost 5, and clipping norm 1.0. **Epoch 2 finished at step 9,474.**

| Saved-checkpoint labeled Dev diagnostic | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Overall Macro-F1 @ 0.5 | 0.340536 | 0.317350 |
| Online Macro-F1 @ 0.5 | 0.360674 | 0.320433 |
| Spoof Recall @ 0.5 | 0.239579 | 0.194496 |
| ROC-AUC (Spoof positive) | 0.527068 | **0.578015** |

The serialized **training-time** evaluation numbers differ slightly from fresh inference on the saved `best_*` snapshots. These two sources must not be interchanged. More training improved ranking but not fixed-threshold Macro-F1; **no Epoch 3 or loss ablation has been executed**. Full detail and preserved artifacts: [two-epoch audit report](docs/p01_epoch1_epoch2_audit.md).

## Checkpoints and recovery

The P01 model artifacts are backed up in a **private** Hugging Face repository and are intentionally omitted from Git:

- `best_online_f1.pt` — model weights selected using Online Dev Macro-F1.
- `best_dev_loss.pt` — model weights selected using Dev loss.
- `last.pt` — full state for resuming training, including optimizer state.
- `config.json` and `dev_epoch_001.json` — run metadata and evaluation summary.

Use a best-model checkpoint for inference; use `last.pt` only when intentionally resuming training. Credentials belong in Kaggle Secrets, not source files. See [Checkpoint recovery](docs/checkpoints.md).

## Progress inference and submission

Once the authorized Progress data and trained model are available, use `scripts/progress_inference_p01.py` with the paths expected by the Kaggle workflow. The script streams `progress.pack` **without extracting the full archive**, saves partial predictions locally, and writes a protocol-ordered `submission_progress.txt`.

The final submission must be packaged as:

```text
submission.zip
`-- scores.txt
```

`scores.txt` must have **30,279 lines** with exactly two whitespace-separated columns: `utterance_id` and `P(spoof)` in `[0, 1]`, with no header. No automatic Hugging Face uploads happen during Progress inference. See [Progress submission guide](docs/progress_submission.md) for the validated packaging procedure.

## Documentation and next steps

The [documentation hub](docs/README.md) links to the full guides for [architecture](docs/architecture.md), [training](docs/training.md), [reproducibility](docs/reproducibility.md), [experiments](docs/experiments.md), and [troubleshooting](docs/troubleshooting.md).

The Dev-only error analysis is complete through Epoch 2. The next step is a documented budget decision between additional CE-only continuation and controlled exploration (class weighting, eligible general-speech backbones, approved RTC consistency or noise-robustness ideas); these remain **proposals, not completed runs**. Changes should be tested one factor at a time and logged with their source commit, configuration, and checkpoint metadata.

**Data policy:** Never commit gated RTCFake files, private protocols, access tokens, pretrained weights, or generated prediction/checkpoint artifacts. Consult the [challenge rules](https://www.junxue.tech/rtc-sdd-challenge/#rules) before using external resources or changing the submission workflow.
