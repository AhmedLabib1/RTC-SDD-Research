# RTC-SDD Research

**Reproducible research and experimentation for real-time speech deepfake detection using XLS-R 300M + AASIST.**

This repository is our research workspace for the [RTC-SDD Challenge](https://www.junxue.tech/rtc-sdd-challenge/). It contains dataset/protocol validation, a pinned integration with the official XLS-R + AASIST baseline, FP32 fine-tuning, full development-set evaluation, checkpoint management, and progress-set inference/submission utilities.

> **Research status â€” 19 October 2026:** The XLS-R + AASIST pipeline has been implemented and run successfully. Experiment **P01 completed one full FP32 fine-tuning epoch**, evaluated on the entire development set, and produced a valid 30,279-recording progress submission. The current model's detection performance is **not competitive yet**; see [Results](#results-and-current-limitations). Some changes were developed in Kaggle using patch datasets and still need to be committed to GitHub. The repository is only fully reproducible once those source files and the cleaned notebook are committed.

## Contents

1. [Project goals](#project-goals)
2. [System overview](#system-overview)
3. [Current implementation status](#current-implementation-status)
4. [Repository structure](#repository-structure)
5. [Data and protocols](#data-and-protocols)
6. [Model and training configuration](#model-and-training-configuration)
7. [Environment setup](#environment-setup)
8. [Experiment workflow](#experiment-workflow)
9. [Evaluation and metrics](#evaluation-and-metrics)
10. [Results and current limitations](#results-and-current-limitations)
11. [Checkpoints and recovery](#checkpoints-and-recovery)
12. [Progress inference and submission](#progress-inference-and-submission)
13. [Reproducibility checklist](#reproducibility-checklist)
14. [Research roadmap](#research-roadmap)
15. [Data security and competition compliance](#data-security-and-competition-compliance)
16. [References](#references)

## Project goals

Speech deepfake detection asks whether an audio recording is genuine (*bonafide*) or synthetically generated/manipulated (*spoof*). The RTC-SDD setting involves multilingual speech, online/offline development conditions, and held-out progress audio with clean/noisy conditions.

Our immediate objectives are to:

- Reproduce a documented, auditable **XLS-R 300M + AASIST** baseline rather than report unverified scores.
- Fine-tune on the authorized RTCFake training data and evaluate without leakage using the provided development labels.
- Track class-wise detection quality, calibration, language/view differences, and noisy-audio robustness.
- Produce correctly ordered, validated competition submissions.
- Preserve source/config versions and checkpoint metadata so experiments can be resumed or reproduced across temporary Kaggle sessions.

**Important distinction:** A training run finishing successfully demonstrates pipeline functionality, **not** good spoof-detection accuracy or successful reproduction of published baseline results.

## System overview

```text
                Authorized RTCFake Train/Dev data
                      (ZIP-backed .pack archives)
                                  |
                                  v
                 Protocol parser + packed-audio reader
                                  |
                                  v
                  16 kHz waveform preprocessing
                  fixed length / crop / repeat pad
                                  |
                   +--------------+--------------+
                   |                             |
           Training only                      Dev inference
         optional RawBoost 5                 no augmentation
                   |                             |
                   +--------------+--------------+
                                  |
                                  v
                  XLS-R 300M SSL feature extractor
                                  |
                                  v
                     AASIST anti-spoofing head
                                  |
                                  v
                   2 logits: [spoof, bonafide]
                                  |
                    +-------------+-------------+
                    |                           |
             Training: loss             Inference: softmax
          weighted CE + Adam              score = P(spoof)
                    |                           |
            FP32 gradients                 Dev metrics /
            + checkpointing              progress scores
                                                |
                                                v
                                       scores.txt -> submission.zip
```

The implementation loads the official upstream model implementation through a pinned third-party checkout. The training integration is maintained in this repository, rather than editing upstream training files in place.

## Current implementation status

| Component | Status | Notes |
|---|---|---|
| Protocol and dataset auditing | Implemented | Strict IDs, label parsing, duplicate validation |
| Train/Dev packed audio | Implemented | ZIP-backed `.pack` members loaded on demand |
| Official XLS-R + AASIST integration | Implemented | Pinned upstream model, configurable XLS-R checkpoint |
| RawBoost | Implemented | Official Algorithm 5; train only |
| FP32 fine-tuning | Executed | P01: one completed epoch |
| Full Dev evaluation | Executed | 15,204 recordings; online/offline and language breakdowns |
| Cloud checkpoint backup | Implemented and used | Private Hugging Face model repository |
| Progress archive reader | Tested and used | Progress archive is **TAR.GZ**, not the train/dev ZIP layout |
| Progress submission | Executed | 30,279 IDs; validated `scores.txt` ZIP format |
| Dev threshold optimization | Planned | Must be fitted on labeled Dev, not on Progress scores |
| Strong, validated challenge performance | **Not achieved** | Current performance requires diagnosis |
| Clean, committed research notebook | Pending repository cleanup | Kaggle work needs to be saved to `notebooks/` |

## Repository structure

The layout below describes the implemented modules and intended home of the cleaned research notebook. Files initially restored from Kaggle patch datasets must be checked into Git before treating this layout as the canonical remote state.

```text
RTC-SDD-Research/
â”œâ”€â”€ README.md
â”œâ”€â”€ pyproject.toml                         # Package and development dependencies
â”œâ”€â”€ src/
â”‚   â””â”€â”€ rtcsdd/
â”‚       â”œâ”€â”€ data/
â”‚       â”‚   â”œâ”€â”€ protocol.py                 # Train/Dev/Progress protocol parsing
â”‚       â”‚   â”œâ”€â”€ packed.py                   # Train/Dev ZIP .pack reader
â”‚       â”‚   â”œâ”€â”€ dataset.py                  # PyTorch training/development dataset
â”‚       â”‚   â”œâ”€â”€ audio.py                    # Crop/pad preprocessing
â”‚       â”‚   â””â”€â”€ rawboost.py                 # Official train-only augmentation adapter
â”‚       â”œâ”€â”€ models/
â”‚       â”‚   â””â”€â”€ xlsr_aasist.py              # Load official model + XLS-R weights
â”‚       â”œâ”€â”€ training/
â”‚       â”‚   â”œâ”€â”€ xlsr_aasist_finetune.py     # Fine-tuning entry point
â”‚       â”‚   â”œâ”€â”€ trainer.py                  # FP32 training, checkpoints, evaluation
â”‚       â”‚   â”œâ”€â”€ config.py                   # Configuration loading/validation
â”‚       â”‚   â””â”€â”€ weighted_loss.py            # Weighted loss/gradient normalization
â”‚       â”œâ”€â”€ configs/
â”‚       â”‚   â””â”€â”€ p01_xlsr_aasist_official_like_fp32.json
â”‚       â”œâ”€â”€ eval/
â”‚       â”‚   â”œâ”€â”€ dev_evaluator.py            # Full development-set evaluation
â”‚       â”‚   â”œâ”€â”€ metrics.py                  # Macro-F1 and subgroup metrics
â”‚       â”‚   â”œâ”€â”€ competition.py              # Competition-oriented evaluation helpers
â”‚       â”‚   â””â”€â”€ submission.py               # Strict two-column score validation
â”‚       â””â”€â”€ storage/
â”‚           â”œâ”€â”€ huggingface.py              # HF storage integration
â”‚           â”œâ”€â”€ backup.py                   # Checkpoint backup handling
â”‚           â””â”€â”€ cli.py                      # Storage CLI
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ setup/                              # Environment / upstream setup
â”‚   â”œâ”€â”€ checkpoint_storage.py               # Storage utilities
â”‚   â””â”€â”€ progress_inference_p01.py           # Progress inference; local score resume
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 00_research_setup.ipynb              # Historical setup notebook
â”‚   â””â”€â”€ 01_xlsr_aasist_p01_pipeline.ipynb   # Planned cleaned notebook location
â”œâ”€â”€ tests/                                  # Protocol, training, storage tests
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ setup.md
â”‚   â””â”€â”€ experiments.md
â”œâ”€â”€ third_party/
â”‚   â””â”€â”€ official_rtc_sdd.pin.json           # Upstream source pin (checkout untracked)
â””â”€â”€ outputs/                                # Local experiment artifacts (NOT tracked)
```

**Naming note:** Historical scripts/notebooks may retain `P00` names, while the completed FP32 experiment is **P01**. `src/rtcsdd/configs/` is the active configuration directory; an earlier root-level `configs/` directory was retired during development.

## Data and protocols

### Authorized data

The RTCFake dataset is access-controlled. Obtain and use it only through the challenge's authorized distribution, and follow the competition rules. This repository does **not** redistribute recordings, labels, or held-out material.

Our Kaggle environment uses an attached private packed dataset, conventionally mounted as:

```text
/kaggle/input/datasets/a7medlabib/rtcfake-packed-private/
â”œâ”€â”€ train_label.txt
â”œâ”€â”€ train_online.pack
â”œâ”€â”€ train_offline.pack
â”œâ”€â”€ dev_label.txt
â”œâ”€â”€ dev_online.pack
â”œâ”€â”€ dev_offline.pack
â”œâ”€â”€ progress.txt
â””â”€â”€ progress.pack
```

The `a7medlabib` mount is specific to our Kaggle account; other users must point the scripts to their **own authorized** data path. This is not a download instruction or a repository dependency that is available publicly.

### Train/Dev

- The **Train** protocol contains **75,785** labeled utterances.
- The **Dev** protocol contains **15,204** labeled utterances.
- Label IDs are **`spoof = 0`** and **`bonafide = 1`**.
- Train/Dev `.pack` files use ZIP-backed containers that are read lazily, with members such as `online/en/Train_000000.wav`.
- Data validation requires mono, finite 16 kHz audio; audio is cropped/padded to the fixed model input length (64,600 samples in the completed run).
- RawBoost is enabled for Train only; Dev evaluation must use unaugmented audio.

### Progress (unlabeled)

The progress set contains **30,279 unique IDs**. Its `progress.txt` follows a different path convention:

```text
clean/online/zh/Progress_000000.wav
noisy/online/zh/Progress_000003.wav
clean/online/en/Progress_000004.wav
```

Unlike Train/Dev, `progress.pack` is a **GZIP-compressed TAR archive**. The TAR members have an additional prefix:

```text
progress/clean/online/en/Progress_017403.wav
```

The progress inference script strips `progress/` to match IDs in `progress.txt`, streams the TAR rather than extracting the entire archive, and writes final scores in **protocol order**, regardless of archive member order.

Do not apply the training `PackedAudioStore` or the original three-part train/dev ID validator directly to these four-part progress IDs.

## Model and training configuration

### Model

- **SSL backbone:** XLS-R 300M, initialized from the authorized pretrained XLS-R checkpoint.
- **Detector:** AASIST anti-spoofing head, using the pinned official model implementation.
- **Classes:** `[spoof, bonafide]` (two output logits).
- **Submitted score:** `softmax(logits, dim=-1)[:, 0]`, i.e. **P(spoof)** in `[0, 1]`.
- Both the SSL backbone and AASIST head were trainable during P01.

### P01: completed experiment

| Setting | Value |
|---|---|
| Experiment ID | `p01_xlsr_aasist_official_like_fp32` |
| Completed epochs | **1** |
| Training examples | **75,785** |
| Optimizer steps | **4,737** |
| Micro-batch | `2` |
| Gradient accumulation | `8` |
| Effective batch | `16` |
| Precision | **FP32**; no AMP autocast or GradScaler |
| Train augmentation | Official RawBoost Algorithm 5 |
| Dev augmentation | Disabled |
| Loss | Class-weighted cross-entropy |
| Class weights (class order `[spoof, bonafide]`) | `[0.1, 0.9]` |
| Gradient clipping | Maximum norm `1.0` |
| Random seed | `1234` (used in the development runs) |
| Training GPU | Single NVIDIA T4 (`cuda:0`) |

The fixed configuration is stored at:

```text
src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json
```

**Precision note:** Earlier FP16/AMP trials produced non-finite/overflowing gradients. The successfully completed P01 run used FP32 throughout. Do not assume AMP is safe for this configuration without a separate validated test.

**Optimization note:** `[0.1, 0.9]` was the *used* class weighting, not a demonstrated optimal choice. Current error analysis suggests it deserves review.

## Environment setup

### Prerequisites

- Python project in this repository.
- A CUDA-capable GPU for model fine-tuning and large-scale inference.
- An authorized RTCFake dataset attached at a local path.
- The official upstream RTC-SDD code checked out at the pinned revision.
- The XLS-R pretrained checkpoint at the configured path.
- Sufficient local disk space for weights, temporary checkpoint saves, and scores.

### 1. Install local package and run tests

For lightweight repository checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

**Full training** requires the pinned PyTorch/CUDA/Fairseq environment described in [`docs/setup.md`](docs/setup.md) and the setup scripts. Our Kaggle training environment used **Python 3.10**, **PyTorch 2.5.1 + CUDA 12.1**, and a pinned Fairseq checkout. The upstream Fairseq version required compatibility handling for deprecated NumPy aliases; see `src/rtcsdd/models/xlsr_aasist.py`.

### 2. Restore pinned upstream code

Use the repository setup process and verify the revision in:

```text
third_party/official_rtc_sdd.pin.json
```

The official checkout lives under `third_party/official_rtc_sdd/` and is intentionally not treated as first-party source. **Do not silently replace the pinned upstream revision with the current upstream HEAD.**

### 3. Check required paths

Our successful Kaggle run used:

```text
Project:       /kaggle/working/RTC-SDD-Research
Python:        /kaggle/working/rtcsdd-baseline-py310/bin/python
Dataset:       /kaggle/input/datasets/a7medlabib/rtcfake-packed-private
XLS-R weights: /kaggle/working/checkpoints/pretrained/xlsr2_300m.pt
Upstream:      /kaggle/working/RTC-SDD-Research/third_party/official_rtc_sdd
P01 config:    /kaggle/working/RTC-SDD-Research/src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json
```

These are **example paths from our runtime**, not portable defaults. Adapt them before running in another environment.

## Experiment workflow

### Stage 1 â€” Verify data and baseline integration

1. Check protocol completeness, labels, unique IDs, archive contents, 16 kHz decoding and preprocessing.
2. Verify the official upstream checkout and XLS-R checkpoint.
3. Run project tests and small CPU/GPU checks as needed.
4. Confirm Train-only RawBoost, class order, and FP32 finite gradients before a long run.

Legacy P00 warmup and GPU smoke tests were used during initial development; **they are not required every time a Kaggle session restarts**.

### Stage 2 â€” Run P01 fine-tuning

The existing entry point is:

```bash
python -m rtcsdd.training.xlsr_aasist_finetune \
  --root /path/to/RTC-SDD-Research \
  --dataset-root /path/to/rtcfake-packed-private \
  --checkpoint /path/to/xlsr2_300m.pt \
  --config /path/to/RTC-SDD-Research/src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json \
  --out /path/to/RTC-SDD-Research/outputs/experiments/p01_xlsr_aasist_official_like_fp32 \
  --epochs 1 \
  --max-minutes 0
```

**Do not execute this command merely to restore the environment.** Without a resume checkpoint it starts a new fine-tuning experiment. To resume an unfinished training run, restore the **full** `last.pt` checkpoint, verify the matching config/runtime, and use `--resume`. The `best_*.pt` files are intended for model selection/inference, **not** as drop-in optimizer-resume checkpoints.

### Stage 3 â€” Evaluate

Development evaluation runs without RawBoost and reports overall and subgroup metrics. The experiment JSON is saved as `dev_epoch_001.json`.

### Stage 4 â€” Produce progress predictions

After the trained weights are restored and a small inference smoke test passes, run the checked-in progress inference script from the project root (if present in the committed revision):

```bash
python scripts/progress_inference_p01.py
```

The script as developed in Kaggle uses these known Kaggle paths. It should be parameterized before treating it as a reusable general-purpose command. It writes a local partial score file so a transient script error can be resumed **while the Kaggle working directory survives**. It does **not** need to upload every 1,000 predictions to Hugging Face.

## Evaluation and metrics

The primary reported development metric here is **Macro-F1** (mean of class-specific F1 scores). We also track accuracy, precision/recall/F1 per class, confusion matrices, language subsets, and online/offline subsets.

The submission score is a continuous **spoof probability**, **not** a hard 0/1 label. Development confusion matrices below were calculated with **threshold = 0.5**; changing that threshold affects discrete-class metrics but should not be described as a new trained model.

The challenge's hidden score should **not** be equated with our clean Online Dev Macro-F1. The local JSON explicitly records `official_hidden_weighted_f1: null` and `metric_kind: clean_online_dev_only_not_competition_score`.

## Results and current limitations

### P01: completed epoch 1

| Measure | Result |
|---|---:|
| Epoch | 1 |
| Global optimizer step | 4,737 |
| Dev weighted loss | **0.670940** |
| Online Dev Macro-F1 (7,465 examples) | **0.362177** |
| Offline Dev Macro-F1 (7,739 examples) | **0.322469** |
| All Dev Macro-F1 (15,204 examples) | **0.342422** |
| All Dev accuracy at threshold 0.5 | **0.343988** |
| All Dev Spoof recall at threshold 0.5 | **0.241684** |
| All Dev Bonafide recall at threshold 0.5 | **0.787645** |

**All-Dev confusion matrix** (rows = true class; columns = predicted class, threshold 0.5):

| True / Predicted | Spoof | Bonafide |
|---|---:|---:|
| Spoof | 2,986 | 9,369 |
| Bonafide | 605 | 2,244 |

This is a serious limitation: the model classifies a large fraction of truly spoofed recordings as bonafide. The completed training run is **operationally successful but not yet a successful detection baseline**.

### Challenge submissions

The team observed two finished Codabench entries on **19 October 2026**, reporting scores **32.2484** and **45.6397** (as displayed by Codabench). Their precise model/run attribution is not established in this README, so they are recorded as observed entries **without claiming either as the definitive P01 result**.

### Hypotheses, not proven causes

- Decision-score concentration and threshold/calibration issues.
- Training objective/class weighting (`[0.1, 0.9]`) and class imbalance.
- Domain differences between training, clean Dev, and noisy held-out audio.
- Insufficient convergence after one epoch or unsuitable training hyperparameters.

Investigate with labeled Dev data and controlled experiments; do **not** use progress leaderboard feedback to fit a per-example decision rule or optimize a threshold against hidden labels.

## Checkpoints and recovery

Training outputs are intentionally **not checked into Git**. A private Hugging Face model repository holds the P01 artifacts:

```text
AhmedLabib1/rtc-sdd-research-checkpoints
â””â”€â”€ experiments/
    â””â”€â”€ p01_xlsr_aasist_official_like_fp32/
        â”œâ”€â”€ best_online_f1.pt       # Selected by Online Dev Macro-F1
        â”œâ”€â”€ best_dev_loss.pt        # Selected by development loss
        â”œâ”€â”€ last.pt                 # Full optimizer/training resume state
        â”œâ”€â”€ config.json
        â””â”€â”€ dev_epoch_001.json
```

- `best_online_f1.pt`: smaller **model inference checkpoint** (~636 MB as shown in the HF interface; size depends on display units).
- `best_dev_loss.pt`: alternative model-selection checkpoint.
- `last.pt`: substantially larger (~3.8 GB), for training resume.
- `dev_epoch_001.json`: reproducible summary of the first epoch's Dev metrics.

Store the HF access token in **Kaggle Secrets** as `HF_TOKEN`, never directly in source code or notebooks. A model checkpoint restored into `outputs/.../best_online_f1.pt` can be loaded through `torch.load(..., map_location="cpu", weights_only=True)` and applied with `model.load_state_dict(checkpoint["model_state_dict"], strict=True)`.

When restarting a Kaggle session:

1. Restore the matching checked-in source, upstream pin, Python environment, and pretrained XLS-R weights.
2. Restore the required P01 model checkpoint and config from the private HF repository.
3. Validate loading and run a small inference smoke test.
4. **Do not restart fine-tuning** unless a new controlled training experiment is intended.

## Progress inference and submission

### Progress loading

- Input archive: `progress.pack`, compressed TAR.GZ (not ZIP).
- Protocol: `progress.txt` with 30,279 unique entries.
- Members can be out of protocol order; output must follow **`progress.txt` order**.
- No progress labels are provided.
- Model input matches training/dev preprocessing; do **not** apply RawBoost during inference.
- Output is `P(spoof)` from the first of two softmax classes.

### Required submission format

```text
submission.zip
â””â”€â”€ scores.txt
```

Each line in `scores.txt` contains exactly two whitespace-separated fields:

```text
clean/online/zh/Progress_000000.wav 0.8234567890
clean/online/zh/Progress_000001.wav 0.1245678900
```

*These two scores are illustrative only, not real model outputs.* Every score must be finite and between `0` and `1`. Every ID must match the official protocol exactly once; no extra header, extra fields, or nested directory inside the ZIP.

Our helper `rtcsdd.eval.submission.write_submission(...)` validates unlabeled examples and score-ID consistency and writes score rows in protocol order. The final `.txt` may be written under a local descriptive name (for example `submission_progress.txt`) before it is packaged as **`scores.txt`** inside `submission.zip`.

### Local results and resumability

The progress inference runner writes local partial scores while processing the streaming archive. Local partial files help with a recoverable Python error, but **Kaggle session resets may remove `/kaggle/working`**. The current submission workflow deliberately avoids continuous Hugging Face uploads of prediction results. Persist final artifacts explicitly before ending the session.

## Reproducibility checklist

Before publishing a result or attempting to reproduce it:

- [ ] Record Git commit, upstream pin, experiment config, model/checkpoint fingerprint, and software environment.
- [ ] Validate authorized dataset splits, IDs, audio sample rate, and preprocessing.
- [ ] Confirm spoof=0 / bonafide=1 and that the submission score is **P(spoof)**.
- [ ] Verify that RawBoost is **Train-only**.
- [ ] Use the intended checkpoint; distinguish model-selection checkpoints from resume checkpoints.
- [ ] Report **which** Dev subgroup and threshold a number refers to.
- [ ] Run at least one small real-audio inference smoke test.
- [ ] Ensure all progress IDs are covered exactly once and preserve their official order.
- [ ] Validate `submission.zip` contains exactly `scores.txt` at its root.
- [ ] Keep private audio, keys, checkpoints, and unneeded outputs out of Git.

## Research roadmap

| Priority | Next investigation | Why |
|---|---|---|
| High | Save and analyze full labeled Dev scores | Distinguish calibration/threshold effects from real ranking failures |
| High | Review weighted CE and class balance | Spoof recall is extremely low at 0.5 threshold |
| High | Analyze clean/noisy and EN/ZH failures | Identify distribution-specific weaknesses |
| Medium | Check optimizer, LR schedule and convergence | One epoch may be insufficient |
| Medium | Controlled class-weight/augmentation ablations | Make changes based on measured Dev improvements |
| Later | Additional training and properly logged submissions | Only after diagnosing baseline behavior |

The next experiment should be driven by analysis of **labeled Train/Dev** results, not guesses based on two leaderboard numbers.

## Data security and competition compliance

- **Do not commit** gated RTCFake recordings, private protocols/labels, pretrained model binaries, full checkpoints, Kaggle tokens, or Hugging Face credentials.
- Do not redistribute private competition data through GitHub, cloud artifacts, or a public notebook.
- Keep `.pack`/`.wav` files, `outputs/`, local `checkpoints/`, and credentials outside version control.
- Consult the [official challenge rules](https://www.junxue.tech/rtc-sdd-challenge/#rules) for current data, external-resource, submission, and model restrictions before running or submitting a new experiment.
- Document any deviations from the official model/config and distinguish an upstream baseline from our fine-tuned variant.

## References

- [RTC-SDD Challenge](https://www.junxue.tech/rtc-sdd-challenge/)
- [Challenge rules](https://www.junxue.tech/rtc-sdd-challenge/#rules)
- [Challenge FAQ](https://www.junxue.tech/rtc-sdd-challenge/#faqs)
- [Official RTC-SDD baseline repository](https://github.com/JunXue-tech/RTC-SDD)
- [Official RTCFake dataset page](https://huggingface.co/datasets/JunXueTech/RTCFake) (access controlled)
- [`docs/setup.md`](docs/setup.md) â€” environment and data setup notes
- [`docs/experiments.md`](docs/experiments.md) â€” experimental design and constraints

---

**Maintainer note:** This README reflects results observed through 19 October 2026. The next repository maintenance step is to commit the final Kaggle-developed code and a cleaned notebook, then verify this documentation against a fresh checkout.
