# RTC-SDD Research — Documentation Hub

**Last updated:** 19 October 2026 · **Project:** XLS-R 300M + AASIST speech deepfake detection · **Status:** P01 Epochs 1 and 2 completed, full labeled Dev diagnostics saved; exploration pending.

This directory documents the **research project as implemented**, including historical decisions, pinned dependencies, the actual first-epoch experiment, limitations, and how to reproduce the training and prediction workflow. It is intended for teammates, reviewers, and future maintainers.

> **Source of truth:** the runnable Python modules and checked-in JSON configurations take precedence over prose. A number of new files were developed as Kaggle patch datasets and may not yet be committed to the remote Git repository. Check the Git commit and actual files before running commands. Never interpret a proposed experiment or a documented command as a completed result.

> **Latest verified results:** [P01 two-epoch audit and handoff](p01_epoch1_epoch2_audit.md), plus the output-stripped [audit notebook](../notebooks/02_p01_xlsr_aasist_audit_and_epoch2_continuation.ipynb). Historical sections below may still describe the first-epoch milestone.

## Where to start

| Your goal | Read | You'll learn |
|---|---|---|
| Understand the whole project | [Architecture](architecture.md) | Pipeline, model, source layout, architectural boundaries |
| Reproduce the runtime | [Setup & environments](setup.md) | Python 3.10, CUDA, Fairseq, pinned official code, Kaggle paths |
| Understand data and privacy | [Data & protocols](data.md) | Train/Dev ZIP packs, held-out TAR.GZ, labels, preprocessing |
| Understand P01 training | [Training](training.md) | FP32, loss, gradient accumulation, resume flags, checkpoints |
| Understand metrics | [Evaluation](evaluation.md) | Dev Macro-F1, confusion matrices, score direction, limitations |
| Reproduce submission inference | [Progress & submission](progress_submission.md) | Streaming TAR, ordered scores.txt and submission.zip |
| Restore trained models | [Checkpoints & recovery](checkpoints.md) | Hugging Face storage, best model vs full resume, security |
| Read the reported numbers | [Results](results.md) | Actual Dev results, observed submissions, caveats |
| Plan controlled improvements | [Experiments](experiments.md) | Historical P00/P01 naming, research hypotheses, decision log |
| Resolve runtime problems | [Troubleshooting](troubleshooting.md) | AMP issues, imports, missing paths, disk, save errors |
| Reproduce and review a result | [Reproducibility checklist](reproducibility.md) | Evidence, seeds, code state, deployment checklist |

## Verified project milestone

The `p01_xlsr_aasist_official_like_fp32` training run originally **finished 1 epoch**: **75,785 Train samples**, **4,737 optimizer updates**, and a full **15,204-sample Dev evaluation**. Online Dev Macro-F1 was **0.362177**, with an overall Dev Macro-F1 of **0.342422** at the fixed 0.5 decision threshold. The model successfully generated a **30,279-entry Progress submission**, and later completed **Epoch 2 (9,474 cumulative optimizer steps)**, with the full [two-epoch analysis](p01_epoch1_epoch2_audit.md). Its detection performance still needs improvement.

Do not treat the local Dev figures as the official clean/noisy weighted leaderboard metric, and do not attribute different team submissions to one checkpoint without retained submission metadata.

## Canonical locations

```text
README.md                                 # Root project introduction
docs/README.md                            # This index
docs/setup.md                             # Operational setup
src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json
src/rtcsdd/training/xlsr_aasist_finetune.py
src/rtcsdd/eval/submission.py
scripts/progress_inference_p01.py        # Currently used in Kaggle; commit needed
outputs/experiments/p01_xlsr_aasist_official_like_fp32/  # Local only
```

**Security:** no gated audio, progress data, tokens, model binary checkpoints, or personal Kaggle secrets are shipped with this documentation. Refer to the [official challenge](https://www.junxue.tech/rtc-sdd-challenge/) and [upstream baseline](https://github.com/JunXue-tech/RTC-SDD) for their current rules and source.
