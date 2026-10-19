# Experiment Registry, Past Decisions and Research Roadmap

## Purpose

Improve a **single-model** speech deepfake detector under the RTC-SDD challenge constraints by changing one factor at a time, measuring Dev behavior, preserving the exact code/config/checkpoint state, and avoiding hidden-set tuning.

This file replaces an outdated plan that said no runnable training/inference existed. We now have completed **P01 FP32 Epochs 1 and 2 (9,474 total optimizer steps)**, saved-checkpoint full Dev ROC-AUC diagnostics and a functioning Progress inference/submission pipeline. The performance remains below research goals.

## 1. Resolve historical experiment ID collisions

There are **two naming systems** in the project's history:

1. The **planning catalog** historically used `P00` for official replication and `P01` for an online-only ablation.
2. The **actual implemented training run** is called **`p01_xlsr_aasist_official_like_fp32`**, and it was trained on **both Online and Offline Train**. It was not an online-only ablation.

To prevent future incorrect comparisons, use unique explicit identifiers such as `p01_xlsr_aasist_official_like_fp32_epoch1` for the existing run, and `ablation_online_only_001` for a prospective online-only experiment. Never claim the existing P01 run tested online-only data.

## 2. Milestones already executed

| Milestone | State | Evidence / limits |
|---|---|---|
| CPU protocol and pack validation | Completed previously | Train/Dev protocol counts, representative audio decoding |
| Official XLS-R+AASIST integration | Implemented and tested | Pinned upstream adapter, trained checkpoint |
| P00 warmup and v0.5.x GPU smoke | Used during development | Historical checks; not necessary to rerun after every restart |
| FP16 overflow investigation | Completed diagnostic | Led to FP32 choice; not a performance benchmark |
| **P01 FP32 training** | **Two full epochs completed** | **9,474 cumulative optimizer steps**, **75,785 Train per epoch** |
| Full Dev evaluation | Completed for Epochs 1 and 2 | **15,204 Dev**, checkpoint ROC-AUC **0.527068 → 0.578015** |
| HF private checkpoint backups | Completed | `last.pt`, `best_online_f1.pt`, `best_dev_loss.pt` |
| Progress inference pipeline | Tested and executed | `progress.pack` TAR.GZ, **30,279** score IDs |
| Official scored submissions | Two observed | Scores **32.2484** and **45.6397**; run attribution unverified |
| Complete source and notebooks pushed to Git | **Pending verification** | Historical remote repo may not contain Kaggle patch updates |

## 3. Critical observations after first epoch

> Epoch 2 is now completed; this first-epoch section is historical. See [P01 two-epoch audit](p01_epoch1_epoch2_audit.md). No class-weight or architecture ablations have been executed.

- **Class imbalance in predicted outcomes:** Spoof recall is approximately **24.17%** on full Dev at threshold 0.5; many true Spoof recordings are labeled Bonafide.
- Online Macro-F1 **36.22%**, Offline **32.25%**, all Dev **34.24%**.
- `[0.1, 0.9]` class weights place nine times more per-example loss weight on Bonafide than Spoof; may be relevant, but causal impact is untested.
- AMP FP16 was numerically unstable in the early setup. FP32 trained successfully, though this may increase resource cost.
- Noise and language domain gaps remain to be characterized using properly allowed local validation, not unlabeled Progress tuning.

## Epoch 2 controlled continuation — COMPLETED

Same P01 class-weighted CE and training settings; resumed original epoch-1 full checkpoint without restarting optimizer. Final step **9,474**. Training-time weighted Dev loss `0.670940 → 0.657102`, but Online Macro-F1 `0.362177 → 0.323371`. Saved-checkpoint ranking ROC-AUC **0.527068 → 0.578015**; Overall Macro-F1 `0.340536 → 0.317350`. Results are labeled Dev only, not official hidden competition scores. Continuation checkpoint and JSON reports were backed up privately. Exact comparisons: [P01 two-epoch audit](p01_epoch1_epoch2_audit.md).

## 4. Planned controlled experiments

The list below is **hypotheses**, not achieved gains. Each comparison must hold other training factors and optimizer-step budget fixed.

| Proposed ID | One changed factor | Control | Primary measurement | Priority |
|---|---|---|---|---|
| `A01_loss_weights` | `[0.1,0.9]` vs `[1,1]` / valid alternatives | Completed P01 configuration | Online Dev Macro-F1 and Spoof recall | Highest |
| `A02_data_mix` | Combined Train vs Online-only Train | Original combined dataset | Online Dev Macro-F1 | High |
| `A03_rawboost` | RawBoost Algo5 enabled/disabled | Enabled | Online Dev + clean diagnostics | High |
| `A04_legal_noise` | Defined **non-speech** noise augmentation | No additional noise | Legal synthetic noisy Dev proxy | High |
| `A05_crop` | First crop vs seeded alternative crop | First 64,600 samples | Online Dev and robustness | Medium |
| `A06_audio_length` | Baseline window vs longer window | 64,600 samples | Quality / VRAM / throughput | Medium |
| `A07_freeze_schedule` | Full fine-tuning vs staged SSL freezing | Both groups trainable | Macro-F1 per GPU hour | Medium |
| `A08_alt_architecture` | Single AASIST vs single Nes2Net | AASIST | Matched-budget Dev | Exploratory |
| `A09_pair_consistency` | Pair-aware representation loss | Plain weighted CE | Dev, no paired leakage | Exploratory |
| `A10_weights_average` | Allowed same-model weight averaging if rules permit | Best single snapshot | Dev only; check legality | Conditional |

Do not change multiple factors and describe the result as a clean ablation. A three-epoch run with a new sampler and new loss cannot identify which component helped.

## 5. Model-selection and challenge constraints

- Optimize using authorized **Train and labeled Dev** only.
- Keep **Progress and final Eval held out**: no training, pseudo-labeling, score-based hyperparameter selection, or class threshold fitting from hidden feedback.
- Challenge score is described as **0.3 clean + 0.7 noisy** F1; our normal Dev JSON does not contain hidden noisy labels.
- The reference evaluator uses **fixed 0.5 decision threshold**. Investigate calibration on Dev for understanding but do not silently change the competition cutoff.
- Do not train on unauthorized external speech or use prohibited external anti-spoof checkpoints. Carefully inspect competition rules/FAQ before expanding the data or attempting averaging.
- Do not presume prediction ensembling is permitted; the intended solution remains a **single model**.

## 6. Definition of a reproducible experiment

Every new experiment should record:

```text
experiment_id, date, owner
hypothesis and exactly one changed factor
Git commit SHA + dirty/clean flag
pinned upstream commit SHA
config JSON hash, training-fingerprint hash
original XLS-R checkpoint SHA256
Train/Dev dataset revision and counts
Python, torch, CUDA, Fairseq versions and GPU model
optimizer/learning rate/schedule/loss weights
batch size/accumulation, actual steps, seed, run time
RawBoost/noise source and seeds
checkpoint metadata + SHA256
Dev metrics (all/online/offline, EN/ZH, class F1, confusion)
legal/compliance notes, decision and follow-up
```

Back up binary checkpoints privately; commit human-readable summaries, config changes, unit tests and source code rather than private audio or large binary artifacts.

## 7. Prioritized next work

**Priority 0 — preservation:** commit the actual FP32 trainer, final configs, score generator and a cleaned reproducible notebook; update docs and verify `git status`. This is independent of model quality.

**Priority 1 — diagnosis without new training:** retain per-ID *authorized Dev* score outputs, check score histograms, mean/variance by class, confusion matrix, language/view subgroups, and whether the low Spoof recall is driven by class weighting or poor discriminative features. Examine noisy-Dev proxy only if constructed from permitted sounds.

**Priority 2 — fair minimal ablation:** first test class weights at matched Train/Dev and optimizer steps, with no simultaneous augmentation or architecture changes. Preserve all reports.

**Priority 3 — training budget:** after evidence justifies it, evaluate an additional epoch versus a fresh controlled alternative. Log GPU hours, memory and cost; do not spend hours rerunning P00 warmups or an already completed epoch due to Kaggle restarts.

## 8. Experiment decision template

```markdown
### EXPERIMENT_ID
- Hypothesis:
- Change from control (exactly one):
- Code commit + config fingerprint:
- Dataset and checkpoint provenance:
- Compute budget:
- Seed(s):
- Dev online Macro-F1:
- Dev subgroup breakdown:
- Quality / robustness trade-offs:
- Decision: retain / reject / investigate
- Next action:
```

Review organizer [challenge](https://www.junxue.tech/rtc-sdd-challenge/), [rules](https://www.junxue.tech/rtc-sdd-challenge/#rules) and [FAQ](https://www.junxue.tech/rtc-sdd-challenge/#faqs) before any new data or model-family decision.
