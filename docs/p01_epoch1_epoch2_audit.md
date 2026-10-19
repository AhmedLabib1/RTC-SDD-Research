# P01 — Epoch 1 vs Epoch 2: Completed Audit and Research Handoff

**Date:** 19 October 2026
**Status:** Both epochs completed, saved-checkpoint inference repeated on all 15,204 labeled Dev examples, all per-ID predictions and analysis JSON retained privately. No Epoch 3 started.
**Source:** audited `.ipynb` archive in `notebooks/02_p01_xlsr_aasist_audit_and_epoch2_continuation.ipynb`; original reports in private `p01_epoch1_epoch2_research_backup.zip`.

## Training setup and provenance

- `p01_xlsr_aasist_official_like_fp32`: Epoch 1 completed at global step **4,737**.
- Continuation ID: `p01_epoch2_continuation`; resumed original `last.pt` with Adam optimizer, RNG, and original config fingerprint preserved; Epoch 2 completed at global step **9,474**.
- Both models were trained with **XLS-R 300M + AASIST**, both components trainable; FP32, Adam `1e-6`, weight decay `1e-4`, micro-batch `2`, gradient accumulation `8`, class weights `[spoof=0.1, bonafide=0.9]`, max gradient norm `1.0`, RawBoost Algorithm 5 on Train only.
- The continuation used Kaggle GPU T4 (`cuda:0`), Python 3.10, CUDA PyTorch 2.5.1, pinned official Fairseq/upstream.

## Serialized training-time Dev evaluations

These metrics originate from `dev_epoch_001.json` and `dev_epoch_002.json` after each epoch. They **are not guaranteed identical** to later inference runs from saved low-precision snapshots.

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Weighted Dev loss | 0.6709395462 | 0.6571018896 |
| Overall Dev Macro-F1 @ 0.5 | 0.3424222530 | 0.3187973282 |
| Online Dev Macro-F1 @ 0.5 | 0.3621771561 | 0.3233706979 |
| Offline Dev Macro-F1 @ 0.5 | 0.3224689931 | 0.3140930245 |
| English Dev Macro-F1 @ 0.5 | 0.2992384300 | 0.2995081723 |
| Chinese Dev Macro-F1 @ 0.5 | 0.3690519904 | 0.3307127296 |

## Comparable serialized-checkpoint Dev diagnostics

**Same Dev protocol, 15,204 examples, fixed threshold 0.5, no RawBoost**; saved inference weights used for each comparison. Epoch 1 uses original `best_online_f1.pt`; Epoch 2 uses `best_dev_loss.pt`. Both were checked for `epoch`/`global_step`, strict model loading, and recorded SHA256.

| Metric | Epoch 1 | Epoch 2 | Delta |
|---|---:|---:|---:|
| Overall Macro-F1 | 0.340536 | 0.317350 | −0.023187 |
| Online Macro-F1 | 0.360674 | 0.320433 | −0.040240 |
| Offline Macro-F1 | 0.320283 | 0.314082 | −0.006200 |
| Spoof recall | 0.239579 | 0.194496 | −0.045083 |
| Predicted Spoof share | 0.2347 | 0.1861 | −0.0486 |
| ROC-AUC | **0.527068** | **0.578015** | **+0.050947** |
| Average precision | 0.826001 | 0.841782 | +0.015780 |

The average precision should be interpreted relative to the Spoof prevalence of ~0.8126. ROC-AUC is threshold-independent, but the official scoring threshold of 0.5 remains fixed.

### ROC-AUC by subgroup — exact paired comparison

| Subgroup | Count | Epoch 1 | Epoch 2 | Delta |
|---|---:|---:|---:|---:|
| All Dev | 15,204 | 0.5271 | 0.5780 | +0.0509 |
| Online | 7,465 | 0.5356 | 0.5580 | +0.0224 |
| Offline | 7,739 | 0.5199 | 0.5971 | +0.0772 |
| English | 5,844 | 0.5196 | 0.5673 | +0.0477 |
| Chinese | 9,360 | 0.5365 | 0.5859 | +0.0493 |
| Online/EN | 2,844 | 0.4999 | 0.5455 | +0.0456 |
| Online/ZH | 4,621 | 0.5587 | 0.5641 | +0.0054 |
| Offline/EN | 3,000 | 0.5391 | 0.5878 | +0.0487 |
| Offline/ZH | 4,739 | 0.5157 | 0.6065 | +0.0908 |

AUC improvement is larger on Offline than Online. The near-flat Online/ZH improvement is an important follow-up but **not** proof of a causal language defect. Small subgroup deltas need uncertainty estimates before strong claims.

### Paired decisions at threshold 0.5

- Of **12,355 actual Spoof**: `2,119` changed from correct→wrong, `1,562` changed wrong→correct; **net −557 correctly detected Spoofs**.
- Of **2,849 actual Bonafide**: `284` changed correct→wrong, `466` changed wrong→correct; **net +182 correctly identified Bonafides**.
- Mean spoof score: `0.46772 → 0.46162` (−0.00609).
- Mean bonafide score: `0.46308 → 0.44810` (−0.01498).

Interpretation: score separation/ranking improved, but score levels shifted downward relative to the fixed decision threshold. This correlates with lower Spoof recall and worse Macro-F1; **no causal proof that the class weights caused this**.

## Artifact storage and integrity

Private Hugging Face model repository: `AhmedLabib1/rtc-sdd-research-checkpoints`.

| Remote prefix | Artifact | Role |
|---|---|---|
| `experiments/p01_xlsr_aasist_official_like_fp32/` | `best_online_f1.pt`, `last.pt`, `best_dev_loss.pt`, `dev_epoch_001.json`, `config.json` | Original Epoch 1 |
| `experiments/p01_epoch2_continuation/` | `last.pt`, `best_dev_loss.pt`, `dev_epoch_002.json`, `config.json` | Epoch 2; original P01 preserved |

`last.pt` is a full-state resume snapshot including optimizer, suitable for resuming training **only** after verifying epoch/step/config fingerprint. Epoch 2 `best_dev_loss.pt` is a compact model-weights snapshot and was verified at `epoch=2`, `global_step=9474`, SHA256 `5666549c8aebc8e47fef857aa370d3dbc47d8930a3084440d9833a74b6dc52cd`. Backup upload of final `last.pt` and `dev_epoch_002.json` was confirmed by the Kaggle run.

The private local archive `p01_epoch1_epoch2_research_backup.zip` includes per-recording labeled Dev scores, manifests, local diagnostics, both serialized Dev JSONs, runtime config, and training log. **Do not commit this private ZIP or its per-utterance score contents.** Keep the ZIP local or in approved private storage.

## Official submission provenance

- **P01 Epoch 1 submission**: ID `1001718`, official displayed Progress score **32.2484** (submission attributed by the experiment owner).
- **Teammate's separate experiment**: ID `1001603`, official displayed Progress score **45.6397**; architecture, loss implementation, training run and checkpoint lineage have **not** been verified.
- Epoch 2 has **not** been submitted in the documented work. No inference result from Dev should be described as an official competition score.

## Open questions and next steps (NOT completed experiments)

1. Establish confidence bounds on paired AUC improvements, accounting for Offline/Online pair dependence where possible.
2. Plan a budget-matched **Class Weight Ablation** with controls; `[0.1, 0.9]` being causal is unproven.
3. Compare conditional continuation to alternate general-speech pretrained backbones; audit eligibility and runtime first.
4. Review teammates' actual training/inference code before attributing their 45.6397 score to CE + PCL.
5. Study permitted PCL/RTC-robustness augmentation carefully against challenge restrictions.
6. No further P01 training until research team explicitly decides exploitation vs exploration. Continue using fixed 0.5 for selection-oriented Dev evaluation.

## Notebook code ownership

Most code in the 34 historical notebook cells is **exploratory audit and diagnostics**, not yet a reusable error-analysis subsystem. Refactoring into shared functions/classes under `src/rtcsdd` is intentionally postponed until agreed; do not claim it is already implemented. The cleaned notebook has all historical operations **disabled by default** to prevent accidental repeated GPU training or private HF writes.
