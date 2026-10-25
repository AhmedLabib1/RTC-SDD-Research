# RTC-SDD — Experiments

We currently focus on three controlled fine-tuning experiments.

| Experiment | Training objective | Status |
|---|---|---|
| **E01** | Weighted Cross-Entropy (CE) | Completed baseline |
| **E02** | Weighted CE + Phoneme Contrastive Learning (PCL) | Epoch 1 completed and audited |
| **E03** | Weighted CE + PCL + Noise Augmentation | Planned |

## Shared experimental setup

- **Backbone:** XLS-R 300M + AASIST.
- **Initialization:** Each experiment starts independently from the original pretrained backbone.
- **Classification:** Spoof = 0, Bonafide = 1.
- **Precision:** FP32.
- **Evaluation:** All Dev and predefined online/offline and language subgroups.
- **Comparison:** Keep dataset partitions and common training settings aligned; document any changes.

## E01 — CE baseline

- Completed as the reference experiment.
- Best reported Online Macro-F1: **0.362177** (epoch 1).
- Corresponding All Dev Macro-F1: **0.342422**.
- Final evaluation notebook:
  `notebooks/04_e01_ce_baseline_final_evaluation.ipynb`

## E02 — CE + PCL

- **Epoch 1 completed:** 4,737 optimizer steps.
- **Dev weighted loss:** 0.689119.
- **Online Macro-F1:** 0.436917.
- **All Dev Macro-F1:** 0.438914.
- **Always-spoof All Dev Macro-F1:** 0.448311.

E02 outperformed E01 on the reported Macro-F1 metrics, but its
All Dev Macro-F1 remained below the always-spoof reference.
A strong bonafide prediction bias was also observed.
These limitations should be addressed and evaluated, rather than
interpreting the improvement over E01 as sufficient performance.

- Configuration: `src/rtcsdd/configs/e02_ce_pcl_fp32.json`
- Evaluation: `docs/results/e02_dev_epoch_001.json`
- Research notebook:
  `notebooks/05_e02_ce_pcl_complete_research.ipynb`

## E03 — CE + PCL + Noise Augmentation

**Planned — not yet trained.**

E03 will retain the E02 CE + PCL objective and introduce controlled,
permitted noise augmentation. Dataset-use restrictions and noise-source
eligibility must be checked before training.

## Checkpoint storage

Large model and optimizer checkpoints are stored separately on
Hugging Face, not in this GitHub repository.
