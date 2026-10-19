# Actual Results and Known Limitations — 19 October 2026

## 1. Evidence sources

This report is based on the project's **real** P01 Epoch 1 development results, the observed checkpoint metadata, the successful three-sample Progress inference smoke test, and the displayed Codabench submissions. These sources answer different questions and must be kept separate.

- Experiment: `p01_xlsr_aasist_official_like_fp32`.
- Runtime checkpoint metadata: `epoch = 1`, `global_step = 4737`.
- Dev metrics source: `dev_epoch_001.json`.
- Train utterances: **75,785**; Dev utterances: **15,204**.
- Primary local metric: **Online Dev Macro-F1**.
- Training finished and produced checkpoint backups; successful execution is not evidence of competitive detection performance.

## 2. P01 development metrics

| Metric | Value |
|---|---:|
| Dev weighted loss | **0.6709395462** |
| All Dev Macro-F1 | **0.3424222530** |
| Online Dev Macro-F1 | **0.3621771561** |
| Offline Dev Macro-F1 | **0.3224689931** |
| English Dev Macro-F1 | **0.2992384300** |
| Chinese Dev Macro-F1 | **0.3690519904** |
| Online English Macro-F1 | **0.3111707021** |
| Online Chinese Macro-F1 | **0.3931530703** |
| Offline English Macro-F1 | **0.2873282654** |
| Offline Chinese Macro-F1 | **0.3444918609** |

All these values use the project's normal Dev evaluation at **threshold 0.5**. They are **not** the hidden progress/final clean-noisy weighted score.

## 3. Full Dev confusion matrix

| True \\ Predicted | Spoof | Bonafide |
|---|---:|---:|
| Spoof | **2,986** | **9,369** |
| Bonafide | **605** | **2,244** |

Derived counts: **12,355 actual Spoof**, **2,849 actual Bonafide**. The model predicts **Bonafide for 11,613 utterances**, including **9,369 spoof utterances**. The difference between the actual and predicted distributions highlights a strong operating-point bias.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Spoof | **0.831523** | **0.241684** | **0.374514** |
| Bonafide | **0.193232** | **0.787645** | **0.310331** |

These figures match the directly observed aggregate report. Treat the class-weighting `[0.1,0.9]` as an **unverified hypothesis** for the poor Spoof recall, not a proven explanation.

## 4. Online Dev detail

Online Dev: **7,465 recordings** with:

```text
true_spoof_pred_spoof       = 1654
true_spoof_pred_bonafide    = 4414
true_bonafide_pred_spoof    = 319
true_bonafide_pred_bonafide = 1078
```

- Spoof precision: **0.838317**; recall: **0.272577**; F1: **0.411392**.
- Bonafide precision: **0.196286**; recall: **0.771654**; F1: **0.312963**.
- Online Macro-F1: **0.362177**.

English-only Online Dev is weaker than Chinese-only Online Dev in this experiment. This is an observation, not proof of a causal language bias; dataset properties, recording artifacts, sample sizes and training mixture may confound the difference.

## 5. Model loading and prediction smoke-test evidence

A trained `best_online_f1.pt` checkpoint was restored from a private Hugging Face model repository. It contained `model_state_dict` with **674 named parameters/buffers**, checkpoint `epoch=1` and `step=4737`; strict load into the official XLS-R+AASIST adapter succeeded.

Three real Progress examples were decoded from the TAR.GZ and scored:

| Progress utterance | P(spoof) |
|---|---:|
| `clean/online/en/Progress_017403.wav` | 0.454088 |
| `clean/online/en/Progress_024306.wav` | 0.456552 |
| `clean/online/en/Progress_015701.wav` | 0.501852 |

All three outputs were finite, inside `[0,1]`, and probabilities summed to approximately one across both classes. Since Progress is unlabeled, these three values are **not** accuracy measurements.

## 6. Observed competition submissions

The challenge interface screenshot showed two `submission.zip` entries with status **Finished**:

| Displayed submission ID | Displayed date/time | Displayed score |
|---|---|---:|
| `1001718` | `2026-10-19 02:50` | **32.2484** |
| `1001603` | `2026-10-19 01:44` | **45.6397** |

**Do not infer** from this screenshot alone that both submissions used the same model, that one was an improvement to the other, or even which one belongs to a particular team member. The user indicated two participants contributed separate submissions. Attribution requires preserving a private record of the file hashes, checkpoint hashes, configuration and submitting account.

## 7. What cannot be concluded yet

- The official **Clean/Noisy component breakdown** has not been captured here.
- There is no proof from these results that an extra epoch would improve performance.
- There is no controlled ablation proving the `[0.1,0.9]` class weights caused the observed class bias.
- There is no clean controlled reproduction demonstrating parity with all aspects of the published upstream baseline.
- Threshold adjustments on hidden Progress feedback are not a compliant way to claim gains if ruled out by the competition protocol; keep the reference threshold at 0.5.

## 8. Measured priorities for follow-up

1. **Preserve artifacts:** source code, cleaned notebook, config/manifests, local Dev JSON and checkpoint metadata, and exactly which score file went with which submission.
2. **Diagnose labeled Dev only:** score distributions, class imbalance, online/offline and English/Chinese subgroups, data-label correctness, and model state/config consistency.
3. **Run a budget-matched class-weight ablation** rather than arbitrarily increasing training epochs.
4. **Test permitted robustness strategies** with synthetic noisy Dev labeled clearly as a proxy, not official hidden F1.
5. **Report clean/noisy official components separately** if the competition provides them and if using them for analysis is consistent with challenge rules.

This document reports **completed observations** separately from **research proposals**. The experiment is a valid technical milestone, but the current detection score is below expectations.
