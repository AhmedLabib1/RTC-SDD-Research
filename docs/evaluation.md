# Development Evaluation and Competition Metrics

## 1. Terminology and what is measurable

**Local Dev metrics** are computed on the labeled RTCFake Dev protocols. **Competition / hidden leaderboard scores** are calculated on a separate, held-out progress/final set with unknown ground-truth labels. The metrics are *not interchangeable*.

The project reports:

- Online Dev **Macro-F1** (primary local checkpoint selection criterion), count **7,465**.
- Offline Dev Macro-F1, count **7,739**.
- All Dev Macro-F1, count **15,204**.
- Subgroups by language (English `en`, Chinese `zh`) and view/language.
- Class-wise precision, recall, F1 and the **confusion matrix**.
- Class-weighted Dev cross-entropy (loss), used for an alternate best snapshot.

### Hidden weighted evaluation

The competition's documented score formula is:

```text
Official score = 0.3 * F1_clean + 0.7 * F1_noisy
```

The two hidden F1 components are unavailable from the normal clean Dev labels. The `official_hidden_weighted_f1` field in `dev_epoch_001.json` is correctly `null`. Do **not** calculate a fake “official” score by substituting Offline Dev or all-Dev Macro-F1 for the hidden noisy component.

## 2. Class convention, score direction and threshold

| Class | Internal label |
|---|---:|
| Spoof | 0 |
| Bonafide | 1 |

The raw model emits `logits` with class columns `[spoof, bonafide]`. Convert to **P(spoof)** as follows:

```python
spoof_scores = torch.softmax(logits.float(), dim=-1)[:, 0]
```

The validator uses a **fixed threshold of 0.5**:

```text
prediction = spoof     if P(spoof) >= 0.5
prediction = bonafide  otherwise
```

**Important competition distinction:** you submit continuous `P(spoof)` scores, not class IDs. The challenge scoring logic controls its threshold. Local experimentation with score histograms or threshold-response curves can diagnose calibration, but **do not silently choose or alter the official 0.5 decision cutoff** to claim a higher official score. Consult the organizer rules for model selection/tuning restrictions.

## 3. Macro-F1 calculations

For a given class, precision and recall are:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
Macro-F1  = (F1_spoof + F1_bonafide) / 2
```

A predicted class is treated as positive separately when calculating each class's precision/recall. Accuracy is useful as a diagnostic but can mislead when classes are imbalanced. The implementation in `src/rtcsdd/eval/metrics.py` validates finite scores in `[0,1]`, exact IDs and the fixed threshold.

## 4. Real P01 Epoch 1 results

Source: `dev_epoch_001.json`, checkpoint metadata `epoch=1, step=4737`.

| Group | Count | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| All Dev | 15,204 | 0.343988 | **0.342422** |
| Online Dev | 7,465 | 0.365975 | **0.362177** |
| Offline Dev | 7,739 | 0.322781 | **0.322469** |
| English Dev | 5,844 | 0.299624 | **0.299238** |
| Chinese Dev | 9,360 | 0.371688 | **0.369052** |
| Online / English | 2,844 | 0.312588 | **0.311171** |
| Online / Chinese | 4,621 | 0.398831 | **0.393153** |
| Offline / English | 3,000 | 0.287333 | **0.287328** |
| Offline / Chinese | 4,739 | 0.345221 | **0.344492** |

Weighted Dev loss: **0.6709395462**.

### Full Dev confusion matrix (fixed threshold 0.5)

| True class \\ Predicted class | Spoof | Bonafide | Total |
|---|---:|---:|---:|
| **Spoof** | 2,986 | 9,369 | 12,355 |
| **Bonafide** | 605 | 2,244 | 2,849 |
| **Total** | 3,591 | 11,613 | 15,204 |

The Spoof recall was **0.241684**, whereas Bonafide recall was **0.787645**. The model incorrectly labels many spoof examples as bonafide. More training **might** help, but the matrix alone does not establish the root cause.

### Online Dev class-specific metrics

- Spoof: precision **0.838317**, recall **0.272577**, F1 **0.411392**.
- Bonafide: precision **0.196286**, recall **0.771654**, F1 **0.312963**.
- Online Macro-F1 = **0.362177**.

## 5. Weighted Dev loss versus classification quality

The Dev weighted loss uses the configured class weights `[0.1, 0.9]`. It is not a substitute for class-balanced Macro-F1. A loss can decrease while the confusion matrix remains skewed. This project retains:

- `best_dev_loss.pt`: lowest weighted Dev loss, a separate selection policy.
- `best_online_f1.pt`: highest *local clean* Online Dev Macro-F1.

After only one full epoch, both checkpoints may represent the same epoch; comparing their saved metadata is the correct verification, not their filenames alone.

## 6. Correct evaluation workflow

1. Use labeled **Dev only**, with the same 64,600-sample preprocessing as P01.
2. Disable RawBoost and any train-only augmentation.
3. Call `model.eval()` and `torch.inference_mode()`.
4. Compute `P(spoof)` from logit column 0.
5. Keep exact Dev utterance IDs and validate all are covered once.
6. Evaluate at 0.5 with `evaluate_subgroups`.
7. Record aggregate and subgroup results, config fingerprint and checkpoint SHA256.
8. Do not tune model hyperparameters based on hidden Progress feedback.

The current `dev_evaluator.py` returns an aggregate score dictionary and weighted loss; it does not automatically persist per-ID Dev scores from every historical run. If per-ID scores are needed for further diagnostics, explicitly save them from a fresh authorized Dev inference.

## 7. Optional noisy-Dev proxy

`eval/competition.py::competition_proxy` accepts a synthetic noisy labeled Online Dev cohort and original clean Online Dev cohort, supplied separately. This helper **does not create synthetic noise**. Its weighted result is deliberately labeled `synthetic_noisy_online_dev_proxy_NOT_official_score`.

Any future synthetic-noise evaluation must document the augmentation source, license, seed, SNR distribution and split selection, and must not contain unauthorized external speech.

## 8. Reading leaderboard submissions safely

The team observed two completed submissions with scores **32.2484** and **45.6397** on a displayed leaderboard. The available screenshot does **not** establish which code or checkpoint produced each entry. Store submission IDs, exact score file hashes and model/config lineage before comparing them or claiming an improvement. Do not use hidden Progress leaderboard feedback to choose model hyperparameters if challenge rules prohibit it.
