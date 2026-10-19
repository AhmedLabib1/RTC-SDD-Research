# Reproducibility, Review and Artifact Preservation Checklist


> **Completed handoff snapshot (19 October 2026):** Epoch 1 + Epoch 2 checkpoints and paired labeled Dev diagnostics preserved. The output-stripped [archived audit notebook](../notebooks/02_p01_xlsr_aasist_audit_and_epoch2_continuation.ipynb) is opt-in by default. Its original code is historical and **should not be Run All-enabled** to replay expensive training. Raw research ZIP includes per-utterance authorized Dev scores; preserve privately, not in Git. Exact results: [two-epoch audit](p01_epoch1_epoch2_audit.md).

Use this checklist before declaring a research result reproduced or a Kaggle working session safely archived.

## 1. Source and provenance

- [ ] The **actual tested** `src/rtcsdd` code, `scripts` and notebook cells are committed to Git.
- [ ] The commit hash is recorded in the experiment report; uncommitted changes are disclosed.
- [ ] The pinned upstream revision matches `third_party/official_rtc_sdd.pin.json` (`bb68fcb57672439180b791c43f4496e924efc3c1`).
- [ ] The original `xlsr2_300m.pt` file origin and SHA256 are documented privately.
- [ ] Any Kaggle patch ZIP overlays are reconciled into one reviewed source tree, not silently stacked by order.

## 2. Environment

- [ ] Python interpreter and environment path are recorded (P01 used Python 3.10 venv).
- [ ] PyTorch, CUDA, Fairseq, NumPy versions are recorded from the **actual runtime**.
- [ ] GPU model/count and training device are recorded (P01 used single T4, `cuda:0`).
- [ ] Any compatibility fix (e.g. NumPy aliases) is in version-controlled first-party code.
- [ ] Package installation and infrastructure tests pass in the specified interpreter.

## 3. Data and preprocessing

- [ ] The Train, Dev and held-out files are accessible **only through approved credentials**.
- [ ] Train count **75,785** and Dev count **15,204**, with no unexpected duplicate IDs.
- [ ] The Train/Dev ZIP-backed `.pack` archives and label mapping are validated.
- [ ] Input is 16 kHz, mono, float32, first **64,600** samples or repeat padding.
- [ ] RawBoost 5 is enabled for Train only; disabled for Dev/Progress.
- [ ] Unlabeled Progress (`progress.txt` / TAR.GZ `progress.pack`) is isolated from training and model selection.

## 4. Training provenance

- [ ] Config JSON and effective run-time overrides are stored alongside a config fingerprint.
- [ ] Experiment ID is unambiguous; distinguish historical planning labels from actual run names.
- [ ] Loss class weights, optimizer, batch, accumulation, seed, epoch, effective batch and exact optimizer-step count are logged.
- [ ] Any numerical precision failures are documented; P01 successful epoch used **FP32**.
- [ ] Dev evaluation covers the full labeled protocol with no augmentation.
- [ ] A checkpoint selected by Dev metrics is not mislabeled as chosen by official hidden competition score.

## 5. Checkpoint integrity

- [ ] `last.pt` is identified as **full resume** and never substituted with a model-only snapshot.
- [ ] `best_online_f1.pt` and `best_dev_loss.pt` metadata, hash and selection metric are recorded.
- [ ] Restore source is trusted, especially for non-weights-only PyTorch load paths.
- [ ] Private artifact storage is verified, and secret tokens are not committed or copied to outputs.

## 6. Evaluation and reports

- [ ] Report Online, Offline and all-Dev Macro-F1 **separately**.
- [ ] Record language subgroups, class-wise precision/recall/F1 and confusion matrix.
- [ ] Use reference threshold **0.5**; submitted scores mean `P(spoof)`.
- [ ] `official_hidden_weighted_f1` is explicitly unavailable for Dev.
- [ ] Any synthetic noisy-Dev score is labeled a **proxy** and has documented augmentation provenance.
- [ ] Conclusions separate observed correlations from tested causal effects.

## 7. Submission and artifact lineage

- [ ] The inference script source matches the checkpoint/config actually used.
- [ ] Checkpoint metadata is logged (P01 `epoch=1`, `step=4737`).
- [ ] There are exactly **30,279** unique Progress IDs.
- [ ] Archive TAR names were mapped correctly to protocol IDs (`progress/` stripped).
- [ ] The final score rows are in the exact order of `progress.txt`.
- [ ] `submission.zip` contains exactly one root-level `scores.txt` with finite `[0,1]` scores.
- [ ] A SHA256 of both `scores.txt` and `submission.zip` is stored privately with submission ID and timestamp.
- [ ] Any leaderboard comparisons establish checkpoint ownership and do not use hidden Progress feedback for prohibited tuning.

## 8. Kaggle session handoff

- [ ] Current notebook is successfully saved or an exported `.ipynb` was verified; the **Failed to save draft** banner is resolved.
- [ ] The cleaned notebook lives under `notebooks/` with outputs/secret values removed.
- [ ] Important Python scripts and final configs are part of the Git commit.
- [ ] The root `README.md` and the links in `docs/README.md` point to real files.
- [ ] Sensitive and large local artifacts are excluded by `.gitignore`, and staged files are reviewed before pushing.
- [ ] A new environment can load the selected model checkpoint and reproduce at least the verified three-clip smoke inference.

**Result-publishing gate:** a reproducible submission requires an exact **Git SHA + effective config + model checkpoint hash + score-file hash** tuple. Without it, two scored submissions are observations, not a controlled experiment comparison.
