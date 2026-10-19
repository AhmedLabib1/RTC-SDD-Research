# P01 Training: XLS-R 300M + AASIST in FP32


> **Epoch 2 update (19 October 2026):** P01 continued using its original FP32 CE settings, Adam/RNG resume state, finished step **9,474**; final checkpoint remote backup verified. The prior 4,737-step epoch-1 run is historical, not the current last training step. No Epoch 3 was run. Training loss improved but fixed-threshold Macro-F1 declined. See [two-epoch audit](p01_epoch1_epoch2_audit.md).

## 1. Completed experiment versus configuration defaults

The **completed** training milestone is **one full Epoch 1**, not a three-epoch claim. The active P01 JSON may specify `training.epochs = 3` as its default target, but the successful historical Kaggle command used `--epochs 1` to finish the first epoch and evaluate it. Command-line operational overrides are essential when reconstructing what was actually run.

Experiment name: `p01_xlsr_aasist_official_like_fp32`. This is an internal run identifier; do not equate it with a separately proposed online-only-vs-all-data hypothesis labeled “P01” in the original experiment planning document.

## 2. Training data and model

- Train: **75,785** labeled RTCFake utterances (Online + Offline).
- Full Dev: **15,204** labeled examples, without augmentation.
- Architecture: pinned official XLS-R 300M + AASIST.
- Both SSL backbone and AASIST head have `requires_grad=True`.
- Pretrained Fairseq XLS-R weights initialize the model; fine-tuning changes weights.
- Training uses a **single CUDA GPU** (`cuda:0`), not a two-GPU DDP implementation.

## 3. Recorded P01 configuration

| Parameter | Value |
|---|---|
| Precision | **FP32**, no AMP autocast or active GradScaler |
| Micro-batch | **2** |
| Gradient accumulation | **8** |
| Effective batch | **16** |
| Optimizer | **Adam** |
| Learning rate | **1e-6** |
| Weight decay | **1e-4** |
| Scheduler | **none** |
| Class weights `[spoof, bonafide]` | **[0.1, 0.9]** |
| Gradient clip max norm | **1.0** |
| RawBoost | Official Algorithm 5, **Train only** |
| Fixed waveform | 16 kHz, 64,600 samples, left crop, repeat pad |
| Seed | **1234** |
| DataLoader workers in config | **2** |
| Eval batch | **2** |
| Checkpoint interval | Every **1,000 optimizer steps** |
| Actual completed run | **1 epoch; 4,737 optimizer steps** |

Read `src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json` for the exact serialized config. Report the **run-time override** (`--epochs 1`) along with the JSON. The run's `outputs/.../config.json` manifest records the effective config and fingerprint and should be preserved privately.

## 4. What an optimizer step means here

With **75,785 recordings**, a micro-batch of **2** produces **37,893** micro-batches (last batch shorter). With **8** accumulated micro-batches per optimizer update, the number of updates in the observed epoch is **4,737**. One optimizer step is not a micro-batch; it is the update after gradient accumulation (or the final partial window).

Gradient normalization uses the summed class-weight denominator of the accumulated window. This matters when the true class proportions vary between micro-batches and the final step has fewer items.

## 5. Precision: why FP32 was selected

Earlier AMP FP16 experiments logged gradient overflows and skipped updates; the team performed a short FP32 diagnostic demonstrating finite gradients and both XLS-R/AASIST parameter groups receiving gradients. The completed training switched to `training.precision = "fp32"`.

Do not switch to FP16 on a future run solely to accelerate training without an equivalent documented overflow/quality check. Best model **storage** in half precision (see [checkpoints](checkpoints.md)) is not the same as AMP **training**.

## 6. Safe, explicit launch command (intentionally starts a new run)

**Do not use this command to restore a session or generate a submission.** It starts new training when `--resume` is not supplied.

```bash
/kaggle/working/rtcsdd-baseline-py310/bin/python \
  -m rtcsdd.training.xlsr_aasist_finetune \
  --root /kaggle/working/RTC-SDD-Research \
  --dataset-root /kaggle/input/datasets/a7medlabib/rtcfake-packed-private \
  --checkpoint /kaggle/working/checkpoints/pretrained/xlsr2_300m.pt \
  --config /kaggle/working/RTC-SDD-Research/src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json \
  --out /kaggle/working/RTC-SDD-Research/outputs/experiments/new_unique_run_name \
  --epochs 1 --max-minutes 0
```

To intentionally continue a **previous run** instead, ensure the full `last.pt` and matching `config.json` manifest exist in the intended output folder, then use **that** `--out` and `--resume`. Do **not** use a best model-only checkpoint as a resume snapshot. Preserve original dataset/pretrained paths or adjust the manifest/restore process in a formally reviewed way; the trainer explicitly checks path and fingerprint consistency.

## 7. Checkpoint policy

- `last.pt` stores **complete** optimizer/state/RNG information for true resume.
- `best_dev_loss.pt` uses **minimum weighted Dev loss**.
- `best_online_f1.pt` uses **maximum clean Online Dev Macro-F1**.
- `dev_epoch_001.json` contains epoch 1's full aggregate metrics and subgroups.
- Model-only best snapshots are stored with floating weights cast to FP16 to conserve space; the run itself was FP32.
- The JSON config fingerprint locks training-affecting settings; changing such settings can invalidate an exact resume.
- Atomic saving requires temporary disk headroom. A checkpoint failure due to disk space is not evidence of bad model performance.

## 8. Interpreting the class weights

The class ID order is `[spoof, bonafide]`, so `[0.1, 0.9]` puts a **9:1 per-example loss weight ratio** in favor of bonafide. The full Dev set has many more Spoof than Bonafide examples. This weighting is a plausible contributor to the strong bonafide bias seen at threshold 0.5, but it is **not proven causal** by one run. A future experiment should compare weights with **everything else controlled**, including seed and training budget.

## 9. Post-training policy

The completed run should first be evaluated and documented. There is no reason to re-execute warmups or restart Epoch 1 after a Kaggle session reset. Restore a model-only best checkpoint and run held-out inference **without gradient updates**. Further epochs or different training settings should only start as deliberately named, controlled experiments after reviewing Dev errors and competition constraints.
