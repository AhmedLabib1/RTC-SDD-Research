# Code review of organizers' supplied RTC-SDD baseline

Source: RTC-SDD `xlsr_aasist/` and `xlsr_nes2net/` Python files supplied by user
17 Oct 2026. **Source inspection only; not a runtime confirmation.**

## Two systems

- XLS-R 300M SSL frontend + AASIST spectro-temporal graph attention backend.
- XLS-R 300M SSL frontend + Nes2Net backend.

Official upstream reports progress weighted Macro-F1 72.50% for AASIST and
68.49% for Nes2Net (clean/noisy separate results in official README).

## Important implementation details to pin

1. `LABEL_TO_ID = {spoof: 0, bonafide: 1}`. Official `main_eval` outputs
   `softmax(logits)[:, 0]`: *P(spoof)*, not *P(bonafide)*.
2. `SpoofAudioDataset.cut=64600`, `pad_audio`: always use **first 64,600**
   samples when audio longer, repeat waveform to pad if shorter.
3. The official train loader passes `use_rawboost=(mode == 'train')`.
   Default `--algo=5` composes linear/nonlinear convolutive and impulsive noise.
   This is part of the official comparison control, not an optional change by default.
4. `Adam(lr=1e-6, weight_decay=1e-4)` defaults; weighted cross-entropy
   `weights=[0.1,0.9]` for `[spoof,bonafide]`.
5. Saves `best_model.pth` when **dev loss** improves, and early stops after
   10 epochs without better dev loss (defaults). This can disagree with Macro-F1.
6. Training prints dev loss and dev accuracy, not dev Macro-F1 or group metrics.
7. Both SSL model classes have a hardcoded pretrained checkpoint path;
   the script files have placeholder conda env/dataset paths.
8. The SSL feature extractor has a dtype/device check that can force
   `self.model.train()` on a device/dtype change, so eval behavior must be examined
   when introducing AMP or refactoring the loader.
9. The upstream requirements pin old Fairseq/Cython/Torch combinations and use
   Python 3.10 in README. Test the Kaggle runtime before installing a replacement.
10. Their `run_train.sh` requests a batch size of **32**, unlike the Python
    parser default of **16**. The actual CLI must be recorded and pinned.

## Phase 1 integration acceptance criteria

- Full upstream repo pinned by immutable Git commit and attribution.
- Keep original upstream code available as reference; changes in team wrapper
  are diffable. No copied checkpoint paths from someone's machine.
- Compare waveform arrays returned by old extracted-file loader and new ZIP pack
  loader, within a stated numeric tolerance, before changing any preprocessing.
- Build a deterministic short/long/online/offline/en/zh/bona/spoof smoke batch.
- Forward and backward with finite loss, reliable half/full precision choice.
- Resume checkpoint test and end-to-end inference and score validation.
- Exact seed, CLI/config, audio policy, RawBoost algo, batch size, optimizer,
  label mapping and checkpoint SHA256 are recorded.
- Audit dev metrics (overall, online, offline, language intersections).
- Report whether 1 T4 is viable, measured peak GPU memory and time per step.

## Do not prematurely change

- Audio crop/pad strategy.
- RawBoost settings or train-vs-eval augment enablement.
- Model/train loss/optimizer/checkpoint-selection policy.
- Train+dev split semantics.
- Score direction or 0.5 threshold.

All of those become **future distinct experiments** after the first control run.
