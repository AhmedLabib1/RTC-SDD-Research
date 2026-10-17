# Experiments and rules

## Research question

Can we make a **single** speech-deepfake detector more robust to previously unseen RTC artifacts and noise? The official score weighs clean/noisy Macro-F1 **0.3 / 0.7**, with fixed **0.5** cutoff. For local model choice use labeled **dev online Macro-F1**, plus per-class/EN/ZH/offline diagnostics. A seeded synthetic noisy-dev split may be used as a **secondary** proxy, not a surrogate for the official held-out score.

## Non-negotiable constraints

- Train/select/tune on the permitted **train/dev** data only. No progress/final audio inspection, training, tuning, pseudo-labeling or selection based on leaderboard feedback.
- No extra speech training data or external anti-spoof checkpoints; document provenance of permissible generic SSL weights. Only compliant **non-speech** external audio for augmentation.
- Do not upload/stream the official audio to live RTC services or generate extra spoof speech by resynthesis.
- **No prediction ensembling.** Same-model weight averaging is described as permitted in the FAQ; verify exact eligibility before use.
- Each submission must use one-model `P(spoof)` scores in `[0,1]`. `spoof=0`, `bonafide=1`; don't tune the official 0.5 threshold. Re-check [rules](https://www.junxue.tech/rtc-sdd-challenge/#rules) and [FAQs](https://www.junxue.tech/rtc-sdd-challenge/#faqs) before submission.

## Experiment queue

| ID | Hypothesis / controlled change | Priority |
|---|---|---|
| P00 | Reproduce *unmodified* XLS-R 300M + AASIST baseline; fixed upstream crop, RawBoost, loss, checkpoint policy | First |
| P01 | Online-only versus all training data (match optimizer-step budget) | High |
| P02 | Class weights `[0.1,0.9]` vs uniform / inverse-frequency | High |
| P03 | RawBoost algo 5 on versus off | High |
| P04 | Allowed non-speech noise augmentation versus none | High |
| P05 | First-window versus random-window training crop | Medium |
| P06 | 4.0375 s versus longer audio window | Medium |
| P07 | Full fine-tuning versus staged XLS-R freeze/unfreeze | Medium |
| P08 | AASIST versus Nes2Net, separately (no ensemble) | Medium |
| P09 | Offline/online-pair representation consistency (no assumed sample alignment) | Exploratory |
| P10 | Allowed same-model-family weight averaging | Medium |

**Order:** first verify audio input parity, pretrained checkpoint SHA, forward/backward, dev metrics and checkpoint resume; then complete P00. Start controlled ablations only after a trustworthy baseline. Changing multiple factors at once is not a causal ablation.

## Every experiment must record

`experiment_id`, control/config, hypothesis, data/weight provenance, Git and upstream commit SHAs, Python/CUDA/Fairseq versions, checkpoint SHA256, seed, sample/crop/augmentation and loss settings, budget (steps/epochs/GPU hours), dev online Macro-F1 and subgroups, and final conclusion. Prefer 2–3 seeds for promising runs if compute permits. Keep private outputs outside Git. Every change comes through a reviewed PR and tests.

**Baseline source notes:** upstream uses 64,600-sample first crop, repeat padding, RawBoost for train, weighted CE `[0.1,0.9]`, Adam LR `1e-6` weight decay `1e-4`, and best checkpoint by dev loss. Train shell example uses batch 32, while parser defaults may differ. Confirm exact flags in the pinned source and avoid silent behavior changes. The official score cannot be computed from the normal dev labels alone.
