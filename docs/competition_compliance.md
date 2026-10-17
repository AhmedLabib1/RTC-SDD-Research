# Competition compliance: research firewall

Authoritative source: https://www.junxue.tech/rtc-sdd-challenge/
(verified 17 Oct 2026; re-check periodically as rules may change).

## Allowed and disallowed examples

| Operation | Decision |
|---|---|
| Train on original RTCFake train audio | Allowed |
| Select models based on labeled RTCFake dev | Allowed |
| Diagnose model failures by listening to labeled dev | Allowed, provided data handling complies |
| Inspect progress/eval audio to infer conditions, speakers or labels | **Forbidden** |
| Tune based on progress-set scores/leaderboard | **Forbidden under no model-selection/tuning rule** |
| Submit predictions on progress to Codabench | Allowed as evaluation, not tuning |
| Use public generic SSL pretrained weights with no external spoof training | Permitted, verify provenance |
| Use checkpoint trained on external anti-spoof datasets | Forbidden |
| Use speech subset of MUSAN for noise augmentation | Forbidden |
| Use PUBLIC non-speech environmental sound for augmentation | Allowed |
| Use local, offline WebRTC-like DSP modules | Allowed if other rules obeyed |
| Send RTCFake audio through Zoom/QQ/WeChat etc. to get augmented audio | Forbidden |
| Create new spoof data using TTS/VC resynthesis | Forbidden |
| Average predictions from independently trained models | Forbidden |
| Average weights of different runs of the same model, single-model inference | Explicitly allowed by FAQ |

**Important:** The FAQ says the evaluation cutoff is always 0.5. We never alter it
for a Codabench submission. In our code a score equal to 0.5 is assigned spoof;
confirm exact tie behavior with the organizer if that corner case matters.

## Programmatic guardrails already in v0.1

- `read_protocol` is strict about labeled train/dev vs unlabeled progress/eval.
- `PackedAudioStore` opens only train/dev archives, never a held-out pack.
- `score-dev` rejects progress or eval protocols.
- `submission` rejects any protocol with labels.
- `submission` enforces one unique score per ID, completeness, bounded numeric scores,
  and exact official protocol order.
- `.gitignore` excludes audio/metadata, checkpoint binaries, private outputs.

Guardrails do not replace human approval. Model weights, licensing and augmentation
provenance require manual verification and documentation in each experiment.

## Dataset redistribution and team access

Hugging Face marks RTCFake as CC BY-NC 4.0, and the challenge repository is gated
to accepted participants. Approval to download is not necessarily permission for
unrestricted redistribution. Keep private Kaggle datasets private, do not share
personal Hugging Face/Kaggle API tokens, and ask organizers if unsure about private
team storage/sharing terms.

## Held-out handling

Only the `submission` path receives progress/eval *IDs* and one-model scores.
Progress and final evaluation audio must never enter any statistics, plots, manual
review, pseudo-label datasets, model-selection loops, or augmentations.
If allowed and available, inference itself operates on these unlabeled audio files;
the scores are passed to Codabench without local label-based feedback.
