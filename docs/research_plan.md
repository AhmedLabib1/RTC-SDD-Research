# RTC-SDD controlled experiment roadmap

## Objective

Maximize generalization from clean/paired offline+online RTCFake training to
**unseen online** RTC speech with additional environmental noise and platform
shift, under the official **single-model and restricted-data rules**.

Official final metric = `0.3 * F1_clean + 0.7 * F1_noisy`, with fixed decision
threshold 0.5. **Neither clean/noisy evaluation subset is exposed as labeled dev.**
Therefore dev-only metrics are proxies, not the official metric.

## Scoreboard principles

Primary local selection metric: **dev ONLINE macro-F1**, because evaluation is
online-only. Mandatory diagnostics: all-dev Macro-F1, EN/ZH online Macro-F1,
class-wise F1, confusion matrix, bonafide recall. In addition, maintain a
**fixed, deterministic, seeded synthetic-noise version of the dev online set**
using only allowed, public NON-SPEECH augmentation sounds. This provides a
*secondary proxy* for robust behavior, **not** a prediction of official noisy F1.

Every candidate must show: seed, data revision, resource provenance, real runtime,
peak GPU VRAM, epochs/steps, checkpoint selection policy, and change versus control.
For promising candidates, run 2–3 seeds **if GPU budget permits**, reporting mean
and range, not only the best lucky run. No progress-set score-based selection.

## Stages & experiments

| ID | Hypothesis (single changed factor) | Controlled comparison | Priority |
|---|---|---|---|
| P00 | The official XLS-R + AASIST can be reproduced | Exact upstream train and inference behavior | Critical |
| P01 | More online focus helps unseen RTC eval | Train on all vs online only; fixed budget and all else | High |
| P02 | Baseline loss weighting is suboptimal | `[.1,.9]` vs `[1,1]` vs inverse-frequency weights | High |
| P03 | RawBoost default may not optimize RTC robustness | baseline algo=5 vs disabled; all else fixed | High |
| P04 | Legal local non-speech noise makes online predictions robust | no noise vs one defined public non-speech noise policy | High |
| P05 | Training crop position affects detection performance | first 4.0375 s vs random 4.0375 s | Medium |
| P06 | Broader window handles long recordings better | 4.0375s vs ~6s; validate VRAM/time and input compatibility | Medium |
| P07 | SSL freeze schedule improves data efficiency | full fine-tune vs staged freeze/unfreeze | Medium |
| P08 | AASIST vs Nes2Net should be decided by controlled data/compute | single AASIST vs single Nes2Net with matched conditions | Medium |
| P09 | Pair-level consistency may resist RTC changes | baseline CE vs CE + embedding consistency for valid train pairs | Exploratory |
| P10 | Same-architecture weight averaging stabilizes performance | one checkpoint vs SWA of same model family | Medium |

**Never** compare two altered factors simultaneously and call the result a causal
ablation. If compute-limited, short-list based on small smoke experiments, then
run a properly budget-matched training comparison.

## Detailed execution order

### Stage 0 — Infrastructure/data (no GPU)

- Verify archive counts, member lookups, labels, sample decoding and performance.
- Pin official repo hash and pretrained checkpoint provenance; document licenses.
- Record Kaggle Python, CUDA, torch, fairseq, GPU memory before touching installs.
- Implement dataset adapter under an `upstream_compatible` flag.
- Verify train/dev separation, labels and 64,600-sample crop/pad parity.
- `pytest` and CI pass; no real gated data in Git.

### Stage 1 — Official baseline and precise dev reports

- One T4 first; mixed precision only if validated, never silently assume it.
- Start with exact official XLS-R+AASIST, RawBoost algo=5, CE `[0.1,0.9]`,
  Adam lr=1e-6, wd=1e-4, 100 max epochs, patience=10, best dev loss.
- Reconcile discrepancy between CLI `--batch_size 32` and parser default 16.
- Reproduce code-side score direction `softmax[:,0]`.
- Report all-dev and **dev online** macro F1 (not just dev accuracy).
- Benchmark throughput; decide on optional DDP after single-GPU correctness.

### Stage 2 — First robust gains (A/B tests)

- `P01`: online-only vs all training. For fair comparison specify whether
  constant epochs or constant optimizer update steps. Online-only has fewer
  unique recordings and may need repeated sampling to match updates.
- `P02`: alter loss weights only; do not change sampler simultaneously.
- `P03`: RawBoost ablation; keep original audio immutable.
- `P04`: add reproducible legal NON-SPEECH noise to a controlled share of batches,
  log noise corpus license and provenance. Do not use MUSAN's speech section.
- Compare dev online F1 and the fixed synthetic noisy-dev proxy.

### Stage 3 — Robustness/architecture only if Stage 2 is solid

- Crop policies, valid pair-level consistency (NO assumed sample alignment),
  Nes2Net single-model alternative, staged SSL fine-tuning, and approved offline
  DSP (codec/AGC/NS/AEC or packet-effect simulation). Document sample rate,
  whether it modifies speech content, and the absence of external speech data.
- Multi-crop test-time score aggregation has **not** been cleared explicitly by
  the FAQ. Do not submit this approach without a precise organizer clarification.
- Avoid premature complex architecture replacement given the challenge schedule.

### Stage 4 — Freeze & reproduce

- Select the best model by DEV-ONLY evidence with measured variance.
- Run controlled confirmation at fixed seeds and train budgets.
- Archive checkpoint SHA256, config, environment, train logs, per-ID dev scores,
  dev reports, augmentation provenance, model architecture and training run IDs.
- Generate held-out submission scores with **one** model only, no inspection.
- Freeze the experiment protocol before final evaluation release.

## Proposed schedule (calendar, subject to actual runtime)

- 17–19 Oct: Stage 0 / reproducible pack validation / upstream parity.
- 20–24 Oct: official AASIST baseline + dev metrics and GPU profiling.
- 25 Oct–01 Nov: P01–P04 controlled experiments.
- 02–08 Nov: best candidates, reproducibility seeds, model freeze.
- 09 Nov: organizers' final evaluation phase begins; inference/submission only.
- 16 Nov: official final submission deadline (follow organizer timezone/AoE).

Dates reflect published timeline, not a guarantee that baseline installation or
training fits the available Kaggle GPU quota.

## Stop/go criteria

STOP before full training if: pack counts wrong, labels reversed, pretrained
weight provenance unresolved, environment fails smoke, data leakage across splits,
held-out evaluation files exposed to training, score direction reversed, or
reproduction fails basic forward/backward and deterministic eval tests.

Advance from baseline to new model ideas only once an official-compatible
checkpoint can generate valid and reproducible dev and (when permitted) held-out
prediction text in the organizer's exact format.
