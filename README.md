# RTC-SDD Research Framework (Starter v0.1)

**Purpose:** A team-oriented, reproducible experiment framework for the
[ICASSP 2027 RTC-SDD Challenge](https://www.junxue.tech/rtc-sdd-challenge/).

> **Scope honesty:** This version contains **tested data/pack readers, dev metrics,
> score validation, run manifests, a Kaggle starter notebook, and governance docs**.
> It **does not yet provide an end-to-end trained XLS-R + AASIST model**.
> The organizers' Fairseq checkpoint and the official model/trainer must be integrated
> and verified before anyone claims a reproduced baseline.

## Non-negotiable challenge compliance

- Train/tune only on the official train/dev data.
- **Never train/tune/select models or manually analyze progress/eval examples**.
- No external speech training/fine-tuning data.
- External augmentation audio must come from **public, non-speech** datasets.
- Do **not** send RTCFake samples through live RTC services for augmentation.
- Do not use public anti-spoof weights trained with external spoofing datasets.
- **No prediction ensembling**; each submission comes from one model.
- Weight averaging runs of the same model is allowed under the challenge FAQ.
- The score file must contain `utterance_id P(spoof)` with probabilities in `[0,1]`.
- The evaluation threshold is fixed at **0.5**; don't tune a submission threshold.

See `docs/competition_compliance.md` and the official rules for authoritative wording.

## Repository structure

```text
RTC-SDD-Research-Starter/
  README.md
  pyproject.toml
  .gitignore
  .env.example
  .github/
    workflows/ci.yml
    pull_request_template.md
    ISSUE_TEMPLATE/experiment.yml
  src/rtcsdd/
    cli.py                      # audited data preflight / dev scoring / score validator
    data/
      protocol.py               # strict labels and split parsing
      packed.py                 # lazy ZIP random-access to the 4 .pack files
      audio.py                  # exact 64600-sample baseline crop/repeat policy
      dataset.py                # PyTorch dataset (import lazy)
    eval/
      metrics.py                # fixed threshold, macro-F1, class metrics, subgroups
      submission.py             # official protocol-order safe score writer
    experiments/
      registry.py               # config/git/environment manifest; no model claim
  configs/
    models/                     # proposed architectural adapters
    experiments/               # one-factor-at-a-time hypotheses
  docs/
    competition_compliance.md
    baseline_review.md
    data_contract.md
    research_plan.md
    kaggle_setup.md
    team_workflow.md
  notebooks/
    00_kaggle_preflight.ipynb
  tests/
    test_packs.py
    test_protocol.py
    test_scores.py
  third_party/
    README.md                   # external, revision-pinned baseline integration
```

The official model implementation, training loops, and reusable augmentations should
be added in **separate PRs**, building on this foundation. Dataset archives,
checkpoint binaries, real labels, output scores, and credentials are not committed.

## Installation and safe local tests

Python 3.10+ is sufficient for **this infrastructure** (the official Fairseq baseline
may have narrower compatibility constraints).

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Kaggle data preflight (no training)

The four team-created `.pack` files are actually ZIP_DEFLATED archives with paths
such as `offline/en/Train_000000.wav` inside. The reader opens each archive lazily
and extracts **one member into memory** when requested; it does not unzip all audio.

```bash
DATA_ROOT=/kaggle/input/datasets/a7medlabib/rtcfake-packed-private

rtcsdd preflight --dataset-root "$DATA_ROOT" --require-audit-counts
rtcsdd smoke-audio --dataset-root "$DATA_ROOT"
rtcsdd benchmark-data --dataset-root "$DATA_ROOT" --samples 128 --workers 0
rtcsdd benchmark-data --dataset-root "$DATA_ROOT" --samples 128 --workers 2
```

The expected audit counts are: train=75,785, dev=15,204, split into
train offline=38,660, train online=37,125, dev offline=7,739,
dev online=7,465. If the uploaded Kaggle dataset does not match,
**STOP** before training and reconcile the dataset versions.

## Local dev scoring (after a model is integrated)

Input score format: `offline/en/Dev_000000.wav 0.12345678`.
**0.9 means highly spoof-like**, not bonafide-like.

```bash
rtcsdd score-dev \
  --protocol "$DATA_ROOT/dev_label.txt" \
  --scores outputs/exp00/dev_scores.txt \
  --out outputs/exp00/dev_metrics.json
```

Metrics include dev overall, dev online/offline, English/Chinese, and all four
view-language intersections. Dev has no official *noisy* group, so the **official
weighted clean/noisy final score cannot be computed from dev**.

## Codabench file validation (after inference on the unlabeled protocol)

```bash
rtcsdd submission --split progress \
  --protocol "$DATA_ROOT/progress.txt" \
  --scores outputs/exp00/progress_raw_scores.txt \
  --out outputs/exp00/progress_submission.txt
```

This command only validates/arranges **predictions already produced by one model**;
it does NOT run inference, nor view/analyze held-out audio. Pass the prepared
score file to Codabench; consult Codabench UI for any upload container requirements.
The final-eval protocol can be used with `--split eval` when the organizers release it.

## Experiment manifest

```bash
rtcsdd register \
  --config configs/experiments/exp00_official_reproduction.yaml \
  --runs-root runs
```

This records the config hash, Git SHA, dirty working tree flag, environment, and
UTC timestamp; it is clearly marked **registered_not_trained**. Future trainer
integration must add checkpoint hash, actual data revision, metrics, durations,
software/dependency versions, and completed status.

## Recommended next work

1. Kaggle preflight + representative sample reads and timing benchmark.
2. Pin a precise upstream RTC-SDD Git commit and acquire approved XLS-R checkpoint.
3. Integrate their unmodified AASIST baseline with the pack loader and no change
   in processing or loss; include short/long-clip forward/backward GPU smoke test.
4. Reproduce training on one T4, preserve full config/seed/checkpoint provenance.
5. Add dev predictions, fixed 0.5 threshold and `online` subgroup reporting.
6. Only then start controlled experiment changes from `docs/research_plan.md`.

## Source references

- Official rules: https://www.junxue.tech/rtc-sdd-challenge/#rules
- Official FAQ: https://www.junxue.tech/rtc-sdd-challenge/#faqs
- Official baseline: https://github.com/JunXue-tech/RTC-SDD
- Authorized RTCFake: https://huggingface.co/datasets/JunXueTech/RTCFake

**Gated-data reminder:** The official dataset is restricted to approved registered
teams. The private Kaggle copy should not be made public. Check authorization and
redistribution conditions before sharing it with teammates or any outside account.
