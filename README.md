# RTC-SDD Research

Reproducible experiment framework for the [RTC-SDD speech deepfake challenge](https://www.junxue.tech/rtc-sdd-challenge/).

**Current status:** CPU dataset validation, fixed-threshold dev metrics, submission checks and experiment registration work. The official XLS-R + AASIST **training/inference integration has not been implemented or reproduced**. No leaderboard results are claimed.

## Layout

```text
RTC-SDD-Research/
├── configs/                # Model and experiment YAMLs (future training runners)
├── src/rtcsdd/             # Pack loader, evaluation, CLI, run registry
├── scripts/setup/          # Pinned upstream checkout, env check, CPU validation
├── notebooks/              # Thin Kaggle launchers, not training implementations
├── tests/                  # Automated checks
├── third_party/            # Pin for external source; checkout is untracked
└── docs/
    ├── setup.md            # Kaggle setup / validation
    └── experiments.md      # Experiment plan and competition constraints
```

## Start

1. Read [docs/setup.md](docs/setup.md) to attach your **private, approved** RTCFake pack dataset and set up Kaggle CPU mode.
2. Run `notebooks/00_research_setup.ipynb` and confirm its final CPU verification passes.
3. Read [docs/experiments.md](docs/experiments.md) before proposing an experiment or collecting held-out scores.

On a local machine, infrastructure checks require only:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

`rtcsdd --help` lists the **implemented** utilities: `preflight`, `smoke-audio`, `benchmark-data`, `score-dev`, `submission`, `register`. There is **no working train or predict command yet**. YAML experiment configs are research proposals, not executable training jobs.

**Never commit or publicly share** gated audio, labels, personal access tokens, pretrained weights, or training outputs. The official code comes from an immutable revision recorded in `third_party/official_rtc_sdd.pin.json`; it is restored by `scripts/setup/bootstrap_official.py` and excluded from Git.

Official resources: [rules](https://www.junxue.tech/rtc-sdd-challenge/#rules) · [FAQ](https://www.junxue.tech/rtc-sdd-challenge/#faqs) · [baseline code](https://github.com/JunXue-tech/RTC-SDD) · [dataset](https://huggingface.co/datasets/JunXueTech/RTCFake).
