# Team experiment protocol

## GitHub policy

- Private main repo initially; default branch is protected and PR-required.
- Research code only in `src/`; actual notebooks **launch scripts**.
- One PR per hypothesis or bugfix, tests required. No silent modifications to
  baseline. Vendor/pin official baseline separately with attribution.
- All approved member changes are reviewed by another member, ideally by
  the owner of the data, evaluation or training system impacted.
- Never force-push `main`, commit gated audio, plaintext tokens, or huge weights.

## Roles (adapt to actual team size)

| Owner | Scope | Deliverables |
|---|---|---|
| Research lead / reproducibility | Controls, prioritization, compliance, code review | Registered experimental comparisons, verified paper logs |
| Data / platform engineer | Archive reader, path integrity, provenance, Kaggle scripts | Preflight, streaming speed, environment |
| Baseline / model engineer | XLS-R+AASIST integration, weight provenance | Verified forward/backward, baseline checkpoint |
| Robustness researcher | RawBoost, legal non-speech noise, local DSP | Augmentation ablations and provenance |
| Evaluation / quality engineer | Macro-F1, score files, subgroup error analysis | Dev report, Codabench-ready score file |

Several people can share a role; roles above are *ownership areas*, not a
claim about your team's size.

## Required experiment record

1. **Hypothesis:** concrete explanation of why one factor should matter.
2. **Control ID:** compare with a known, pinned baseline configuration.
3. **Independent variable:** precisely *one* changed factor.
4. **Allowed data/assets:** origin/license; no external speech or anti-spoof weights.
5. **GPU budget:** device count, epochs or optimizer-step budget, wall time.
6. **Environment:** Git SHA, Python/PyTorch/Fairseq/CUDA versions, checkpoint SHA.
7. **Data protocol:** exact train/dev version + online/offline distribution.
8. **Training:** seed, optimizer, LR, batch/grad-accumulation, preprocessing,
   sampling, augmentation, loss, checkpoint policy and failure recovery.
9. **Results:** dev overall, dev online (primary), dev offline, en/zh, per class,
   confusion; fixed synthetic-noise diagnostic if used, with fixed noise seed.
10. **Conclusion:** expected-vs-observed, caveats, next experiment.

## Example report table

| Run ID | Change | Seed | Dev ONLINE Macro-F1 | Dev offline | Synthetic noisy-dev | GPU hours | Status |
|---|---|---|---|---|---|---|---|
| P00-S1234 | Official baseline | 1234 | TBD | TBD | TBD | TBD | not run |
| P03-S1234 | Disable RawBoost | 1234 | TBD | TBD | TBD | TBD | not run |

**Do not fill TBD with organizer-held-out leaderboard scores.** Don't rank
candidates on progress or final evaluation outcomes.

## Daily cadence

- Before runs: write/configure experiment + PR review + environment check.
- During runs: log full trace and handle preemptions/checkpoint rotation.
- After runs: record results and hypothesis outcome, push code/config, upload
  private checkpoint to approved location, link immutable artifact hashes.
- Weekly: choose only dev-justified research directions and revise compute budget.

## Reproduction package for paper/ranking audit

- README covering environment, authorized data access steps, fixed versions.
- Git commit and documentation of every external resource, pretrained SSL
  checkpoint provenance and license, precise data protocol, full config.
- Setup scripts, deterministic evaluation and inference, score-file validator.
- At least one trained checkpoint plus checksum in authorized storage.
- Logs and dev results plus exact single-model submission score file.
- Compliance declaration and realistic non-determinism limitations.
