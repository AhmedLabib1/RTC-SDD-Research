# Official baseline integration (pending)

Upstream: https://github.com/JunXue-tech/RTC-SDD

The upstream code is NOT vendored into this starter, and this starter must NOT
be represented as a trained replacement. Before integration:

1. Choose immutable upstream Git commit and record it here.
2. Inspect upstream license/attribution and dependency chain (Fairseq etc.).
3. Create a reproducible Python 3.10 compatible environment (check Kaggle).
4. Acquire XLS-R 300M pretrained weights from authorized original source, record
   URL/revision/SHA256 and show no external spoofing-data post-training.
5. Compare the official librosa/extracted-file waveforms to the new ZIP/SoundFile
   data loader, with numerical tolerance.
6. Add a configurable `checkpoint_path` argument rather than hardcoding paths.
7. Preserve faithful upstream baseline path for reproduction and add custom
   experiments behind explicit options/flags.
8. Run required model smoke, full dev evaluator, and checkpoint-resume tests.

Suggested location after integration:

```text
third_party/official_rtc_sdd/   # pinned Git submodule (or explicit vendor commit)
src/rtcsdd/models/             # wrappers and stable API to upstream models
src/rtcsdd/training/           # reusable trainer, sampler, checkpoint callbacks
configs/models/                # YAML for per-model parameters
```

Until these steps succeed, there is intentionally no `train` CLI command.
