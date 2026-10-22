# Full pretrained-backbone benchmark (v2)

Scope: Complete RTCFake Train (75,785) and Dev (15,204) **only**; no Progress or Final. Original speech pretrained weights with frozen backbone and unweighted logistic regression probe. Local Dev threshold is 0.5 and score means P(spoof). No fine-tuning.

## Where

`notebooks/05_full_pretrained_backbone_benchmark.ipynb` for Kaggle; `scripts/benchmark_pretrained_backbones_full.py` for direct execution. This adds files to the existing framework; does not modify P01.

## Command

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_pretrained_backbones_full.py \
 --data-root /kaggle/input/datasets/a7medlabib/rtcfake-packed-private \
 --model facebook/wav2vec2-base --train-samples 0 --dev-samples 0 \
 --batch-size 2 --chunk-size 256 --workers 2 --output-dir /kaggle/working/full_benchmark
```

`0` means **complete split**. `--chunk-size` sets resumable feature-file size, **not** sampling. One `.npz` cache per chunk is atomically written, with fingerprint, IDs and labels validated on resume. `result.json` only appears after complete Train/Dev extraction and probe fit. The notebook runs one model at a time, on only GPU 0, and reports failures without destroying other model progress.

Kaggle `/kaggle/working` outputs may disappear across sessions. To resume across sessions, preserve/copy the entire private `full_benchmark/` directory as a Kaggle notebook saved output or private Kaggle dataset, then restore to the same output path. Do **not** add embeddings/chunks to public GitHub; derived private-audio representations may be sensitive. Large `.npz` caches consume significant storage. Preserve `result.json` and summary in GitHub after checking that publication is permitted.

## Fairness and limitations

All five models get identical deterministic input waveforms, preprocessing and sample IDs, with frozen encoders, mean/std pooling, unweighted logistic regression, and fixed 0.5 threshold. Model architectures differ, so frozen-probe scores rank the accessible representations under this protocol, **not** guaranteed fine-tuning outcomes. Original checkpoint lineage and challenge eligibility need human model-card audit; allowlist alone is not certification. Check all failures before drawing conclusions. Checkpoint revision SHA is included in every feature cache fingerprint to prevent mixing versions. Main scoring is local Dev Online macro-F1, alongside Offline and overall scores and class recalls. Official Codabench 30% clean + 70% noisy is unavailable from these labels.

Resource warning: an initial 480-recording smoke test took ~576 seconds on a Kaggle T4. Full 90,989 recordings across five backbones could require many GPU-hours. Never claim a completion time. Kaggle sessions have limits; resume is valid only for persisted files.
