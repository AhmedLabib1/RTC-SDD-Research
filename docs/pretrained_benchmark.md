# RTC-SDD: frozen pretrained backbone screening (pre-fine-tuning)

This exploratory benchmark compares **original, non-spoof-trained public speech backbones** with identical frozen features and a lightweight supervised probe on the **official Train** split. It selects candidates with labeled **Dev** only. No Progress/Final data may be loaded, screened, or analyzed.

## Method

- Reuse `rtcsdd.data.protocol.read_protocol`, `rtcsdd.data.dataset.RTCFakePackedDataset`, and `rtcsdd.eval.metrics.evaluate_subgroups` as-is.
- 16kHz mono, left crop/repeat pad to 64600 samples, no RawBoost at screening.
- Seed 1234, proportional-stratified sampling by view/language/label; default 3200 Train / 1600 Dev. Use identical subset sizes and seed for every model. `0` requests the entire split.
- Pretrained backbone frozen, FP32, mean+std pool across sequence, StandardScaler + fixed `LogisticRegression(C=1)`, trained only on official Train labels.
- Validation uses probability of class 0 (spoof), fixed threshold 0.5; prefer **online Dev Macro-F1**, inspect spoof recall, English/Chinese subgroups, wall-clock and GPU constraints. Report offline Dev as secondary. Subsample screening cannot substitute for full Dev evaluation.
- No trained spoof detector checkpoint, classifier head, or AASIST weights are loaded. A probe is a screening method, not the fine-tuned XLS-R+AASIST P01 architecture; ranking is not a guarantee of downstream fine-tuning success.
- Public speech-pretraining corpora in checkpoint provenance are *historical pretraining* and distinct from participant use of external speech data for training; only allowed public released weights are used.

## Pretrained model candidates / evidence

- `facebook/wav2vec2-base` — original self-supervised LibriSpeech pretrained encoder: https://huggingface.co/facebook/wav2vec2-base
- `facebook/hubert-base-ls960` — original self-supervised HuBERT encoder: https://huggingface.co/facebook/hubert-base-ls960
- `microsoft/wavlm-base-plus` — 94k hours speech pretraining: https://huggingface.co/microsoft/wavlm-base-plus
- `facebook/data2vec-audio-base` — self-supervised speech pretraining, **not** `-960h` ASR fine-tune: https://huggingface.co/facebook/data2vec-audio-base
- `facebook/wav2vec2-xls-r-300m` — 436k hours multilingual self-supervised speech pretraining: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- `microsoft/wavlm-large` — 94k hours speech pretraining, second round: https://huggingface.co/microsoft/wavlm-large

Check the exact public revision and upstream model training lineage before declaring final eligibility. **Exclude** speech deepfake / ASVspoof / anti-spoofing pretrained descendants, even if they share an allowed architecture. See official rules: https://www.junxue.tech/rtc-sdd-challenge/

## Kaggle

Use a Kaggle GPU notebook, enable Internet for downloading publicly available original weights, and attach your authorized private Kaggle RTCFake dataset. Install `transformers`, `scikit-learn`, and `huggingface_hub` in the same environment as the existing repository.

```bash
pip install -e .
pip install 'transformers>=4.44,<5' 'scikit-learn>=1.4,<2' 'huggingface_hub>=0.30,<1'
python -m rtcsdd preflight --dataset-root /kaggle/input/YOUR_DATASET_FOLDER --require-audit-counts
python scripts/benchmark_pretrained_backbones.py --data-root /kaggle/input/YOUR_DATASET_FOLDER --model facebook/wav2vec2-base
```

The `notebooks/04_pretrained_backbone_benchmark.ipynb` notebook runs the models one-by-one and exports JSON results and a CSV summary. Model weights download once through the public Hugging Face Hub. **Never** commit data files, cached checkpoints, extracted embeddings, utterance-level dev predictions, or authorization tokens.

Only after results have been reviewed should we use a separate Kaggle training experiment for shortlisted full fine-tuning. Compare against P01 in a **distinct experiment table**, because P01 is already fine-tuned and the current probe is not.

## Suggested branch

`research/pretrained-backbone-benchmark` — commit benchmark script, notebook, documentation and aggregate JSON/CSV summaries after privacy review. Do not overwrite P01 checkpoints, runs or the main branch.
