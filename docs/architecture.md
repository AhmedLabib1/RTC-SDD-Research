# System Architecture and Code Map

## 1. Purpose

This project builds a **single-model speech deepfake detector** that scores each utterance with its estimated probability of being synthetic/manipulated (`spoof`). The first implemented model uses the official **XLS-R 300M + AASIST** architecture, a pinned upstream implementation, and a Kaggle-friendly first-party training/evaluation pipeline.

A working pipeline must keep three distinct concerns separate:

1. **Training and Dev** use labeled, authorized RTCFake data and may update or select model weights.
2. **Progress inference** uses held-out, unlabeled audio solely to generate predictions in the organizer's format.
3. **Experiment governance** records provenance, environment, checkpoint lineage, and compliance constraints so that published metrics remain auditable.

## 2. End-to-end pipeline

```text
TRAIN / DEV (authorized RTCFake)           PROGRESS (held-out)
  train_label.txt / dev_label.txt            progress.txt
  train_*.pack / dev_*.pack                  progress.pack (TAR.GZ)
              |                                   |
              v                                   v
    Strict protocol validator               ID list, no labels
              |                                   |
    PackedAudioStore (ZIP)                 Streaming TAR reader
              |                                   |
     soundfile, mono, 16 kHz                soundfile, mono, 16 kHz
              |                                   |
      fixed 64,600 samples                  fixed 64,600 samples
              |                                   |
       Train-only RawBoost 5                         |
              |                                   |
              +-----------------+-----------------+
                                |
                     XLS-R 300M backbone
                                |
                       AASIST model head
                                |
                   2 logits [spoof, bonafide]
                                |
                +---------------+-----------------+
                |                                 |
         weighted CE, Adam                 softmax[:, 0]
         backward, FP32                    P(spoof) in [0,1]
                |                                 |
           last.pt / best*                  scores.txt in protocol order
          full Dev evaluation                    |
                |                          submission.zip
        local diagnostics
```

The diagram is conceptual. The implementation applies RawBoost to Train examples **before** cropping/padding; it is disabled for Dev and Progress. No full archive extraction is needed for inference.

## 3. Model implementation

The adapter in `src/rtcsdd/models/xlsr_aasist.py` dynamically imports the pinned `xlsr_aasist/model/model.py` from the upstream checkout. It redirects the expected checkpoint path to the available pretrained Fairseq `xlsr2_300m.pt` only during construction and then returns the model on the selected CUDA device.

- Backbone: **XLS-R 300M**, learned speech representation.
- Anti-spoofing head: **AASIST**, produces two class logits.
- Class 0 = **spoof**; class 1 = **bonafide**.
- Submission score = `torch.softmax(logits.float(), dim=-1)[:, 0]`.
- Both backbone and head were **trainable** in P01. No ensemble is used.

The adapter handles deprecated NumPy aliases required by the pinned Fairseq, rather than editing the upstream repository checkout.

## 4. Module responsibilities

| Module | Responsibility | Inputs / outputs |
|---|---|---|
| `data/protocol.py` | Strict labeled Train/Dev ID parsing; legacy held-out parser limitations | `.txt` → `Example` records |
| `data/packed.py` | Lazy ZIP reader for Train/Dev | `Example` → float32 waveform |
| `data/audio.py` | Deterministic crop/repeat-pad | waveform → length 64,600 |
| `data/rawboost.py` | Official RawBoost Algorithm 5 adapter | Train waveform → augmented waveform |
| `data/dataset.py` | PyTorch Train/Dev dataset | `(audio, label, uid)` |
| `models/xlsr_aasist.py` | Instantiate official model with XLS-R checkpoint | model on GPU |
| `training/config.py` | Validate config / fingerprint training dynamics | JSON → config dict |
| `training/weighted_loss.py` | Weighted CE numerators, gradient normalization | logits/labels → train loss and norm |
| `training/trainer.py` | FP32/AMP training loop, resume and checkpoint selection | dataset + model → checkpoints/Dev reports |
| `training/checkpointing.py` | Atomic checkpoints and compact best snapshots | `.pt` binary checkpoints |
| `eval/dev_evaluator.py` | Full unaugmented Dev inference | metrics and loss |
| `eval/metrics.py` | Fixed threshold and subgroup confusion | per-ID scores → diagnostics |
| `eval/competition.py` | Online metric and optional synthetic noisy-Dev proxy | Dev-only metrics |
| `eval/submission.py` | Validate two-column score text, preserve protocol order | `scores.txt` |
| `storage/*` | Optional private checkpoint backup/restore | HF model repo files |
| `scripts/progress_inference_p01.py` | TAR.GZ Progress stream + local score resume | unlabeled Progress → ordered submission text |

**Important implementation gap:** `data/protocol.py` and `PackedAudioStore` in the reference v0.5 code were built around **three-part** Train/Dev paths and ZIP archives. The actual Progress protocol has **four-part** IDs (e.g. `clean/online/en/...`) and the archive is a TAR.GZ with a `progress/` prefix. The Kaggle inference script explicitly handles that difference; treating the existing generic parser/pack reader as a universal Progress reader would be incorrect.

## 5. Train and inference interfaces

### Training

Entry point: `python -m rtcsdd.training.xlsr_aasist_finetune`.

Required command arguments: `--root`, `--dataset-root`, `--checkpoint`, `--out`. Recommended explicit `--config` with the active JSON. Supported operational overrides include `--epochs`, `--batch-size`, `--accumulate`, `--workers`, `--max-minutes`, and `--resume`.

### Development evaluation

`eval/dev_evaluator.py` returns a score map `{utterance_id: P(spoof)}` plus weighted CE and calls `evaluate_subgroups`. It computes the metrics at the **fixed 0.5 threshold**, including English/Chinese and Online/Offline groups.

### Progress inference

The standalone Kaggle script loads the **best model-only checkpoint**, streams the TAR.GZ, batches audio into GPU inference, persists a local score map, and creates a protocol-ordered submission text file. The ZIP packaging step is separate. See [progress_submission.md](progress_submission.md).

## 6. Checkpoints and lifecycle

- `last.pt`: **full training resume** including model, Adam optimizer, GradScaler state (possibly disabled), RNG state, epoch/batch/step counters, and fingerprint.
- `best_online_f1.pt`: **model-only** snapshot, selected by Online Dev Macro-F1.
- `best_dev_loss.pt`: **model-only** snapshot, selected by lowest weighted Dev loss.
- The model-only snapshot stores floating-point weights as **FP16 on disk** and loads into the instantiated model for inference. This does **not** mean that FP16 training was used; the successful P01 training used FP32.

See [checkpoints.md](checkpoints.md) for exact limitations and restore procedures.

## 7. Design constraints

- Keep the pinned official implementation fixed for comparison.
- Keep labels/score direction invariant: `spoof=0`, `bonafide=1`, score=P(spoof).
- Keep hidden Progress/Eval out of training, validation, hyperparameter search, and threshold selection.
- Keep Dev clean and labeled; if building a noisy Dev proxy, mark it as a **synthetic proxy** rather than a hidden challenge score.
- Keep first-party code and configs version-controlled, binaries/audio private, credentials in secrets.
- Record training-affecting changes in the experiment manifest; a passing smoke test does not establish model quality.
