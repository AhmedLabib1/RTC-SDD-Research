# Setup — Kaggle (CPU first)

**Inputs:** your approved private `rtcfake-packed-private` dataset and a Kaggle private dataset containing *this repository's code*. Keep the accelerator **None** for this notebook. Do not store tokens in code.

## One-time CPU bootstrap

1. Attach both private datasets to the notebook.
2. Open `notebooks/00_research_setup.ipynb` and run cells top-to-bottom. The notebook locates/copies the repository to `/kaggle/working/RTC-SDD-Research` and installs the infrastructure package in editable mode.
3. The notebook creates an isolated Python 3.10 environment with `uv`, restores the pinned official source under `third_party/official_rtc_sdd/`, and runs `scripts/setup/verify_setup.py`.

The final CPU verification confirms: 8 unit tests, pinned/clean upstream checkout, install **plan** consistency, membership of 90,989 train/dev protocol IDs in four `.pack` files, and representative decoding across train/dev, offline/online, en/zh, spoof/bonafide. It does **not** decode every WAV, prove train-loader parity, or execute XLS-R.

Manual equivalent after mounting/copying and installing the package:

```bash
python scripts/setup/bootstrap_official.py
python scripts/setup/install_training_env.py \
  --python /kaggle/working/rtcsdd-baseline-py310/bin/python
python scripts/setup/verify_setup.py \
  --dataset-root /kaggle/input/datasets/a7medlabib/rtcfake-packed-private \
  --python310 /kaggle/working/rtcsdd-baseline-py310/bin/python
```

**Data:** `.pack` is a ZIP archive containing WAVs, read lazily. The expected train/dev counts are 75,785 / 15,204, split train offline/online 38,660/37,125 and dev offline/online 7,739/7,465. Audio is 16 kHz mono. Label mapping: **spoof=0, bonafide=1**. The current baseline audio policy is first 64,600 samples or repeated padding; parity with the organizers' loader must still be proved.

## Official training environment (not yet validated)

The upstream requirements pin PyTorch 2.5.1+cu121, NumPy 1.26.4, Hydra 1.0.7 and Fairseq by Git revision. The installer provides a **check-only** default; `--install` is an unverified installation path, for a separately approved GPU/runtime session. The GPU notebook must test Python/Fairseq/Hydra compatibility, XLS-R checkpoint provenance and SHA256, actual CUDA imports, forward/backward, VRAM, training-resume and inference parity **before** a full training run.

Kaggle `/kaggle/working` is temporary. Keep code/configs in the private GitHub repository, checkpoint binaries in approved private storage, and rerun setup on a new session. Do not use a token in a notebook output or share the private dataset beyond permitted team access.
