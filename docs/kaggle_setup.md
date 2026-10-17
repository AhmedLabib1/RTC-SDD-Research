# Kaggle operations playbook

## Golden rule

**Kaggle is a temporary worker, Git is the source of truth.** Never make a
training notebook the only copy of the model implementation. Store code in
`src/`, configs in `configs/`, and keep notebooks as thin launchers.

## Before each Kaggle run

1. Create a PRIVATE notebook and attach only the APPROVED PRIVATE RTCFake pack dataset.
2. Select `GPU T4 x2` in Kaggle only when needed; one T4 is enough for preflight.
3. Clone or add your team's GitHub repo at a pinned commit.
4. Run `!python --version`, `!nvidia-smi`, `!python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'`.
5. Check that the Kaggle runtime supports the **old Fairseq** requirements before
   any full model install; Python 3.10 was used in official README but notebooks
   may run another Python version.
6. `!python -m pip install -e '.[dev]'` in the cloned repo.
7. Run tests, pack preflight, 16-category decode smoke, and a dataloader
   throughput benchmark before a GPU model smoke test.
8. Run `rtcsdd benchmark-data --dataset-root ... --samples 128 --workers 2`
   to measure actual ZIP throughput without a GPU training step.
9. Record Git SHA, framework version, archived pretrained checkpoint SHA256,
   dataset version, resolved configs and dependency list.

## Recommended notebook skeleton

```python
import os, sys, subprocess
DATA_ROOT = '/kaggle/input/datasets/a7medlabib/rtcfake-packed-private'
print(sys.version)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'rtcsdd', 'preflight',
                '--dataset-root', DATA_ROOT, '--require-audit-counts'], check=True)
subprocess.run([sys.executable, '-m', 'rtcsdd', 'smoke-audio',
                '--dataset-root', DATA_ROOT], check=True)
```

## Practical model training

- Begin on `cuda:0` and measure peak VRAM for XLS-R 300M + AASIST.
- Start with a tiny subset, batch size 1 or 2, and validate forward/backward.
- Use AMP (fp16/autocast) only once stability and parity are verified; dynamic
  loss scaling may be needed on T4 for stable fp16.
- Grad-accumulation can increase **effective batch size**, not physical memory.
  Save the exact `physical_batch`, `accum_steps`, and `global_batch`.
- Only use `torchrun --standalone --nproc_per_node=2` after DDP is implemented,
  proven correct, and tested on two GPUs. `cuda:0` does not magically use 32GB.
- Never trust default Kaggle package upgrades without recording resolved versions.
- Check training time and memory budget with empirical smoke steps before
  committing to 100 epochs; avoid promises about completion time.

## Data access performance

ZIP random access avoids extraction, but `zipfile.read` decompresses each WAV
on each access. Benchmark `num_workers` 0, 2, 4; ZIP concurrency, CPU and I/O
may bottleneck. If performance is unacceptable, build a private **ephemeral**
working-data cache only for train/dev, with quota and privacy compliance.
Do not include progress/eval files in the cache or audit pipeline.

## Checkpoint persistence

`/kaggle/working` can be ephemeral across sessions. Save each best checkpoint
and run manifest as a PRIVATE versioned Kaggle Notebook output or approved
private team storage and verify restored SHA256. Push code/configs to GitHub,
not binary checkpoints to Git. Preserve lineage: `run_id → Git SHA → config →
checkpoint hash → dev metrics → submitted score-file hash`.

## Team dataset access

Do NOT share a personal Kaggle API token. Grant authorized teammates private
dataset access where Kaggle supports it, or have each approved member request
access to the official HF source. Ensure organizer gating/redistribution terms
allow the chosen storage/sharing method.
