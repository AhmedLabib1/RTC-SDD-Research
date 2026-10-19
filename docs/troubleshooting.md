# Troubleshooting Kaggle, Fairseq, Checkpoints and Submissions

This guide covers problems **actually encountered** during the first P01 implementation, plus direct operational checks. Always capture the full traceback and exact runtime paths before changing source code or starting more GPU work.

## 1. Kaggle shows `Failed to save draft`

**Meaning:** the UI could not confirm that the current notebook draft was persisted; execution output and notebook source may be in different states.

**Do:** keep the browser tab open, inspect the Kaggle save status, try `Save Version` if appropriate and **download/export the current `.ipynb`** when possible. Do not assume edits are available after closing or session reset. Confirm the exported notebook includes your final cells. **Saving the notebook UI is distinct from pushing the Git repository.**

## 2. `/kaggle/working` lost files after a new session

**Expected risk:** Kaggle working directories are volatile. Re-clone your source, attach the correct Kaggle datasets, restore the pinned environment/pretrained XLS-R, and download only the needed fine-tuned best checkpoint from private storage. A Progress submission **does not require restoring `last.pt`**.

Never rerun training or warmups merely to reconstruct outputs from a previous completed epoch.

## 3. Patches report `Attach the v0.4 dataset` / `Expected one ... found 0`

**Cause:** old discovery logic searched specific ZIP archive names even when the files were attached as expanded Kaggle datasets with arbitrary names such as `shhhhhhh-5`.

**Do:** inspect `/kaggle/input` for the unique required files. Select the intended source version explicitly, validate that **every** required file exists, then copy; don't use an arbitrary first match. Multiple attached copies of `trainer.py` may include different historical versions. Follow known patch ordering only for historical reconstruction, ideally replace with one reviewed canonical Git commit.

## 4. `ModuleNotFoundError: rtcsdd` or environment mismatch

**Cause:** the notebook kernel and Python 3.10 training venv are different interpreters; editable install may target the wrong one.

**Do:** invoke the virtual environment interpreter explicitly, e.g.:

```bash
/kaggle/working/rtcsdd-baseline-py310/bin/python -m pip show rtcsdd-research
```

The project can be installed in its venv with `uv pip install --python /path/to/python -e /path/to/RTC-SDD-Research`. Confirm `cwd` and repository root paths in `subprocess.run`.

## 5. Fairseq import errors or deprecated NumPy aliases

**Cause:** old pinned Fairseq dependencies expect aliases removed from newer NumPy. The adapter in `src/rtcsdd/models/xlsr_aasist.py` sets required aliases before Fairseq import.

**Do:** use the pinned Python 3.10 environment and the committed model adapter; don't randomly upgrade Fairseq/Torch/NumPy while comparing model runs.

## 6. `[AMP OVERFLOW]`, infinite gradients or skipped steps

**Observed:** early FP16 AMP training ran into overflow or non-finite-gradient issues. The successful P01 used **FP32** with `precision='fp32'`, a batch of 2, and accumulation of 8.

**Do:** verify the config and current trainer, inspect finite gradient diagnostics and ensure both XLS-R/AASIST groups receive gradients. Do **not** interpret a completed setup check as validation of FP16. Any alternative precision needs a measured, controlled run.

## 7. `Resume checkpoint missing`

**Cause:** an old training cell expects `outputs/.../last.pt` in the current working directory; new session no longer has it.

**Do:** if merely generating a submission, load `best_online_f1.pt` instead. If intentionally continuing training, restore the full `last.pt` and matching `config.json`, verify provenance and use `--resume`. Do not substitute model-only `best_online_f1.pt` for optimizer-resume state.

## 8. Config fingerprint / output directory already exists

**Cause:** the trainer safeguards experiment reproducibility. It refuses a new run over an old output directory and rejects resumes with changed training-affecting config/dataset/pretrained paths.

**Do:** use a **new unique output folder** for a new experiment, or preserve original settings and resume correctly. Don't delete checkpoints or manually bypass fingerprints without documenting consequences.

## 9. Disk space insufficient for checkpoint saving

**Cause:** `last.pt` includes optimizer tensors and is much larger than model-only best snapshots; atomic save needs temporary space.

**Do:** inspect `shutil.disk_usage('/kaggle/working')`, delete **only confirmed disposable files**, keep private checkpoint backups, and favor the ~636 MB best model for inference rather than restoring a ~3.8 GB training snapshot.

## 10. `progress.pack` fails ZIP checks

**Cause:** Progress `.pack` is a **TAR.GZ** stream, whereas Train/Dev `.pack` files are ZIP-backed. The first bytes `1f8b08` identify GZIP.

**Do:** use `tarfile.open(PACK, mode='r|gz')` and stream files. Do not assume `zipfile.is_zipfile(PACK)` must return true.

## 11. Progress parser rejects a path like `clean/online/zh/...`

**Cause:** the legacy `read_protocol` expects three components `view/language/file.wav`; actual Progress uses four components `clean_or_noisy/view/language/file.wav`.

**Do:** use the dedicated Progress script, validate the prefix, strip `progress/` from TAR member paths, and preserve the exact textual Progress IDs. Do not silently patch the Train/Dev parser without adding tests.

## 12. Score direction or class order appears reversed

**Truth:** `spoof=0`, `bonafide=1`, the submitted value is **`softmax(logits)[:, 0]`**. Verify this against `eval/dev_evaluator.py` and `eval/metrics.py` before touching the submission code. Do not invert scores without concrete evidence.

## 13. `submission.zip` rejected or unexpectedly low score

**Checks before any new training:**

- ZIP contains **exactly** `scores.txt` at its root, not `submission_progress.txt` or a nested directory.
- There are **30,279** score rows, matching `progress.txt` exactly and in order.
- Each row is `utterance_id P(spoof)`; scores are finite and between 0 and 1.
- The correct checkpoint (epoch/step/hash) was loaded with `strict=True`.
- Inferenced audio used fixed 16 kHz/64,600 samples, no train-only augmentation.
- The model is in `.eval()` and runs under `torch.inference_mode()`.
- A low leaderboard score **does not by itself prove** ZIP corruption, wrong score direction, or a specific training bug.

## 14. `HF_TOKEN missing`

**Cause:** token not connected to the Kaggle notebook or secret disabled. Load through `kaggle_secrets.UserSecretsClient`, and avoid printing secrets. Progress inference **doesn't use HF_TOKEN** once its model file has been restored. Don't add network uploads to slow down score generation.

## 15. Multiple GPU devices but code uses only one

The P01 trainer selects `cuda:0`; seeing two T4 devices in Kaggle does not mean the implementation uses two-GPU data parallelism. Do not assume doubled performance or launch separate processes sharing the same output checkpoint path.

## 16. Where to look first

| Problem area | Primary source |
|---|---|
| Setup/revisions | `scripts/setup/*`, `third_party/official_rtc_sdd.pin.json` |
| Audio packing | `src/rtcsdd/data/{packed,protocol,audio}.py` |
| FP32/AMP | `src/rtcsdd/training/{trainer,config}.py` |
| Checkpoints | `src/rtcsdd/training/checkpointing.py`, `storage/*` |
| Dev metrics | `src/rtcsdd/eval/{dev_evaluator,metrics}.py` |
| Submission validation | `src/rtcsdd/eval/submission.py`, `scripts/progress_inference_p01.py` |

Record the current Git hash and the exact command before editing anything. Preserve a copy of the failing output if it matters for future reproducibility.
