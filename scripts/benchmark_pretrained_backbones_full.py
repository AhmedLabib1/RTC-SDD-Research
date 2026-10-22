"""Resumable, full-RTCFake frozen-backbone benchmark (train/dev only).

Extracts feature chunks atomically, validates every cached chunk, and runs a
fixed unweighted logistic-regression probe after complete feature extraction.
Never reads Progress/Final. Outputs local Dev metrics, NOT Codabench scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtcsdd.data.dataset import RTCFakePackedDataset
from rtcsdd.data.protocol import read_protocol
from rtcsdd.eval.metrics import evaluate_subgroups

CANDIDATES = (
    "facebook/wav2vec2-base",
    "facebook/hubert-base-ls960",
    "microsoft/wavlm-base-plus",
    "facebook/data2vec-audio-base",
    "facebook/wav2vec2-xls-r-300m",
)
PROTOCOL_VERSION = "frozen_mean_std_linear_probe_full_v2"


def manifest_sha(items):
    return hashlib.sha256("\n".join(e.utterance_id for e in items).encode()).hexdigest()


def select_examples(items, limit, seed):
    if limit == 0 or limit >= len(items):
        return list(items)
    groups = defaultdict(list)
    for e in items:
        groups[(e.view, e.language, e.label)].append(e)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    quotas = {k: limit * len(v) / len(items) for k, v in groups.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    for k in sorted(groups, key=lambda k: (quotas[k] - counts[k], str(k)), reverse=True):
        if sum(counts.values()) >= limit:
            break
        counts[k] += 1
    selected = {e.utterance_id for k, group in groups.items() for e in group[:counts[k]]}
    return [e for e in items if e.utterance_id in selected]


def atomically_save_npz(path, features, labels, ids, fingerprint):
    # No pickle: all metadata is simple unicode/int arrays.
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as f:
        np.savez(f, features=features, labels=labels,
                 ids=np.asarray(ids, dtype=str), fingerprint=np.asarray(fingerprint))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_chunk(path, expected, fingerprint):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            x = data["features"]
            y = data["labels"]
            ids = data["ids"].tolist()
            sig = str(data["fingerprint"].item())
        if (sig != fingerprint or ids != [e.utterance_id for e in expected]
                or y.tolist() != [e.label for e in expected]
                or x.ndim != 2 or len(x) != len(expected)
                or not np.isfinite(x).all()):
            raise ValueError("cached data does not match current benchmark")
        return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False), ids
    except Exception as exc:
        # Never silently recompute potentially incompatible outputs.
        raise RuntimeError(f"Invalid cached chunk {path}: {exc}; remove after inspection") from exc


def extract_chunk(items, root, model, batch_size, workers, device):
    import torch
    from torch.utils.data import DataLoader

    dataset = RTCFakePackedDataset(items, root, rawboost={"enabled": False})
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    embeddings, labels, ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for waves, y, batch_ids in loader:
            hidden = model(input_values=waves.to(device, non_blocking=True)).last_hidden_state
            pooled = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False)), dim=1)
            embeddings.append(pooled.float().cpu().numpy())
            labels.extend(y.tolist())
            ids.extend(batch_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.concatenate(embeddings, axis=0), np.asarray(labels, dtype=np.int64), ids


def build_fingerprint(args, checkpoint_sha, train, dev):
    manifest = {
        "protocol": PROTOCOL_VERSION, "model": args.model, "revision": checkpoint_sha,
        "train_sha": manifest_sha(train), "dev_sha": manifest_sha(dev),
        "seed": args.seed, "crop": "left", "pad": "repeat", "audio_samples": 64600,
        "pooling": "mean_std", "precision": "fp32",
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def run(args):
    import torch
    from huggingface_hub import model_info
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel

    if args.model not in CANDIDATES:
        raise ValueError("Checkpoint not on screened candidate allowlist")
    if min(args.train_samples, args.dev_samples) < 0 or min(args.batch_size, args.chunk_size) < 1:
        raise ValueError("Invalid sample/batch/chunk counts")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))

    data_root = Path(args.data_root)
    train = select_examples(read_protocol(data_root / "train_label.txt", "train"), args.train_samples, args.seed)
    dev = select_examples(read_protocol(data_root / "dev_label.txt", "dev"), args.dev_samples, args.seed)
    if set(x.utterance_id for x in train) & set(x.utterance_id for x in dev):
        raise ValueError("Train/Dev IDs overlap")
    if {e.label for e in train} != {0, 1} or {e.label for e in dev} != {0, 1}:
        raise ValueError("Both classes required")

    # Resolve immutable SHA; never allow cached features from a different checkpoint revision.
    checkpoint_sha = model_info(args.model, revision=args.revision).sha
    fingerprint = build_fingerprint(args, checkpoint_sha, train, dev)
    work = Path(args.output_dir) / args.model.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    final = work / "result.json"
    if final.exists():
        current = json.loads(final.read_text())
        if current.get("cache_fingerprint") == fingerprint:
            print(f"Already complete: {final}", flush=True)
            return
        raise RuntimeError(f"Different benchmark already saved at {final}; use another output directory")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None
    nparams = None
    timings = {"train": 0., "dev": 0.}
    start = time.perf_counter()
    for split, examples in (("train", train), ("dev", dev)):
        files = []
        for offset in range(0, len(examples), args.chunk_size):
            batch_examples = examples[offset:offset + args.chunk_size]
            part = work / f"{split}_{offset:06d}_{len(batch_examples):04d}.npz"
            saved = read_chunk(part, batch_examples, fingerprint)
            if saved is None:
                if model is None:
                    print(f"Loading {args.model}@{checkpoint_sha[:12]} onto {device}", flush=True)
                    model = AutoModel.from_pretrained(
                        args.model, revision=checkpoint_sha, trust_remote_code=False,
                    ).to(device).eval()
                    for p in model.parameters():
                        p.requires_grad_(False)
                    nparams = sum(p.numel() for p in model.parameters())
                t0 = time.perf_counter()
                x, y, ids = extract_chunk(batch_examples, data_root, model,
                                          args.batch_size, args.workers, device)
                timings[split] += time.perf_counter() - t0
                if ids != [e.utterance_id for e in batch_examples]:
                    raise RuntimeError("DataLoader order mismatch")
                atomically_save_npz(part, x, y, ids, fingerprint)
                print(f"{split}: saved {offset + len(batch_examples)}/{len(examples)} -> {part.name}", flush=True)
            else:
                print(f"{split}: reused {offset + len(batch_examples)}/{len(examples)}", flush=True)
            files.append(part)
        if not files:
            raise RuntimeError("No data")

    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    x_splits, y_splits = {}, {}
    for split, examples in (("train", train), ("dev", dev)):
        arrays, labs, gathered_ids = [], [], []
        for offset in range(0, len(examples), args.chunk_size):
            part = work / f"{split}_{offset:06d}_{len(examples[offset:offset + args.chunk_size]):04d}.npz"
            x, y, ids = read_chunk(part, examples[offset:offset + args.chunk_size], fingerprint)
            arrays.append(x)
            labs.append(y)
            gathered_ids.extend(ids)
        if gathered_ids != [e.utterance_id for e in examples]:
            raise ValueError(f"{split} cache order mismatch")
        x_splits[split] = np.concatenate(arrays)
        y_splits[split] = np.concatenate(labs)

    print("Training frozen-feature logistic regression probe...", flush=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=args.seed,
        class_weight=None))
    clf.fit(x_splits["train"], y_splits["train"])
    p_spoof = clf.predict_proba(x_splits["dev"])[:, list(clf[-1].classes_).index(0)]
    scores = dict(zip([e.utterance_id for e in dev], map(float, p_spoof)))
    metrics = evaluate_subgroups(dev, scores)
    if nparams is None:
        # Fully cached run: avoid loading the model merely to get a parameter count.
        nparams = None
    result = {
        "benchmark": PROTOCOL_VERSION, "checkpoint": args.model,
        "checkpoint_commit": checkpoint_sha, "cache_fingerprint": fingerprint,
        "pretraining_eligibility": "Requires explicit model-card provenance approval",
        "train_count": len(train), "dev_count": len(dev), "seed": args.seed,
        "train_manifest_sha256": manifest_sha(train), "dev_manifest_sha256": manifest_sha(dev),
        "encoder_parameters": nparams, "pooling": "mean_std", "precision": "fp32",
        "probe": "StandardScaler + LogisticRegression(C=1, class_weight=None)",
        "chunk_size": args.chunk_size, "batch_size": args.batch_size,
        "seconds": {"extract_train_this_run": round(timings["train"], 2),
                    "extract_dev_this_run": round(timings["dev"], 2),
                    "run_wall": round(time.perf_counter() - start, 2)},
        "metrics": metrics,
        "warning": "LOCAL DEV ONLY; not an official Codabench score. Probe quality does not guarantee fine-tuning rank.",
    }
    temp = final.with_suffix(".json.partial")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, final)
    print(json.dumps({"SAVED": str(final), "overall_macro_f1": metrics["all_dev"]["macro_f1"],
                      "online_macro_f1": metrics["view/online"]["macro_f1"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", required=True, choices=CANDIDATES)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output-dir", default="/kaggle/working/full_benchmark")
    ap.add_argument("--train-samples", type=int, default=0, help="0 means full train")
    ap.add_argument("--dev-samples", type=int, default=0, help="0 means full dev")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
