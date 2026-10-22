"""Frozen-backbone linear-probe screening; train/dev only, NEVER progress/final.

Usage: python scripts/benchmark_pretrained_backbones.py --data-root /kaggle/input/<dataset> \
  --model facebook/wav2vec2-base --train-samples 3200 --dev-samples 1600
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtcsdd.data.dataset import RTCFakePackedDataset
from rtcsdd.data.protocol import read_protocol
from rtcsdd.eval.metrics import evaluate_subgroups

# Original, publicly released speech SSL checkpoints; no spoof-specific descendants.
CANDIDATES = (
    "facebook/wav2vec2-base",
    "facebook/hubert-base-ls960",
    "microsoft/wavlm-base-plus",
    "facebook/data2vec-audio-base",
    "facebook/wav2vec2-xls-r-300m",
    "microsoft/wavlm-large",
)


def stratified_subset(items, limit, seed):
    if limit == 0 or limit >= len(items):
        return list(items)
    groups = defaultdict(list)
    for ex in items:
        groups[(ex.view, ex.language, ex.label)].append(ex)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    sizes = {k: len(v) for k, v in groups.items()}
    # Largest-remainder allocation preserves original dataset proportions.
    quotas = {k: limit * n / len(items) for k, n in sizes.items()}
    counts = {k: min(sizes[k], int(quotas[k])) for k in groups}
    remaining = limit - sum(counts.values())
    ordering = sorted(groups, key=lambda k: (quotas[k] - int(quotas[k]), str(k)), reverse=True)
    while remaining:
        progressed = False
        for k in ordering:
            if counts[k] < sizes[k]:
                counts[k] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Unable to stratify")
    selected = {e.utterance_id for k, group in groups.items() for e in group[:counts[k]]}
    return [e for e in items if e.utterance_id in selected]


def encode(examples, data_root, model, batch_size, workers, device):
    import torch
    from torch.utils.data import DataLoader
    dataset = RTCFakePackedDataset(examples, data_root, rawboost={"enabled": False})
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    result, labels, ids = [], [], []
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for waves, y, batch_ids in loader:
            waves = waves.to(device, non_blocking=True)
            hidden = model(input_values=waves).last_hidden_state
            # Mean and std pooling; consistent across all backbones.
            features = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=1)
            result.append(features.float().cpu().numpy())
            labels.extend(y.numpy().tolist())
            ids.extend(batch_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(result), np.asarray(labels, dtype=np.int64), ids, time.perf_counter() - start


def run(args):
    import torch
    from transformers import AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if args.model not in CANDIDATES:
        raise ValueError(f"Unreviewed model checkpoint: {args.model}")
    if args.train_samples < 0 or args.dev_samples < 0 or args.batch_size < 1:
        raise ValueError("Sample counts must be >=0 and batch size >=1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = Path(args.data_root)
    train = stratified_subset(read_protocol(data_root / "train_label.txt", "train"), args.train_samples, args.seed)
    dev = stratified_subset(read_protocol(data_root / "dev_label.txt", "dev"), args.dev_samples, args.seed)
    if len({x.label for x in train}) != 2 or len({x.label for x in dev}) != 2:
        raise ValueError("Both classes needed for probe and evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"trust_remote_code": False}
    if args.revision:
        kwargs["revision"] = args.revision
    start = time.perf_counter()
    backbone = AutoModel.from_pretrained(args.model, **kwargs).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    nparams = sum(p.numel() for p in backbone.parameters())
    x_train, y_train, _, train_s = encode(train, data_root, backbone, args.batch_size, args.workers, device)
    x_dev, y_dev, dev_ids, dev_s = encode(dev, data_root, backbone, args.batch_size, args.workers, device)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Fixed, small supervised probe; this is NOT full encoder fine-tuning.
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                                solver="lbfgs", random_state=args.seed))
    probe.fit(x_train, y_train)
    # sklearn classes_[0] = 0 = spoof. Evaluate frozen 0.5 threshold.
    spoof_idx = list(probe[-1].classes_).index(0)
    p_spoof = probe.predict_proba(x_dev)[:, spoof_idx]
    scores = dict(zip(dev_ids, map(float, p_spoof)))
    metrics = evaluate_subgroups(dev, scores)
    from huggingface_hub import model_info
    info = model_info(args.model, revision=args.revision)
    output = {
        "benchmark": "frozen_mean_std_linear_probe_v1",
        "checkpoint": args.model,
        "checkpoint_commit": info.sha,
        "pretraining_provenance": "Human-reviewed official model card required; see documentation",
        "seed": args.seed,
        "train_count": len(train), "dev_count": len(dev),
        "train_manifest_sha256": hashlib.sha256("\n".join(e.utterance_id for e in train).encode()).hexdigest(),
        "dev_manifest_sha256": hashlib.sha256("\n".join(e.utterance_id for e in dev).encode()).hexdigest(),
        "encoder_parameters": nparams,
        "pooling": "mean_std", "probe": "StandardScaler + LogisticRegression(C=1, no class weighting)",
        "precision": "fp32", "batch_size": args.batch_size,
        "seconds": {"extract_train": round(train_s, 2), "extract_dev": round(dev_s, 2),
                    "total": round(time.perf_counter() - start, 2)},
        "metrics": metrics,
        "warning": "LOCAL DEV ONLY; no noisy eval and not official progress score",
    }
    out = Path(args.output) if args.output else Path("benchmark_results") / (args.model.replace("/", "__") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(out), "model": args.model,
                      "online_dev_macro_f1": metrics["view/online"]["macro_f1"],
                      "offline_dev_macro_f1": metrics["view/offline"]["macro_f1"],
                      "online_spoof_recall": metrics["view/online"]["spoof"]["recall"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True, choices=CANDIDATES)
    parser.add_argument("--revision", default=None, help="Optional immutable checkpoint commit")
    parser.add_argument("--train-samples", type=int, default=3200, help="0 = complete train")
    parser.add_argument("--dev-samples", type=int, default=1600, help="0 = complete dev")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
