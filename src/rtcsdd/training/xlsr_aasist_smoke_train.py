"""First REAL packed-audio fine-tuning run (short-run baseline, RawBoost off).

Not yet a reproduction of the official long training schedule. No held-out audio used.
Run with the pinned Python 3.10 environment and --steps 10 before increasing steps.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from rtcsdd.data.dataset import RTCFakePackedDataset
from rtcsdd.data.protocol import read_protocol
from rtcsdd.eval.metrics import evaluate_subgroups
from rtcsdd.models.xlsr_aasist import load_xlsr_aasist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(total: int, count: int, seed: int) -> list[int]:
    if count == 0 or count >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), count))


def save_checkpoint(path: Path, model, step: int, options: dict) -> None:
    # Keep one rolling checkpoint. CPU serialization minimizes GPU allocations.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "options": options,
        "optimizer_state_saved": False,  # Short-run checkpoint; no exact optimizer resume.
    }, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--dev-samples", type=int, default=128, help="0=full dev")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.accumulate < 1 or args.save_every < 1:
        parser.error("steps, batch-size, accumulate, save-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; enable GPU T4 in Kaggle")

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    train_examples = read_protocol(args.dataset_root / "train_label.txt", "train")
    dev_examples = read_protocol(args.dataset_root / "dev_label.txt", "dev")
    selected = subset_indices(len(dev_examples), args.dev_samples, args.seed)
    dev_subset_examples = [dev_examples[i] for i in selected]

    train_dataset = RTCFakePackedDataset(train_examples, str(args.dataset_root))
    dev_dataset = RTCFakePackedDataset(dev_examples, str(args.dataset_root))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator,
                              pin_memory=True, drop_last=True)
    dev_loader = DataLoader(Subset(dev_dataset, selected), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    print("Loading pinned upstream XLS-R + AASIST...", flush=True)
    model = load_xlsr_aasist(args.root / "third_party/official_rtc_sdd", args.checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    weight = torch.tensor([0.1, 0.9], dtype=torch.float32, device=device)
    options = vars(args).copy()
    options = {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}
    (args.out / "config.json").write_text(json.dumps(options, indent=2) + "\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    t0 = time.perf_counter()
    data_iter = iter(train_loader)
    while step < args.steps:
        try:
            audio, labels, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            audio, labels, _ = next(data_iter)
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(audio)
            loss = F.cross_entropy(logits.float(), labels, weight=weight)
        scaler.scale(loss / args.accumulate).backward()
        micro += 1
        if micro % args.accumulate:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % 2 == 0:
            print(f"step={step}/{args.steps} loss={loss.item():.4f} elapsed_s={time.perf_counter()-t0:.1f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.out / "last.pt", model, step, options)
            print(f"[SAVED] step {step}", flush=True)

    print("Evaluating selected dev subset...", flush=True)
    model.eval()
    scores = {}
    with torch.inference_mode():
        for audio, _, ids in dev_loader:
            audio = audio.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(audio)
            prob_spoof = logits.float().softmax(dim=-1)[:, 0].cpu().tolist()
            scores.update(zip(ids, prob_spoof))
    metrics = evaluate_subgroups(dev_subset_examples, scores)
    (args.out / "dev_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"dev_subset_macro_f1={metrics['all_dev']['macro_f1']:.4f}", flush=True)
    print("Done. Checkpoint:", args.out / "last.pt", flush=True)
    print("Not official full-dev or challenge weighted macro-F1.", flush=True)


if __name__ == "__main__":
    main()
