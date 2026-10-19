"""CLI entrypoint for P00 XLS-R 300M + AASIST official-like fine-tuning.

  /kaggle/working/rtcsdd-baseline-py310/bin/python \
    -m rtcsdd.training.xlsr_aasist_finetune \
    --root /kaggle/working/RTC-SDD-Research \
    --dataset-root /kaggle/input/datasets/a7medlabib/rtcfake-packed-private \
    --checkpoint /kaggle/working/checkpoints/pretrained/xlsr2_300m.pt \
    --out /kaggle/working/RTC-SDD-Research/outputs/experiments/p00_xlsr_aasist_official_like

Run the same command with --resume to continue from last.pt.
This single-GPU run does not use T4 x2 distributed data parallel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rtcsdd.training.config import read_config, with_overrides
from rtcsdd.training.trainer import train, epoch_permutation  # legacy test helper


def batches_for(total: int, batch_size: int) -> int:
    """Compatibility for v0.4 schedule tests; training uses optimizer_steps_per_epoch."""
    import math
    return math.ceil(total / batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Project root containing third_party/official_rtc_sdd")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Private packed RTCFake dataset root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original pretrained XLS-R Fairseq checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Experiment output directory")
    parser.add_argument("--config", type=Path, help="Optional configuration JSON; bundled P00 used by default")
    parser.add_argument("--resume", action="store_true", help="Resume an existing v0.5 last.pt checkpoint")
    parser.add_argument("--epochs", type=int, help="Override total number of epochs")
    parser.add_argument("--batch-size", type=int, help="Micro-batch size; default=8, may OOM on T4")
    parser.add_argument("--accumulate", type=int, help="Gradient accumulation; default=2")
    parser.add_argument("--workers", type=int, help="DataLoader workers")
    parser.add_argument("--max-minutes", type=float, help="Graceful pause window, 0=no time cap")
    args = parser.parse_args()
    config = with_overrides(
        read_config(args.config),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.accumulate,
        workers=args.workers,
        max_minutes=args.max_minutes,
    )
    train(
        config,
        project_root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pretrained_checkpoint=args.checkpoint.resolve(),
        output=args.out.resolve(),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
