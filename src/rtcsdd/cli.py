"""Command-line interface for dataset preflight, dev scoring and submissions.

There is deliberately no unverified `train` command in version 0.1.
Integration with the organizers' Fairseq/XLS-R baseline is Phase 1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from time import perf_counter
from pathlib import Path

from rtcsdd.data.packed import PackedAudioStore
from rtcsdd.data.protocol import read_protocol
from rtcsdd.eval.metrics import evaluate_subgroups
from rtcsdd.eval.submission import read_score_file, write_submission
from rtcsdd.experiments.registry import register_experiment

EXPECTED_SPLIT_COUNTS = {"train": 75_785, "dev": 15_204}
EXPECTED_VIEW_COUNTS = {
    ("train", "offline"): 38_660,
    ("train", "online"): 37_125,
    ("dev", "offline"): 7_739,
    ("dev", "online"): 7_465,
}


def preflight(args) -> None:
    root = Path(args.dataset_root)
    store = PackedAudioStore(root)
    audit = {"root": str(root), "splits": {}, "archives": {}, "warnings": []}
    try:
        for split in ("train", "dev"):
            examples = read_protocol(root / f"{split}_label.txt", split)
            count = len(examples)
            if args.require_audit_counts and count != EXPECTED_SPLIT_COUNTS[split]:
                raise ValueError(f"{split} count mismatch: {count} vs {EXPECTED_SPLIT_COUNTS[split]}")
            audit["splits"][split] = {
                "records": count,
                "labels": dict(Counter("spoof" if e.label == 0 else "bonafide" for e in examples)),
                "views": dict(Counter(e.view for e in examples)),
                "languages": dict(Counter(e.language for e in examples)),
            }
            for view in ("offline", "online"):
                subset = [e for e in examples if e.view == view]
                if not subset:
                    raise ValueError(f"No {view} entries in {split}")
                archive_count = store.member_count(split, view)
                if args.require_audit_counts and archive_count != EXPECTED_VIEW_COUNTS[split, view]:
                    raise ValueError(f"{split}/{view} archive count mismatch: {archive_count}")
                # Membership is checked via ZIP central directory; WAV bytes are not decoded.
                missing = [e.utterance_id for e in subset if not store.has_member(e)]
                if missing:
                    raise ValueError(f"Missing {len(missing)} ZIP members, first={missing[:3]}")
                if archive_count != len(subset):
                    audit["warnings"].append(
                        f"{split}/{view}: archive has {archive_count} WAVs vs {len(subset)} protocol IDs"
                    )
                audit["archives"][f"{split}_{view}"] = {
                    "path": str(store.archive_path(subset[0])),
                    "wavs": archive_count,
                    "protocol_members_found": len(subset),
                }
        print(json.dumps(audit, indent=2))
    finally:
        store.close()


def smoke_audio(args) -> None:
    root = Path(args.dataset_root)
    store = PackedAudioStore(root)
    audit = []
    try:
        for split in ("train", "dev"):
            records = read_protocol(root / f"{split}_label.txt", split)
            for view in ("offline", "online"):
                for lang in ("en", "zh"):
                    for label in (0, 1):
                        sample = next(
                            (e for e in records if (e.view, e.language, e.label) == (view, lang, label)),
                            None,
                        )
                        if sample is None:
                            continue
                        samples, sr = store.read(sample)
                        audit.append({
                            "split": split, "view": view, "lang": lang,
                            "label": label, "id": sample.utterance_id,
                            "duration_s": round(len(samples) / sr, 3),
                        })
        print(json.dumps({"smoke_read_count": len(audit), "results": audit}, indent=2))
    finally:
        store.close()


def benchmark_data(args) -> None:
    """Time model-free packed WAV loading on an allowed training/dev split."""
    import random

    from rtcsdd.data.dataset import RTCFakePackedDataset
    from torch.utils.data import DataLoader, Subset

    root = Path(args.dataset_root)
    records = read_protocol(root / f"{args.split}_label.txt", args.split)
    if args.samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Expected samples>0, batch-size>0 and workers>=0")
    sample_count = min(args.samples, len(records))
    random_ids = random.Random(args.seed).sample(range(len(records)), sample_count)
    dataset = RTCFakePackedDataset(records, str(root))
    loader = DataLoader(
        Subset(dataset, random_ids), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=False,
    )
    started = perf_counter()
    seen = 0
    for waveforms, labels, ids in loader:
        if waveforms.shape[1] != 64_600 or not len(ids) == len(labels):
            raise ValueError("Invalid data batch during benchmark")
        seen += len(ids)
    seconds = perf_counter() - started
    print(json.dumps({
        "split": args.split, "samples": seen, "workers": args.workers,
        "batch_size": args.batch_size, "seconds": round(seconds, 3),
        "samples_per_second": round(seen / max(seconds, 1e-9), 2),
        "note": "Model-free audio throughput, not GPU training throughput",
    }, indent=2))


def score_dev(args) -> None:
    records = read_protocol(args.protocol, "dev")
    scores = read_score_file(args.scores)
    results = evaluate_subgroups(records, scores)
    rendered = json.dumps(results, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def submit(args) -> None:
    examples = read_protocol(args.protocol, args.split)
    scores = read_score_file(args.scores)
    write_submission(examples, scores, args.out)
    print(f"Validated and wrote {len(examples)} scores to {args.out}")


def register(args) -> None:
    directory = register_experiment(args.config, args.runs_root)
    print(f"Experiment manifest created: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rtcsdd", description="RTC-SDD research utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight", help="Validate train/dev protocol-to-pack membership")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--require-audit-counts", action="store_true")
    p.set_defaults(func=preflight)

    p = sub.add_parser("smoke-audio", help="Decode representative train/dev WAVs from .pack")
    p.add_argument("--dataset-root", required=True)
    p.set_defaults(func=smoke_audio)

    p = sub.add_parser("benchmark-data", help="Benchmark randomized train/dev pack reads")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=benchmark_data)

    p = sub.add_parser("score-dev", help="Strict macro-F1 + subgroup reporting on labeled dev")
    p.add_argument("--protocol", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out")
    p.set_defaults(func=score_dev)

    p = sub.add_parser("submission", help="Validate/order progress/eval scores; no model inference")
    p.add_argument("--split", choices=["progress", "eval"], required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=submit)

    p = sub.add_parser("register", help="Create immutable experiment config/environment manifest")
    p.add_argument("--config", required=True)
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=register)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
