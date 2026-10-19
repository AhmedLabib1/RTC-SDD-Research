import io
import json
import tarfile
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rtcsdd.models.xlsr_aasist import load_xlsr_aasist
from rtcsdd.data.audio import crop_or_pad
from rtcsdd.eval.submission import read_score_file, write_submission
from rtcsdd.data.protocol import Example

ROOT = Path("/kaggle/working/RTC-SDD-Research")
DATA = Path("/kaggle/input/datasets/a7medlabib/rtcfake-packed-private")
OUT = ROOT / "outputs/experiments/p01_xlsr_aasist_official_like_fp32"

CHECKPOINT = OUT / "best_online_f1.pt"
PRETRAINED = Path("/kaggle/working/checkpoints/pretrained/xlsr2_300m.pt")
OFFICIAL = ROOT / "third_party/official_rtc_sdd"
CONFIG = ROOT / "src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json"

PACK = DATA / "progress.pack"
PROTOCOL = DATA / "progress.txt"

PARTIAL = OUT / "progress_scores_partial.txt"
SUBMISSION = OUT / "submission_progress.txt"

BATCH_SIZE = 2
DEVICE = "cuda:0"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA unavailable"

    for path in [CHECKPOINT, PRETRAINED, CONFIG, PACK, PROTOCOL]:
        assert path.is_file(), f"Missing: {path}"

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    dcfg = cfg["data"]

    ids = [
        line.strip()
        for line in PROTOCOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(ids) == 30279
    assert len(ids) == len(set(ids))

    expected = set(ids)

    scores = read_score_file(PARTIAL) if PARTIAL.exists() else {}
    assert set(scores).issubset(expected), "Unknown saved IDs"

    print("=" * 65, flush=True)
    print("P01 FULL PROGRESS INFERENCE", flush=True)
    print(f"Total recordings: {len(ids)}", flush=True)
    print(f"Already processed: {len(scores)}", flush=True)
    print("HF Backup: DISABLED", flush=True)
    print("=" * 65, flush=True)

    if len(scores) < len(ids):
        print("\nLoading trained model...", flush=True)

        model = load_xlsr_aasist(
            OFFICIAL, PRETRAINED, DEVICE
        )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        del checkpoint
        model.eval()

        print("MODEL READY!", flush=True)

        pending = []
        seen = set()

        def predict_batch(output_file):
            nonlocal pending

            if not pending:
                return

            batch = torch.stack([
                audio for _, audio in pending
            ]).to(DEVICE)

            with torch.inference_mode():
                logits = model(batch)
                batch_scores = (
                    torch.softmax(logits.float(), dim=-1)[:, 0]
                    .cpu().tolist()
                )

            for (uid, _), score in zip(pending, batch_scores):
                assert math.isfinite(score) and 0 <= score <= 1

                scores[uid] = float(score)
                output_file.write(f"{uid} {score:.10f}\n")

            # Save locally after every batch
            output_file.flush()
            pending.clear()

            count = len(scores)

            if count % 500 < BATCH_SIZE or count == len(ids):
                print(
                    f"[PROGRESS] {count}/{len(ids)} "
                    f"({count / len(ids) * 100:.1f}%)",
                    flush=True,
                )

        print("\nStarting inference...", flush=True)

        with PARTIAL.open("a", encoding="utf-8") as output_file:
            with tarfile.open(PACK, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue

                    assert member.name.startswith("progress/")
                    uid = member.name.removeprefix("progress/")

                    if uid not in expected:
                        raise ValueError(f"Unknown audio: {uid}")

                    if uid in seen:
                        raise ValueError(f"Duplicate audio: {uid}")

                    seen.add(uid)

                    if uid in scores:
                        continue

                    handle = archive.extractfile(member)
                    assert handle is not None

                    audio, sr = sf.read(
                        io.BytesIO(handle.read()),
                        dtype="float32",
                        always_2d=False,
                    )

                    assert sr == 16000
                    assert audio.ndim == 1
                    assert audio.size > 0
                    assert np.isfinite(audio).all()

                    fixed = crop_or_pad(
                        audio,
                        length=dcfg["audio_length"],
                        crop=dcfg["crop"],
                        pad=dcfg["pad"],
                    )

                    pending.append(
                        (uid, torch.from_numpy(fixed))
                    )

                    if len(pending) == BATCH_SIZE:
                        predict_batch(output_file)

                predict_batch(output_file)

        assert len(scores) == len(ids), (
            f"Incomplete: {len(scores)}/{len(ids)}"
        )

    # Generate the final submission in official protocol order
    scores = read_score_file(PARTIAL)

    assert set(scores) == expected

    examples = [
        Example(
            utterance_id=uid,
            split="progress",
            view=uid.split("/")[1],
            language=uid.split("/")[2],
            label=None,
        )
        for uid in ids
    ]

    write_submission(examples, scores, SUBMISSION)

    validated = read_score_file(SUBMISSION)
    assert list(validated) == ids

    print("\n" + "=" * 65, flush=True)
    print("SUBMISSION READY!", flush=True)
    print("File:", SUBMISSION, flush=True)
    print("Recordings:", len(validated), flush=True)
    print("Min score:", min(validated.values()), flush=True)
    print("Max score:", max(validated.values()), flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
