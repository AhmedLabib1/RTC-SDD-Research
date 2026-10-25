
"""E02 PCL alignment shard cache — schema v1."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

SCHEMA_VERSION = 1

REQUIRED_META = {
    "schema_version",
    "split",
    "start",
    "stop",
    "manifest_sha256",
    "model_sha256",
    "sample_rate",
    "audio_length",
    "crop",
    "pad",
    "excluded_ids",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pair_key_sha256(
    offline_id: str,
    online_id: str,
) -> str:
    """Stable digest without storing recording IDs."""

    raw = (
        offline_id.encode("utf-8")
        + b"\x00"
        + online_id.encode("utf-8")
    )

    return _sha256_bytes(raw)


def shard_path(
    root: str | Path,
    split: str,
    start: int,
    stop: int,
) -> Path:

    if split not in {"train", "dev"}:
        raise ValueError("Invalid split")

    if not 0 <= start < stop:
        raise ValueError("Invalid shard bounds")

    return (
        Path(root)
        / split
        / f"{split}_{start:06d}_{stop:06d}.json.gz"
    )


def validate_payload(
    metadata: dict,
    records: list[dict],
) -> None:

    if not REQUIRED_META.issubset(metadata):
        raise ValueError("Missing metadata fields")

    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported schema version")

    if metadata["split"] not in {"train", "dev"}:
        raise ValueError("Unsupported split")

    start = int(metadata["start"])
    stop = int(metadata["stop"])

    if start < 0 or stop <= start:
        raise ValueError("Invalid coverage bounds")

    if len(records) != stop - start:
        raise ValueError("Shard coverage mismatch")

    if (
        int(metadata["sample_rate"]) != 16000
        or int(metadata["audio_length"]) != 64600
    ):
        raise ValueError("Unexpected audio configuration")

    if (
        metadata["crop"] != "left"
        or metadata["pad"] != "repeat"
    ):
        raise ValueError("Unexpected crop/pad settings")

    for name in ("manifest_sha256", "model_sha256"):
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            metadata[name],
        ):
            raise ValueError(f"Invalid {name}")

    if not isinstance(metadata["excluded_ids"], list):
        raise ValueError("Invalid excluded_ids")

    for index, record in enumerate(records, start):

        expected_fields = {
            "pair_index",
            "pair_hash",
            "offline_frames",
            "online_frames",
            "matches",
        }

        if set(record) != expected_fields:
            raise ValueError("Unexpected record fields")

        if record["pair_index"] != index:
            raise ValueError("Non-contiguous pair indices")

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            record["pair_hash"],
        ):
            raise ValueError("Invalid pair hash")

        off_n = record["offline_frames"]
        on_n = record["online_frames"]

        if (
            not isinstance(off_n, int)
            or not isinstance(on_n, int)
            or min(off_n, on_n) <= 0
        ):
            raise ValueError("Invalid frame counts")

        previous_off = -1
        previous_on = -1

        for match in record["matches"]:

            if (
                not isinstance(match, list)
                or len(match) != 5
                or not all(
                    isinstance(value, int)
                    for value in match
                )
            ):
                raise ValueError("Malformed match")

            token, os_, oe, ns, ne = match

            if not (
                0 <= token < 392
                and 0 <= os_ < oe <= off_n
                and 0 <= ns < ne <= on_n
            ):
                raise ValueError(
                    "Invalid token or phoneme frame span"
                )

            if (
                os_ <= previous_off
                or ns <= previous_on
            ):
                raise ValueError(
                    "Non-monotonic matching"
                )

            previous_off = os_
            previous_on = ns


def write_shard(
    root: str | Path,
    metadata: dict,
    records: list[dict],
) -> tuple[Path, str]:
    """
    Atomically write one complete shard.

    Existing outputs are never overwritten.
    """

    validate_payload(metadata, records)

    path = shard_path(
        root,
        metadata["split"],
        metadata["start"],
        metadata["stop"],
    )

    checksum_path = Path(
        str(path) + ".sha256"
    )

    if path.exists() or checksum_path.exists():
        raise FileExistsError(
            "Shard already exists; inspect before resuming"
        )

    payload = {
        "metadata": metadata,
        "records": records,
    }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    compressed = gzip.compress(
        raw,
        compresslevel=6,
        mtime=0,
    )

    digest = _sha256_bytes(compressed)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + f".{uuid.uuid4().hex}.partial"
    )

    temporary_checksum = checksum_path.with_name(
        checksum_path.name
        + f".{uuid.uuid4().hex}.partial"
    )

    try:
        with temporary_path.open("xb") as f:
            f.write(compressed)
            f.flush()
            os.fsync(f.fileno())

        with temporary_checksum.open(
            "x",
            encoding="ascii",
        ) as f:
            f.write(digest + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_path, path)
        os.replace(temporary_checksum, checksum_path)

    finally:
        # Only remove temporary files created by this call.
        temporary_path.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)

    return path, digest


def read_shard(
    path: str | Path,
    *,
    expected_metadata: dict | None = None,
) -> dict:

    path = Path(path)

    checksum_path = Path(
        str(path) + ".sha256"
    )

    if (
        not path.is_file()
        or not checksum_path.is_file()
    ):
        raise FileNotFoundError(
            "Incomplete shard or missing checksum"
        )

    expected_sha = checksum_path.read_text(
        encoding="ascii"
    ).strip()

    compressed = path.read_bytes()

    actual_sha = _sha256_bytes(compressed)

    if (
        not re.fullmatch(
            r"[0-9a-f]{64}",
            expected_sha,
        )
        or actual_sha != expected_sha
    ):
        raise ValueError(
            "Shard SHA256 verification failed"
        )

    payload = json.loads(
        gzip.decompress(compressed).decode("utf-8")
    )

    validate_payload(
        payload["metadata"],
        payload["records"],
    )

    metadata = payload["metadata"]

    expected_name = shard_path(
        ".",
        metadata["split"],
        metadata["start"],
        metadata["stop"],
    ).name

    if path.name != expected_name:
        raise ValueError(
            "Shard filename and metadata disagree"
        )

    if (
        expected_metadata is not None
        and metadata != expected_metadata
    ):
        raise ValueError(
            "Cache metadata mismatch"
        )

    return payload
