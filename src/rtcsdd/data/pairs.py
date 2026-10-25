"""Validated RTCFake offline-online pairing for PCL experiments.

This module only resolves protocol references. Audio loading,
augmentation, batching, and phoneme alignment remain separate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .protocol import Example, validate_utterance_id


@dataclass(frozen=True)
class PairGroup:
    """One offline recording and its optional online counterpart."""

    offline: Example
    online: Example | None

    @property
    def is_paired(self) -> bool:
        return self.online is not None

    @property
    def members(self) -> tuple[Example, ...]:
        if self.online is None:
            return (self.offline,)
        return (self.offline, self.online)


def _normalize_csv_id(
    raw_value: str,
    *,
    split: str,
    expected_view: str,
    row_number: int,
) -> str:
    """Convert split/view/lang/file.wav to view/lang/file.wav."""

    value = str(raw_value).strip()
    prefix = f"{split}/"

    if not value.startswith(prefix):
        raise ValueError(
            f"Row {row_number}: invalid split prefix"
        )

    normalized = value[len(prefix):]

    try:
        view, _ = validate_utterance_id(
            normalized,
            split=split,
        )
    except ValueError:
        raise ValueError(
            f"Row {row_number}: invalid recording ID"
        ) from None

    if view != expected_view:
        raise ValueError(
            f"Row {row_number}: expected {expected_view} view"
        )

    return normalized


def load_pair_groups(
    csv_path: str | Path,
    protocol_examples: Sequence[Example],
    *,
    split: str,
) -> list[PairGroup]:
    """Resolve and validate every protocol sample exactly once.

    Complete pairs contain two recordings.
    Incomplete pairs contain one offline recording.

    The returned groups preserve CSV row order.
    """

    if split not in {"train", "dev"}:
        raise ValueError(
            "Pair groups are supported only for Train/Dev"
        )

    if not protocol_examples:
        raise ValueError("Protocol examples cannot be empty")

    if any(item.split != split for item in protocol_examples):
        raise ValueError("Protocol split mismatch")

    protocol_index = {
        item.utterance_id: item
        for item in protocol_examples
    }

    if len(protocol_index) != len(protocol_examples):
        raise ValueError("Duplicate protocol IDs")

    groups: list[PairGroup] = []
    seen_ids: set[str] = set()

    with Path(csv_path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("Pair CSV has no header")

        required = {"offline_id", "online_id"}

        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Pair CSV requires offline_id and online_id"
            )

        for row_number, row in enumerate(reader, start=2):

            if None in row:
                raise ValueError(
                    f"Row {row_number}: unexpected extra columns"
                )

            raw_offline = row.get("offline_id")
            raw_online = row.get("online_id")

            if not raw_offline or not str(raw_offline).strip():
                raise ValueError(
                    f"Row {row_number}: missing offline ID"
                )

            offline_id = _normalize_csv_id(
                raw_offline,
                split=split,
                expected_view="offline",
                row_number=row_number,
            )

            if offline_id in seen_ids:
                raise ValueError(
                    f"Row {row_number}: duplicate offline reference"
                )

            offline = protocol_index.get(offline_id)

            if offline is None:
                raise ValueError(
                    f"Row {row_number}: offline ID absent from protocol"
                )

            seen_ids.add(offline_id)
            online = None

            if raw_online and str(raw_online).strip():

                online_id = _normalize_csv_id(
                    raw_online,
                    split=split,
                    expected_view="online",
                    row_number=row_number,
                )

                if online_id in seen_ids:
                    raise ValueError(
                        f"Row {row_number}: duplicate online reference"
                    )

                online = protocol_index.get(online_id)

                if online is None:
                    raise ValueError(
                        f"Row {row_number}: online ID absent from protocol"
                    )

                if offline.label != online.label:
                    raise ValueError(
                        f"Row {row_number}: paired label mismatch"
                    )

                if offline.language != online.language:
                    raise ValueError(
                        f"Row {row_number}: paired language mismatch"
                    )

                seen_ids.add(online_id)

            groups.append(
                PairGroup(
                    offline=offline,
                    online=online,
                )
            )

    missing = set(protocol_index) - seen_ids

    if missing:
        raise ValueError(
            f"Protocol coverage incomplete: {len(missing)} "
            "recordings were not represented"
        )

    if len(seen_ids) != len(protocol_index):
        raise ValueError("Unexpected protocol coverage mismatch")

    return groups
