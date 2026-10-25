"""Deterministic pair-preserving micro-batches for RTCFake PCL."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from .pairs import PairGroup
from .protocol import Example


class PairPreservingBatchSampler:
    """Produce index batches without splitting offline-online pairs.

    A complete pair occupies one batch of two recordings.
    Unpaired offline recordings are batched together, two at a time.

    Batch order is deterministic for a given seed and epoch.
    No samples are dropped.
    """

    def __init__(
        self,
        records: Sequence[Example],
        groups: Sequence[PairGroup],
        *,
        seed: int = 1234,
        epoch: int = 0,
    ):

        if not records or not groups:
            raise ValueError("Records and groups cannot be empty")

        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")

        index = {
            record.utterance_id: position
            for position, record in enumerate(records)
        }

        if len(index) != len(records):
            raise ValueError("Duplicate protocol records")

        pair_batches: list[tuple[int, int]] = []
        single_indices: list[int] = []

        pair_ids: set[tuple[str, str]] = set()

        for group in groups:

            offline_idx = index.get(
                group.offline.utterance_id
            )

            if offline_idx is None:
                raise ValueError(
                    "Pair references an unknown offline recording"
                )

            if group.online is None:
                single_indices.append(offline_idx)
                continue

            online_idx = index.get(
                group.online.utterance_id
            )

            if online_idx is None:
                raise ValueError(
                    "Pair references an unknown online recording"
                )

            pair_batches.append(
                (offline_idx, online_idx)
            )

            pair_ids.add(
                (
                    group.offline.utterance_id,
                    group.online.utterance_id,
                )
            )

        # Deterministic RNG independent of global random state.
        rng = random.Random(seed + epoch)

        # Shuffle unpaired recordings before grouping them.
        rng.shuffle(single_indices)

        single_batches = [
            tuple(single_indices[position:position + 2])
            for position in range(0, len(single_indices), 2)
        ]

        batches = pair_batches + single_batches

        # Mix paired batches and CE-only batches.
        rng.shuffle(batches)

        # Every protocol recording must appear exactly once.
        flattened = [
            index_value
            for batch in batches
            for index_value in batch
        ]

        if (
            len(flattened) != len(records)
            or len(set(flattened)) != len(records)
            or set(flattened) != set(range(len(records)))
        ):
            raise ValueError(
                "Batch construction failed exact sample coverage"
            )

        self._batches = tuple(batches)
        self._pair_ids = frozenset(pair_ids)

        self.seed = seed
        self.epoch = epoch
        self.paired_batch_count = len(pair_batches)
        self.ce_only_batch_count = len(single_batches)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._batches:
            yield list(batch)

    def is_pair_batch(self, utterance_ids: Sequence[str]) -> bool:
        """Check whether IDs form an original ordered pair."""

        return (
            len(utterance_ids) == 2
            and tuple(utterance_ids) in self._pair_ids
        )
