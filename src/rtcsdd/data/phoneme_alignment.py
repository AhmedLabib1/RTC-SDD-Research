"""CTC phoneme segmentation and monotonic pair alignment.

Pure-Python implementation. No torch dependency or model loading.

CTC frame IDs are converted to contiguous non-blank segments.
Offline and online segments are matched using an ordered
Longest Common Subsequence (LCS) alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Collection, Sequence


@dataclass(frozen=True)
class CTCSegment:
    """A predicted token spanning [start_frame, end_frame)."""

    token_id: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class MatchedPhoneme:
    """Corresponding offline-online phoneme segments."""

    token_id: int
    offline: CTCSegment
    online: CTCSegment


def ctc_token_segments(
    frame_token_ids: Sequence[int],
    *,
    blank_id: int,
    excluded_ids: Collection[int] = (),
) -> list[CTCSegment]:
    """Extract CTC segments, retaining their frame boundaries.

    Repeated tokens separated by a blank remain separate segments.
    """

    if not frame_token_ids:
        return []

    excluded = set(excluded_ids)
    tokens = [int(value) for value in frame_token_ids]

    if any(value < 0 for value in tokens):
        raise ValueError("CTC token IDs must be nonnegative")

    segments: list[CTCSegment] = []

    start = 0
    previous = tokens[0]

    for frame in range(1, len(tokens) + 1):

        current = (
            tokens[frame]
            if frame < len(tokens)
            else None
        )

        if current == previous:
            continue

        if (
            previous is not None
            and previous != blank_id
            and previous not in excluded
        ):
            segments.append(
                CTCSegment(
                    token_id=previous,
                    start_frame=start,
                    end_frame=frame,
                )
            )

        previous = current
        start = frame

    return segments


def align_phoneme_segments(
    offline: Sequence[CTCSegment],
    online: Sequence[CTCSegment],
) -> tuple[MatchedPhoneme, ...]:
    """Match identical tokens while preserving temporal order.

    The dynamic program computes an LCS over phoneme token IDs.
    Unmatched insertions and deletions are skipped.
    """

    n = len(offline)
    m = len(online)

    if n == 0 or m == 0:
        return ()

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n - 1, -1, -1):

        for j in range(m - 1, -1, -1):

            if offline[i].token_id == online[j].token_id:

                dp[i][j] = 1 + dp[i + 1][j + 1]

            else:

                dp[i][j] = max(
                    dp[i + 1][j],
                    dp[i][j + 1],
                )

    matches: list[MatchedPhoneme] = []

    i = 0
    j = 0

    while i < n and j < m:

        if offline[i].token_id == online[j].token_id:

            matches.append(
                MatchedPhoneme(
                    token_id=offline[i].token_id,
                    offline=offline[i],
                    online=online[j],
                )
            )

            i += 1
            j += 1

        elif dp[i + 1][j] >= dp[i][j + 1]:

            i += 1

        else:

            j += 1

    return tuple(matches)


def alignment_similarity(
    matched_count: int,
    offline_count: int,
    online_count: int,
) -> float:
    """Symmetric token-sequence matching score."""

    denominator = offline_count + online_count

    if denominator == 0:
        return 0.0

    if matched_count < 0 or 2 * matched_count > denominator:
        raise ValueError("Invalid alignment statistics")

    return 2.0 * matched_count / denominator
