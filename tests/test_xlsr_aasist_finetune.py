"""Run coverage checks when PyTorch is available in development environment."""

import pytest

pytest.importorskip("torch")

from rtcsdd.training.xlsr_aasist_finetune import batches_for, epoch_permutation


def test_each_record_seen_once_per_epoch():
    first = epoch_permutation(73, 1234, 0)
    second = epoch_permutation(73, 1234, 1)
    assert len(first) == 73
    assert sorted(first) == list(range(73))
    assert first == epoch_permutation(73, 1234, 0)
    assert first != second


def test_resume_batch_offset():
    order = epoch_permutation(25, 1234, 0)
    batch = 8
    processed = 2
    previously_consumed = order[:processed * batch]
    remaining = order[processed * batch:]
    assert len(set(previously_consumed) | set(remaining)) == 25
    assert set(previously_consumed).isdisjoint(remaining)
    assert batches_for(25, 8) == 4
