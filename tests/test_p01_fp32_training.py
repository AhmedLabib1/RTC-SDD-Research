"""Tests for independent FP32 P01 experiment and optimizer safety."""
import math
from pathlib import Path

import pytest
import torch

from rtcsdd.training.config import read_config, fingerprint, validate
from rtcsdd.training.trainer import apply_optimizer_update, optimizer_steps_per_epoch


P01 = Path(__file__).resolve().parents[1] / 'src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json'


def test_p01_fp32_config_preserves_official_like_settings():
    p00 = read_config()
    p01 = read_config(P01)
    assert p01['experiment'] == 'p01_xlsr_aasist_official_like_fp32'
    assert p00['training']['precision'] == 'amp_fp16'
    assert p01['training']['precision'] == 'fp32'
    assert p01['training']['batch_size'] == 2
    assert p01['training']['gradient_accumulation'] == 8
    assert optimizer_steps_per_epoch(75785, 2, 8) == 4737
    assert p01['training']['epochs'] == 3
    assert p01['training']['class_weights'] == [0.1, 0.9]
    assert p01['training']['learning_rate'] == 1e-6
    assert p01['training']['weight_decay'] == 1e-4
    assert p01['training']['lr_scheduler'] == 'none'
    assert p01['data'] == p00['data']
    assert p01['model'] == p00['model']
    assert fingerprint(p01) != fingerprint(p00)
    validate(p01)


def test_fp32_update_applies_valid_weight_gradient():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.tensor([[1.0, 2.0]])
    model(x).sum().backward()
    before = model.weight.detach().clone()
    skipped, scale_before, scale_after = apply_optimizer_update(
        optimizer=optim, scaler=None, amp=False, grad_norm=1.0
    )
    assert not skipped
    assert scale_before == scale_after == 1.0
    assert not torch.equal(before, model.weight.detach())
    assert len(optim.state) == 2


@pytest.mark.parametrize('bad_grad', [float('nan'), float('inf'), -float('inf')])
def test_fp32_nonfinite_grad_never_corrupts_adam(bad_grad):
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.tensor([[1.0, 2.0]])
    model(x).sum().backward()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(FloatingPointError, match='NOT executed'):
        apply_optimizer_update(optimizer=optim, scaler=None, amp=False, grad_norm=bad_grad)
    assert len(optim.state) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
