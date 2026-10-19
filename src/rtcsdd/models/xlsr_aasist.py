"""Load the pinned upstream XLS-R+AASIST model with a configurable SSL checkpoint.

Uses the exact upstream model class. The third_party checkout remains unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch


def load_xlsr_aasist(upstream_root: str | Path, ssl_checkpoint: str | Path, device: str):
    import numpy as np

    # Pinned legacy Fairseq expects removed NumPy aliases.
    for name, value in {
        "float": float, "int": int, "complex": complex,
        "bool": bool, "object": object, "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    import torch
    from fairseq import checkpoint_utils

    upstream_root = Path(upstream_root)
    checkpoint = Path(ssl_checkpoint)
    model_file = upstream_root / "xlsr_aasist/model/model.py"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Confirm we are hooking the exact fixed-path call from the pinned upstream.
    placeholder = "/path/to/pretrained/xlsr2_300m.pt"
    if model_file.read_text(encoding="utf-8").count(placeholder) != 1:
        raise RuntimeError("Official checkpoint-loading interface changed")

    spec = importlib.util.spec_from_file_location("rtcsdd_official_aasist", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model: {model_file}")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    actual_loader = checkpoint_utils.load_model_ensemble_and_task

    def redirect_checkpoint(paths, *args, **kwargs):
        if list(paths) != [placeholder]:
            return actual_loader(paths, *args, **kwargs)
        return actual_loader([str(checkpoint)], *args, **kwargs)

    # Patch only during construction, not globally during training.
    with patch.object(checkpoint_utils, "load_model_ensemble_and_task", redirect_checkpoint):
        model = upstream.Model(argparse.Namespace(), device)

    return model.to(torch.device(device))
