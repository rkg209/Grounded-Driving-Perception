"""Determinism and device selection.

Reproducibility is a stated goal of this project: a metric that cannot be reproduced cannot be
defended (CLAUDE.md, H8). Every entry point calls set_seed() before touching a model.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (all devices). Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    'auto' prefers cuda (cluster) → mps (M4 laptop) → cpu. An explicit request for an
    unavailable device raises rather than silently falling back: silently training on CPU for
    ten hours because CUDA was missing is a failure mode worth being loud about.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    if preference not in ("cpu", "mps", "cuda"):
        raise ValueError(f"unknown device: {preference!r}")

    return torch.device(preference)
