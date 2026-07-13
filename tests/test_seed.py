from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from gdp.seed import select_device, set_seed


def test_set_seed_makes_runs_reproducible():
    def draw():
        set_seed(1234)
        return (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert draw() == draw()


def test_different_seeds_differ():
    set_seed(1)
    a = float(torch.rand(1))
    set_seed(2)
    b = float(torch.rand(1))
    assert a != b


def test_select_device_auto_returns_available_device():
    device = select_device("auto")
    assert device.type in ("cuda", "mps", "cpu")


def test_select_device_cpu_always_works():
    assert select_device("cpu").type == "cpu"


def test_unavailable_device_raises_rather_than_falling_back():
    """Silently degrading cuda→cpu would hide a broken cluster job for hours."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            select_device("cuda")


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="unknown device"):
        select_device("tpu")
