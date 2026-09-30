"""`resume` must hand `torch.set_rng_state` a CPU tensor, whatever device the checkpoint loaded to.

Both trainers `torch.load(..., map_location=self.device)`, which on CUDA puts the saved RNG state on
the GPU; `torch.set_rng_state` then raises "RNG state must be a torch.ByteTensor". It only ever
failed on the cluster (job 421721) because every laptop test resumes on CPU, where there is nothing
to move. No GPU here, so a stand-in for a device tensor checks the `.cpu()` hop instead. No model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gdp.train.trainer import DetectorTrainer
from gdp.vqa.trainer import VLMTrainer


class _DeviceTensor:
    """Stands in for an off-CPU tensor: only `.cpu()` yields what set_rng_state accepts."""

    def __init__(self, real: torch.Tensor) -> None:
        self._real = real

    def cpu(self) -> torch.Tensor:
        return self._real


@pytest.mark.parametrize("trainer_cls", [VLMTrainer, DetectorTrainer])
def test_resume_moves_rng_state_to_cpu(trainer_cls, tmp_path, monkeypatch):
    real_rng = torch.get_rng_state()
    state = {
        "step": 7,
        "micro_step": 56,
        "optimizer": {},
        "scheduler": None,
        "rng_state": _DeviceTensor(real_rng),
    }
    monkeypatch.setattr(torch, "load", lambda *a, **k: state)

    seen = []
    monkeypatch.setattr(torch, "set_rng_state", lambda s: seen.append(s))

    fake_self = SimpleNamespace(
        device=torch.device("cpu"),
        scheduler=None,
        optimizer=SimpleNamespace(load_state_dict=lambda s: None),
    )
    trainer_cls.resume(fake_self, tmp_path)

    assert seen == [real_rng] and isinstance(seen[0], torch.Tensor)
    assert fake_self.step == 7
