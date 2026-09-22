"""Guards on the spec-03 resume and checkpoint-retention contract, without loading a real model.

A 12-epoch BDD100K run is ~209k steps and will outlive at least one SLURM walltime, so resume is
on the critical path, not an edge case. Before [SEQ-0130] it silently resumed onto the
*pretrained* weights (only step/optimizer were restored), and kept every ~2GB checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

import gdp.cli
from gdp.cli import app
from gdp.config import TrainingConfig
from gdp.paths import resolve
from gdp.train.trainer import DetectorTrainer

runner = CliRunner()


class _StubModel(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}")


class _StubProcessor:
    def save_pretrained(self, path: Path) -> None:
        pass


def test_save_checkpoint_keeps_only_the_newest_n(tmp_path):
    config = TrainingConfig(keep_last_checkpoints=2)
    trainer = DetectorTrainer(
        _StubModel(), _StubProcessor(), config, tmp_path, device=torch.device("cpu")
    )
    # Steps chosen so lexical order (1000 < 500) disagrees with numeric order.
    for step in (500, 1000, 1500, 2000):
        trainer.step = step
        trainer.save_checkpoint()

    kept = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert kept == ["checkpoint-1500", "checkpoint-2000"]


@pytest.mark.parametrize(
    "field,value",
    [("keep_last_checkpoints", 0), ("num_workers", -1), ("enc_loss_scale", -1.0)],
)
def test_training_config_rejects_bad_values(field, value):
    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value}).validate()


class _Loaded(Exception):
    pass


@pytest.mark.parametrize("resume", [None, "runs/03-finetune/x/checkpoint-500"])
def test_train_detector_loads_weights_from_the_resume_checkpoint(monkeypatch, resume):
    """`--resume` must load the checkpoint's weights; `trainer.resume` only restores step,
    optimizer and schedule on top of whatever model it is handed."""
    seen = []

    def fake_from_pretrained(source, **kwargs):
        seen.append(source)
        raise _Loaded  # stop before any run dir is created or training starts

    monkeypatch.setattr(
        gdp.cli.GroundingDinoForObjectDetection, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(gdp.cli.AutoProcessor, "from_pretrained", lambda *a, **k: None)

    args = ["train", "detector", "--dataset", "mini_bdd", "--split", "val"]
    if resume:
        args += ["--resume", resume]
    result = runner.invoke(app, args)

    assert isinstance(result.exception, _Loaded)
    assert seen == [resume or "IDEA-Research/grounding-dino-tiny"]


def test_finetune_job_never_resumes_from_the_overfit_gate_checkpoint():
    """The gate writes runs/03-finetune/<ts>/checkpoint-200; a bare glob over
    runs/03-finetune/*/checkpoint-* picked it up as the full run's resume point."""
    script = resolve("scripts/slurm/finetune_detector.slurm").read_text()
    assert "LATEST_CKPT=$(ls -dt runs/03-finetune/*/checkpoint-*" not in script
    assert '-newer "$STATE_DIR/overfit_gate.path"' in script
