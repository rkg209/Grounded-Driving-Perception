from __future__ import annotations

import pytest
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from gdp.config import TrainingConfig
from gdp.data.core import load_dataset
from gdp.train.dataset import DetectionCollator
from gdp.train.trainer import DetectorTrainer

FIXTURE_ANNOTATIONS = "tests/fixtures/mini_bdd/annotations.json"
FIXTURE_ROOT = "tests/fixtures/mini_bdd"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_ROOT)


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture
def model():
    return GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)


def _batch(dataset, processor):
    collator = DetectionCollator(processor, dataset.prompt())
    return collator(list(dataset.samples[:2]))


def test_three_steps_loss_finite_and_decreasing(dataset, processor, model, tmp_path):
    """The laptop's job (design decision 5): prove the plumbing, not real convergence."""
    config = TrainingConfig(freeze_text_encoder=True, gradient_checkpointing=False, log_every=1)
    trainer = DetectorTrainer(model, processor, config, tmp_path, device=torch.device("cpu"))
    trainer.build_scheduler(num_training_steps=3)

    batch = _batch(dataset, processor)
    losses = []
    for _ in range(3):
        record = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(record["loss"]))
        losses.append(record["loss"])

    assert losses[-1] < losses[0]
    log_lines = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 3


def test_freeze_text_encoder_stops_its_gradients(dataset, processor, model, tmp_path):
    config = TrainingConfig(freeze_text_encoder=True, log_every=1)
    trainer = DetectorTrainer(model, processor, config, tmp_path, device=torch.device("cpu"))
    trainer.build_scheduler(num_training_steps=1)

    for name, param in trainer.model.named_parameters():
        if name.startswith("model.text_backbone"):
            assert not param.requires_grad

    batch = _batch(dataset, processor)
    trainer.train_step(batch)
    for name, param in trainer.model.named_parameters():
        if name.startswith("model.text_backbone"):
            assert param.grad is None


def test_nan_loss_aborts(dataset, processor, model, tmp_path, monkeypatch):
    config = TrainingConfig(log_every=1)
    trainer = DetectorTrainer(model, processor, config, tmp_path, device=torch.device("cpu"))
    trainer.build_scheduler(num_training_steps=1)

    class _NanOutputs:
        loss = torch.tensor(float("nan"))
        loss_dict = {}

    monkeypatch.setattr(trainer.model, "forward", lambda **kwargs: _NanOutputs())
    batch = _batch(dataset, processor)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train_step(batch)


def test_checkpoint_save_and_resume_restores_step_and_optimizer(
    dataset, processor, model, tmp_path
):
    config = TrainingConfig(log_every=1)
    trainer = DetectorTrainer(model, processor, config, tmp_path, device=torch.device("cpu"))
    trainer.build_scheduler(num_training_steps=2)

    batch = _batch(dataset, processor)
    trainer.train_step(batch)
    ckpt_dir = trainer.save_checkpoint()
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "trainer_state.pt").is_file()
    assert (ckpt_dir / "config.json").is_file()

    saved_step = trainer.step
    saved_opt_state = trainer.optimizer.state_dict()

    resumed_model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)
    resumed = DetectorTrainer(
        resumed_model, processor, config, tmp_path, device=torch.device("cpu")
    )
    resumed.build_scheduler(num_training_steps=2)
    resumed.resume(ckpt_dir)

    assert resumed.step == saved_step
    assert len(resumed.optimizer.state_dict()["state"]) == len(saved_opt_state["state"])
