from __future__ import annotations

import pytest
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from gdp.config import TrainingConfig
from gdp.data.core import load_dataset
from gdp.train.dataset import DetectionCollator
from gdp.train.trainer import DetectorTrainer, weighted_loss

pytestmark = pytest.mark.model_heavy

FIXTURE_ANNOTATIONS = "tests/fixtures/mini_bdd/annotations.json"
FIXTURE_ROOT = "tests/fixtures/mini_bdd"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_ROOT)


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def base_model():
    return GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)


@pytest.fixture
def model(base_model):
    """Reuse the module's one loaded model instead of reloading per test (was 4 reloads,
    each ~700MB — the repeated-load pattern that drove the laptop to 40GB during a full
    pytest run). Reset requires_grad so a previous test's freeze_text_encoder=True doesn't
    leak into a test that expects everything trainable (DetectorTrainer only ever sets
    requires_grad=False, never back to True — see trainer.py)."""
    for param in base_model.parameters():
        param.requires_grad = True
    return base_model


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


def test_weighted_loss_matches_hf_at_unit_encoder_scale(dataset, processor, model):
    """Guards `weighted_loss`'s mirrored weight table against drift in HF's loss: at
    enc_loss_scale=1.0 it must reproduce `outputs.loss`; at 0 it must be O(1) ([SEQ-0132])."""
    batch = _batch(dataset, processor)
    with torch.no_grad():
        outputs = model(**batch)

    at_hf = weighted_loss(outputs.loss_dict, model.config, 1.0)
    assert float(at_hf) == pytest.approx(float(outputs.loss), rel=1e-5)
    assert float(weighted_loss(outputs.loss_dict, model.config, 0.0)) < 10.0


def test_hf_encoder_box_losses_carry_no_gradient(dataset, processor, model):
    """The premise of enc_loss_scale=0 ([SEQ-0135]): HF computes the two-stage encoder box losses
    on detached proposals, so they are constants to the optimizer, while the decoder's are not.
    If a transformers upgrade stops detaching, this fails and the decision must be revisited."""
    model.train()
    outputs = model(**_batch(dataset, processor))
    loss_dict = outputs.loss_dict
    assert loss_dict["loss_bbox_enc"].grad_fn is None
    assert loss_dict["loss_giou_enc"].grad_fn is None
    assert loss_dict["loss_bbox"].grad_fn is not None
    assert loss_dict["loss_giou"].grad_fn is not None


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
        # The trainer builds its loss from loss_dict (weighted_loss), not from `.loss`.
        loss = torch.tensor(float("nan"))
        loss_dict = {"loss_ce": torch.tensor(float("nan"))}

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

    # Must be a genuinely independent instance (not the shared `base_model`) to prove
    # `resume()` restores state rather than the test coincidentally reusing live weights —
    # the one deliberate extra reload left after deduping the other four.
    resumed_model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)
    resumed = DetectorTrainer(
        resumed_model, processor, config, tmp_path, device=torch.device("cpu")
    )
    resumed.build_scheduler(num_training_steps=2)
    resumed.resume(ckpt_dir)

    assert resumed.step == saved_step
    assert len(resumed.optimizer.state_dict()["state"]) == len(saved_opt_state["state"])
