"""Training-loop coverage for `gdp.vqa.trainer.VLMTrainer` (spec 07 task 6).

Tiny random `Qwen2_5_VLForConditionalGeneration` + LoRA, real forward/backward, on the M4's CPU
(CLAUDE.md §4 — training never runs in-session, but exercising a few steps of a ~10M-parameter
model on synthetic data is not a training run). `model_heavy` because it constructs the model class
and drives the real processor. Run deliberately with:

    uv run pytest -m model_heavy tests/test_vqa_train_loop.py
"""

from __future__ import annotations

import json

import pytest
import torch

from gdp.config import Config
from gdp.data.drivelm import convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.dataset import DriveLMVQADataset, VQACollator
from gdp.vqa.lora import attach_lora
from gdp.vqa.trainer import VLMTrainer

pytestmark = pytest.mark.model_heavy


def make_tiny_qwen_vl():
    """Same shrunken architecture as `tests/test_vqa_lora.py`'s helper — duplicated rather than
    cross-imported (plan's stated alternative to a shared `tests/fixtures/tiny_qwen_vl.py` module,
    and this repo has no precedent for treating `tests/fixtures/` as an importable package)."""
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

    text_config = Qwen2_5_VLTextConfig(
        vocab_size=152064,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=151643,
        eos_token_id=151645,
        rope_parameters={
            "type": "mrope",
            "mrope_section": [2, 1, 1],
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        spatial_merge_size=2,
        fullatt_block_indexes=(0, 1),
    )
    config = Qwen2_5_VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    return Qwen2_5_VLForConditionalGeneration(config)


@pytest.fixture(scope="module")
def qwen_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


@pytest.fixture(scope="module")
def single_batch(qwen_processor):
    """One real, fully encoded QA pair from the `mini_drivelm` fixture, collated to batch size 1 —
    module-scoped so the fixture's processor round-trip runs once, not once per test."""
    from gdp.config import load_config

    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    records, _, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    dataset = DriveLMVQADataset(records[:1], qwen_processor, cfg.nuscenes_root_path(), num_views=1)
    return VQACollator(qwen_processor)([dataset[0]])


def _make_trainer(qwen_processor, run_dir, *, grad_accum=1, lr=1e-2, gradient_checkpointing=False):
    cfg = Config()
    cfg.vlm_training.lr = lr
    cfg.vlm_training.grad_accum = grad_accum
    cfg.vlm_training.gradient_checkpointing = gradient_checkpointing
    cfg.vlm_training.log_every = 1
    model = attach_lora(make_tiny_qwen_vl(), cfg)
    return VLMTrainer(model, qwen_processor, cfg.vlm_training, run_dir, device=torch.device("cpu"))


def test_three_steps_finite_and_loss_decreases(qwen_processor, single_batch, tmp_path):
    trainer = _make_trainer(qwen_processor, tmp_path, grad_accum=1, lr=1e-2)
    trainer.build_scheduler(num_training_steps=3)

    losses = []
    for _ in range(3):
        record = trainer.train_step(single_batch)
        assert torch.isfinite(torch.tensor(record["loss"]))
        losses.append(record["loss"])

    assert losses[-1] < losses[0]
    log_lines = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 3  # grad_accum=1, log_every=1 -> every step logged


def test_vision_tower_has_no_gradient_after_backward(qwen_processor, single_batch, tmp_path):
    trainer = _make_trainer(qwen_processor, tmp_path)
    trainer.build_scheduler(num_training_steps=1)
    trainer.train_step(single_batch)

    for name, param in trainer.model.named_parameters():
        if ".visual." in name:
            assert param.grad is None, f"{name} has a gradient — LoRA leaked into the vision tower"


def test_nan_loss_aborts_before_optimizer_step(qwen_processor, single_batch, tmp_path, monkeypatch):
    trainer = _make_trainer(qwen_processor, tmp_path)
    trainer.build_scheduler(num_training_steps=1)
    before = {
        name: param.detach().clone()
        for name, param in trainer.model.named_parameters()
        if param.requires_grad
    }

    real_forward = trainer.model.forward

    def poisoned_forward(*args, **kwargs):
        outputs = real_forward(*args, **kwargs)
        outputs.loss = outputs.loss * float("nan")
        return outputs

    monkeypatch.setattr(trainer.model, "forward", poisoned_forward)

    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train_step(single_batch)

    assert trainer.step == 0
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert torch.equal(param.detach(), before[name])


def test_grad_accum_steps_optimizer_exact_count(qwen_processor, single_batch, tmp_path):
    trainer = _make_trainer(qwen_processor, tmp_path, grad_accum=3)
    trainer.build_scheduler(num_training_steps=2)

    n_micro_batches = 7
    for _ in range(n_micro_batches):
        trainer.train_step(single_batch)

    assert trainer.step == n_micro_batches // 3


def test_checkpoint_resume_restores_state_on_a_freshly_built_model(
    qwen_processor, single_batch, tmp_path
):
    from peft import PeftModel

    trainer = _make_trainer(qwen_processor, tmp_path / "first", grad_accum=1, lr=1e-2)
    trainer.build_scheduler(num_training_steps=5)
    for _ in range(2):
        trainer.train_step(single_batch)
    ckpt_dir = trainer.save_checkpoint()
    saved_step = trainer.step
    saved_lr = trainer.optimizer.param_groups[0]["lr"]

    # `is_trainable=True` is required: PEFT's default `from_pretrained` loads adapters in
    # inference mode (every LoRA param frozen), which would leave the resumed optimizer with an
    # empty parameter list.
    resumed_model = PeftModel.from_pretrained(make_tiny_qwen_vl(), ckpt_dir, is_trainable=True)
    resumed_trainer = VLMTrainer(
        resumed_model,
        qwen_processor,
        trainer.config,
        tmp_path / "resumed",
        device=torch.device("cpu"),
    )
    resumed_trainer.build_scheduler(num_training_steps=5)
    resumed_trainer.resume(ckpt_dir)

    assert resumed_trainer.step == saved_step
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
