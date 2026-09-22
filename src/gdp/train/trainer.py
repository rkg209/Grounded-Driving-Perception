"""`DetectorTrainer`: the fine-tuning loop for Grounding-DINO (spec 03 design decision 3).

A plain PyTorch loop, not `transformers.Trainer` — every training decision (param groups, freezing,
schedule, NaN handling) is visible in this ~200-line file rather than behind a general-purpose
abstraction, which is exactly what makes it possible to explain and defend each choice (CLAUDE.md
H1: this is *adaptation*, and the adaptation's mechanics should be inspectable, not hidden).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    get_cosine_schedule_with_warmup,
)

from gdp.config import TrainingConfig
from gdp.train.logging_jsonl import JsonlLogger

BACKBONE_PREFIX = "model.backbone"
TEXT_BACKBONE_PREFIX = "model.text_backbone"


def _scalar(value: torch.Tensor | float) -> float:
    return value.detach().item() if isinstance(value, torch.Tensor) else float(value)


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "labels":
            moved[key] = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in label.items()}
                for label in value
            ]
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# HF's GroundingDinoForObjectDetectionLoss hardcodes the class-loss weight (2.0) rather than
# reading it from the model config; mirrored here so `weighted_loss` reproduces its total exactly
# when `enc_class_loss_weight` equals it (asserted in tests/test_train_loop.py).
HF_CLASS_LOSS_WEIGHT = 2.0


def weighted_loss(
    loss_dict: dict[str, torch.Tensor], model_config: Any, enc_class_loss_weight: float
) -> torch.Tensor:
    """HF's Grounding-DINO total loss, with the encoder-proposal class term reweighted.

    On the pretrained checkpoint the two-stage encoder's proposal scores are uncalibrated for
    HF's sigmoid focal loss: ~890 of 900 proposals score > 0.5, so `loss_ce_enc` is ~65,000 while
    every other term is < 1 ([SEQ-0132]). Under gradient clipping it then owns the whole update,
    pushed through the shared encoder the decoder reads. Our detection path never reads encoder
    logits (they only rank proposals), so spec 03 drops that term by default and keeps the rest.
    """
    base = {
        "loss_ce": HF_CLASS_LOSS_WEIGHT,
        "loss_bbox": model_config.bbox_loss_coefficient,
        "loss_giou": model_config.giou_loss_coefficient,
    }
    weights = dict(base)
    if model_config.two_stage:
        weights.update({f"{k}_enc": v for k, v in base.items()})
        weights["loss_ce_enc"] = enc_class_loss_weight
    if model_config.auxiliary_loss:
        for i in range(model_config.decoder_layers - 1):
            weights.update({f"{k}_{i}": v for k, v in base.items()})
    return sum(loss_dict[k] * w for k, w in weights.items() if k in loss_dict)


class DetectorTrainer:
    """Owns the model, optimizer, schedule and step counter for one fine-tuning run.

    `freeze_text_encoder=True` (the config default) sets `requires_grad=False` on every
    `model.text_backbone.*` parameter: the text encoder already embeds the fixed 10-class prompt
    well (spec 02's zero-shot baseline proves it can *find* the right tokens); the fine-tuning
    signal is in the vision backbone and fusion/decoder heads learning BDD100K's visual domain.
    """

    def __init__(
        self,
        model: GroundingDinoForObjectDetection,
        processor: GroundingDinoProcessor,
        config: TrainingConfig,
        run_dir: str | Path,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.step = 0

        # `train_step` performs one optimizer step per call, unconditionally — there is no
        # accumulation buffer anywhere in this loop. The config field exists (and is validated),
        # so a cluster run could set `grad_accum: 4` expecting an effective batch of 16 and get
        # an effective batch of 4 with no warning: a config that lies about what the run did.
        # Refuse the value we cannot honour rather than silently ignoring it.
        if config.grad_accum != 1:
            raise NotImplementedError(
                f"training.grad_accum={config.grad_accum} but gradient accumulation is not "
                "implemented — this loop steps the optimizer every batch. Use training.batch_size "
                "to change the effective batch, or implement accumulation in train_step first."
            )

        self.model.to(self.device)
        if config.freeze_text_encoder:
            for name, param in self.model.named_parameters():
                if name.startswith(TEXT_BACKBONE_PREFIX):
                    param.requires_grad = False
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.optimizer = self._build_optimizer()
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.logger = JsonlLogger(self.run_dir / "train_log.jsonl")

        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
            import wandb

            self._wandb = wandb.init(project="gdp-finetune-detector", dir=str(self.run_dir))

    def _build_optimizer(self) -> AdamW:
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith(BACKBONE_PREFIX) else other_params).append(param)
        return AdamW(
            [
                {"params": backbone_params, "lr": self.config.lr * self.config.backbone_lr_mult},
                {"params": other_params, "lr": self.config.lr},
            ],
            weight_decay=self.config.weight_decay,
        )

    def build_scheduler(self, num_training_steps: int) -> None:
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One forward/backward/optimizer step. Returns the step's logged record.

        A non-finite loss raises immediately (design decision 4, acceptance 4) rather than
        stepping the optimizer on it: a NaN gradient poisons every parameter it touches, and a
        checkpoint saved after that is worse than no checkpoint at all.
        """
        batch = _move_to_device(batch, self.device)
        self.model.train()

        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = self.model(**batch)
            # Not `outputs.loss`: see `weighted_loss` for why the encoder class term is reweighted.
            loss = weighted_loss(
                outputs.loss_dict, self.model.config, self.config.enc_class_loss_weight
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {self.step}: {loss.item()!r} — aborting rather than "
                "training on a poisoned gradient"
            )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), self.config.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1

        loss_dict = outputs.loss_dict
        record = {
            "step": self.step,
            # Wall clock, so step time (and so a run's walltime) can be read off the gate's log.
            "time": time.time(),
            "loss": loss.item(),
            "loss_ce": _scalar(loss_dict.get("loss_ce", 0.0)),
            "loss_bbox": _scalar(loss_dict.get("loss_bbox", 0.0)),
            "loss_giou": _scalar(loss_dict.get("loss_giou", 0.0)),
            # Logged even at weight 0, so a drift in the untrained term is visible, not hidden.
            "loss_ce_enc": _scalar(loss_dict.get("loss_ce_enc", 0.0)),
            "loss_bbox_enc": _scalar(loss_dict.get("loss_bbox_enc", 0.0)),
            "loss_giou_enc": _scalar(loss_dict.get("loss_giou_enc", 0.0)),
            "lr": self.optimizer.param_groups[-1]["lr"],
            "grad_norm": float(grad_norm),
        }
        if self.step % self.config.log_every == 0:
            self.logger.log(record)
            if self._wandb is not None:
                self._wandb.log(record, step=self.step)
        return record

    def save_checkpoint(self) -> Path:
        ckpt_dir = self.run_dir / f"checkpoint-{self.step}"
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rng_state": torch.get_rng_state(),
            },
            ckpt_dir / "trainer_state.pt",
        )
        self._prune_checkpoints()
        return ckpt_dir

    def _prune_checkpoints(self) -> None:
        """Keep only the newest `keep_last_checkpoints` in this run dir. Only the newest is ever
        resumed from; a full BDD100K run otherwise writes ~800GB of checkpoints to scratch."""
        ckpts = sorted(
            (p for p in self.run_dir.glob("checkpoint-*") if p.name.split("-", 1)[1].isdigit()),
            key=lambda p: int(p.name.split("-", 1)[1]),
        )
        for stale in ckpts[: -self.config.keep_last_checkpoints]:
            shutil.rmtree(stale)

    def resume(self, checkpoint_dir: str | Path) -> None:
        """Restore step/optimizer/scheduler/RNG. Call this **after** `build_scheduler`.

        Weights are *not* restored here: the model must already have been loaded from
        `checkpoint_dir` via `from_pretrained(checkpoint_dir)`, which is what the CLI's `--resume`
        does. Resuming onto the pretrained weights would carry the step counter and optimizer
        over to an untrained model and silently discard everything before the checkpoint.

        The order matters and is enforced rather than documented: with no scheduler built yet,
        the saved LR-schedule position has nowhere to go, and a resumed run would silently
        restart its cosine warmup from step 0 every time SLURM preempted it — a training curve
        with a sawtooth in it that nobody would attribute to a load order.
        """
        state = torch.load(Path(checkpoint_dir) / "trainer_state.pt", map_location=self.device)
        if state["scheduler"] is not None and self.scheduler is None:
            raise RuntimeError(
                f"{checkpoint_dir} carries scheduler state but no scheduler has been built — "
                "call build_scheduler(total_steps) before resume(), or the LR schedule restarts "
                "from warmup step 0"
            )
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_state"])
        self.step = state["step"]
