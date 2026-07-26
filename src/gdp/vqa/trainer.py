"""`VLMTrainer`: the LoRA fine-tuning loop for Qwen2.5-VL (spec 07), mirroring `DetectorTrainer`'s
plain-PyTorch-loop shape (spec 03 design decision 3) — every training decision stays inspectable in
this file rather than behind `transformers.Trainer`.

Unlike `DetectorTrainer`, `grad_accum` is genuinely implemented rather than refused: a 3B VLM at
`batch_size=1` (one multi-image driving scene already fills a GPU at Qwen2.5-VL's dynamic
resolution) needs accumulation to reach a usable effective batch size. The optimizer steps only
every `grad_accum` micro-batches, with the loss scaled by `1/grad_accum` before `backward()` so
gradients sum to the same magnitude as a true batch of that size.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from gdp.config import VLMTrainingConfig
from gdp.train.logging_jsonl import JsonlLogger


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


class VLMTrainer:
    """Owns the LoRA-attached model, optimizer, schedule and step counter for one fine-tuning run.

    `model` must already be a `peft.PeftModel` produced by `gdp.vqa.lora.attach_lora` — this class
    does not attach LoRA itself, matching `gdp.vqa.lora`'s single responsibility.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: VLMTrainingConfig,
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
        self._micro_step = 0

        self.model.to(self.device)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Required alongside a frozen base model + LoRA: without it the input embeddings carry
            # no grad, and gradient checkpointing's recomputed backward graph has nothing to attach
            # to, so LoRA parameters silently get zero gradient.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer.zero_grad()
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.logger = JsonlLogger(self.run_dir / "train_log.jsonl")

        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
            import wandb

            self._wandb = wandb.init(project="gdp-finetune-vlm", dir=str(self.run_dir))

    def build_scheduler(self, num_training_steps: int) -> None:
        """`num_training_steps` counts **optimizer steps** (post-accumulation), not micro-batches —
        the caller (`gdp train vlm`) is responsible for dividing by `grad_accum` first."""
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One micro-batch forward/backward. Steps the optimizer (and advances `self.step`) only
        every `grad_accum` calls.

        A non-finite loss raises immediately, before `backward()` and before any optimizer step
        (design decision matching `DetectorTrainer`): a NaN gradient poisons every LoRA parameter
        it touches, and a checkpoint saved after that is worse than no checkpoint at all.
        """
        batch = _move_to_device(batch, self.device)
        self.model.train()

        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = self.model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at micro-step {self._micro_step}: {loss.item()!r} — aborting "
                "rather than accumulating a poisoned gradient"
            )

        (loss / self.config.grad_accum).backward()
        self._micro_step += 1

        record: dict[str, float] = {
            "micro_step": self._micro_step,
            "step": self.step,
            "loss": loss.item(),
        }

        if self._micro_step % self.config.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), self.config.max_grad_norm
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1
            record["step"] = self.step
            record["lr"] = self.optimizer.param_groups[-1]["lr"]
            record["grad_norm"] = float(grad_norm)

            if self.step % self.config.log_every == 0:
                self.logger.log(record)
                if self._wandb is not None:
                    self._wandb.log(record, step=self.step)

        return record

    def save_checkpoint(self) -> Path:
        """Writes the **adapter only** (`PeftModel.save_pretrained` never dumps the frozen 3B
        base), plus the processor and trainer state — never a `metrics.json` (H2, spec 08's
        filename, no evaluation happens here)."""
        ckpt_dir = self.run_dir / f"checkpoint-{self.step}"
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        torch.save(
            {
                "step": self.step,
                "micro_step": self._micro_step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rng_state": torch.get_rng_state(),
            },
            ckpt_dir / "trainer_state.pt",
        )
        return ckpt_dir

    def resume(self, checkpoint_dir: str | Path) -> None:
        """Restore step/optimizer/scheduler/RNG. Call this **after** `build_scheduler`, exactly as
        `DetectorTrainer.resume` requires — with no scheduler built yet, the saved LR-schedule
        position has nowhere to go, and a resumed run would silently restart cosine warmup from
        step 0 every time SLURM preempted it.
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
        self._micro_step = state["micro_step"]
