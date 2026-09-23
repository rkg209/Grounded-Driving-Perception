"""LoRA attachment for Qwen2.5-VL's language-model attention/MLP projections, freeze asserted.

**Trap 4** (verified against the installed `peft==0.19.1` / `transformers==5.13.1`): Qwen2.5-VL's
vision tower MLP (`Qwen2_5_VLMLP` inside `model.visual.blocks.*`) uses the *same*
`gate_proj`/`up_proj`/`down_proj` names as the language model's MLP
(`model.language_model.layers.*.mlp.*`) — confirmed by listing `named_parameters()` on a real
(shrunken) model: `model.visual.blocks.0.mlp.gate_proj.weight` and
`model.language_model.layers.0.mlp.gate_proj.weight` both exist. PEFT's
`tuners_utils.check_target_module_exists`, when `target_modules` is a plain list of short names,
matches by `key.endswith(f".{name}")` — a bare `target_modules=["gate_proj", ...]` list would
attach LoRA to the vision tower we claim to freeze, and the run would train fine, with a falling
loss and no error, so nothing would catch it. `LM_TARGET_RE` is instead a **regex string**: PEFT
matches a string `target_modules` with `re.fullmatch` against the full parameter path
(`match_target_against_key`), so anchoring it on `^model\\.language_model\\.` makes a vision-tower
match structurally impossible rather than merely unlikely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model

from gdp.config import Config

LM_TARGET_RE = r"^model\.language_model\..*\.(q|k|v|o|gate|up|down)_proj$"


def attach_lora(model: Any, cfg: Config) -> Any:
    """Wrap `model` in a `peft.PeftModel` with LoRA on `LM_TARGET_RE`-matched language-model
    projections only, then assert the vision tower stayed frozen before returning."""
    lora_config = LoraConfig(
        r=cfg.vlm.lora_r,
        lora_alpha=cfg.vlm.lora_alpha,
        lora_dropout=cfg.vlm_training.lora_dropout,
        target_modules=LM_TARGET_RE,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    assert_vision_tower_frozen(peft_model)
    return peft_model


def load_lora_for_resume(model: Any, checkpoint_dir: str | Path) -> Any:
    """Wrap `model` with the LoRA adapter saved in `checkpoint_dir`, trainable.

    A resume must continue from the checkpoint's adapter weights. `VLMTrainer.resume` restores
    only step/optimizer/schedule, so attaching a *fresh* adapter (`attach_lora`) there would put
    step N's optimizer onto an untrained adapter and silently discard all prior training (the
    same bug fixed for the detector in progress_report [SEQ-0130])."""
    peft_model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=True)
    assert_vision_tower_frozen(peft_model)
    return peft_model


def assert_vision_tower_frozen(model: Any) -> None:
    """Raise if any parameter under `model.visual.` is trainable, or if any LoRA adapter name
    contains `.visual.` — the two-line check guarding trap 4's failure mode, which otherwise has
    no symptom until someone notices the vision tower drifted."""
    for name, param in model.named_parameters():
        if ".visual." not in name:
            continue
        if param.requires_grad:
            raise RuntimeError(
                f"vision tower parameter {name!r} is trainable — LoRA leaked into the frozen "
                "backbone (trap 4)"
            )
        if "lora" in name.lower():
            raise RuntimeError(f"a LoRA adapter is attached under the vision tower: {name!r}")


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    """`{trainable_params, total_params, trainable_pct}` — belongs in every run's
    `train_config.json` (H1: proves the run is LoRA *adaptation*, not "trained a VLM")."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }
