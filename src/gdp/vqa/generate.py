"""Adapter loading + a single generation check — a **demo, not a result** (H2).

No accuracy, no scoring, nothing that could be mistaken for a metric: spec 08 owns every number
that could be called a VQA result. `answer()` exists only to prove the adapter loads back and
produces coherent text on a fixture image (spec 07 acceptance criterion 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gdp.data.drivelm import DriveLMRecord
from gdp.vqa.chat import build_messages


def load_adapter(base_id: str, adapter_dir: str | Path, *, device: torch.device) -> Any:
    """Load the frozen base + LoRA adapter for inference (`is_trainable=False`, the default)."""
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()
    return model


def answer(
    processor: Any,
    model: Any,
    record: DriveLMRecord,
    *,
    nuscenes_root: str | Path,
    device: torch.device,
    num_views: int = 1,
    max_new_tokens: int = 128,
) -> str:
    """Generate an answer to `record.question` — the user turn only, never `record.answer`."""
    user_message = build_messages(record, num_views)[:1]
    text = processor.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    views = record.view_order[:num_views]
    images = [Image.open(Path(nuscenes_root) / record.image_paths[v]).convert("RGB") for v in views]

    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
