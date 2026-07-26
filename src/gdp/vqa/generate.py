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


def generate_with_stats(
    processor: Any,
    model: Any,
    record: DriveLMRecord,
    *,
    nuscenes_root: str | Path,
    device: torch.device,
    num_views: int = 1,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    num_beams: int = 1,
) -> dict[str, Any]:
    """Generate an answer to `record.question` — the user turn only, never `record.answer`.

    `do_sample`/`num_beams` default to greedy decoding and are passed through explicitly (rather
    than left to inherit the checkpoint's `generation_config`) so this demo path and spec 08's
    `gdp.vqa.predict` share one decoding path (spec 08 design decision 3) — pinning it here, not
    just at the `VQAEvalConfig` layer, so a future default change in `generation_config` can never
    silently make base vs. fine-tuned runs incomparable (H8).

    Returns `{"text", "num_input_tokens", "num_generated_tokens"}` — `predict_split` (spec 08 task
    3) needs the token counts for `predictions.jsonl`'s per-line schema; `answer()` below is a thin
    wrapper for the spec-07 demo CLI, which only needs the text.
    """
    user_message = build_messages(record, num_views)[:1]
    text = processor.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    views = record.view_order[:num_views]
    images = [Image.open(Path(nuscenes_root) / record.image_paths[v]).convert("RGB") for v in views]

    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
        )
    num_input_tokens = inputs["input_ids"].shape[1]
    new_tokens = generated[:, num_input_tokens:]
    decoded = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return {
        "text": decoded,
        "num_input_tokens": int(num_input_tokens),
        "num_generated_tokens": int(new_tokens.shape[1]),
    }


def answer(
    processor: Any,
    model: Any,
    record: DriveLMRecord,
    *,
    nuscenes_root: str | Path,
    device: torch.device,
    num_views: int = 1,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    num_beams: int = 1,
) -> str:
    """Thin wrapper around `generate_with_stats` returning just the decoded text (spec 07's demo
    CLI never needs token counts — spec 08 owns every number that could be called a result)."""
    return generate_with_stats(
        processor,
        model,
        record,
        nuscenes_root=nuscenes_root,
        device=device,
        num_views=num_views,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )["text"]
