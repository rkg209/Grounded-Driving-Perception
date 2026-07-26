"""`DriveLMRecord` -> Qwen2.5-VL chat messages, through the model's own chat template.

No hand-rolled `<|im_start|>...` prompt string anywhere (spec 06 design decision 8) — a
hand-written template is the classic silent-mismatch bug between how a model is fine-tuned and how
it is evaluated. `build_messages` is pure and cheap to unit-test against a plain dict; only
`format_example`'s round-trip through the real processor needs a model download
(`@pytest.mark.model_heavy`, CLAUDE.md §4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gdp.data.drivelm import DriveLMRecord


def build_messages(record: DriveLMRecord, num_views: int = 1) -> list[dict[str, Any]]:
    """The Qwen2.5-VL message list for one QA pair: `num_views` images (in `record.view_order`
    order, so `CAM_FRONT` is always first) + the question, then the answer as the assistant turn.

    `num_views` is a training-tractability knob (`vlm.num_views`, spec 06 design decision 5) — the
    record itself always carries all six camera paths regardless of how many are used here.
    """
    if not 1 <= num_views <= len(record.view_order):
        raise ValueError(f"num_views must be in [1, {len(record.view_order)}], got {num_views}")
    views = record.view_order[:num_views]
    image_content = [{"type": "image", "image": record.image_paths[v]} for v in views]
    return [
        {"role": "user", "content": [*image_content, {"type": "text", "text": record.question}]},
        {"role": "assistant", "content": [{"type": "text", "text": record.answer}]},
    ]


def format_example(processor: Any, record: DriveLMRecord, num_views: int = 1) -> dict[str, Any]:
    """Chat-format one record through the real processor's `apply_chat_template`.

    Returns the templated text plus the image paths it references, in `view_order`'s order —
    training/eval code opens the actual image bytes; this function only owns the text side.
    """
    messages = build_messages(record, num_views)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    views = record.view_order[:num_views]
    return {
        "qa_id": record.qa_id,
        "text": text,
        "image_paths": [record.image_paths[v] for v in views],
    }


def load_jsonl(path: str | Path) -> list[DriveLMRecord]:
    """Read a `{train,val}.jsonl` written by `gdp.data.drivelm.write_jsonl` back into records."""
    records = []
    for line in Path(path).read_text().splitlines():
        records.append(DriveLMRecord(**json.loads(line)))
    return records
