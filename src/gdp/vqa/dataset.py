"""`DriveLMRecord`s -> batched Qwen2.5-VL training inputs, with an answer-only loss mask.

`chat.build_messages` builds the message list; this module is where "training/eval code opens the
actual image bytes" (`chat.py`'s module docstring) actually happens, and where `gdp.vqa.labels`'s
assertion-backed mask gets applied per record. A record whose answer boundary can't be resolved or
whose sequence is too long is dropped with a counted reason (H7) at dataset-construction time, never
silently truncated (`truncation=` is never passed to the tokenizer) or stripped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from gdp.data.drivelm import DriveLMRecord
from gdp.vqa.chat import build_messages
from gdp.vqa.labels import (
    IGNORE_INDEX,
    AnswerBoundaryUnresolvable,
    assert_answer_only_mask,
    build_answer_labels,
    expand_image_placeholders,
    prompt_token_length,
)

DROP_REASONS = ("seq_too_long", "answer_boundary_unresolvable")


@dataclass(frozen=True)
class VQAExample:
    """One encoded, label-masked QA pair. Un-batched: `VQACollator` pads/concats across examples."""

    qa_id: str
    input_ids: torch.Tensor  # [seq_len]
    attention_mask: torch.Tensor  # [seq_len]
    labels: torch.Tensor  # [seq_len]
    mm_token_type_ids: torch.Tensor  # [seq_len]
    pixel_values: torch.Tensor  # [sum_patches_over_images, 1176]
    image_grid_thw: torch.Tensor  # [num_images, 3]


class DriveLMVQADataset(TorchDataset):
    """Encodes every record eagerly at construction (not lazily per `__getitem__`): the only way to
    know whether a record is `seq_too_long` or `answer_boundary_unresolvable` is to fully run it
    through the processor, so there is no cheaper filtering pass to do first. This trades memory for
    simplicity — fine for laptop fixtures, not a design that claims to scale: one CAM_FRONT example
    holds ~34 MB of float32 pixels, so 64G of RAM caps eager encoding at ~1,800 pairs (job 419933
    died with a CPU OOM on 321,893).

    `lazy=True` encodes in `__getitem__` instead, so RAM is bounded by the batch, not the dataset.
    The price is that drops are only discovered when an example is drawn: a dropped record is
    replaced by the next one in order (wrapping) so the loader never sees a hole, and `drop_stats`
    counts drops *encountered so far* — read it after training, not before. `len(ds)` is then the
    number of records, not the number of usable ones.
    """

    def __init__(
        self,
        records: list[DriveLMRecord],
        processor: Any,
        nuscenes_root: str | Path,
        num_views: int = 1,
        max_seq_len: int = 4096,
        lazy: bool = False,
    ) -> None:
        self.processor = processor
        self.nuscenes_root = Path(nuscenes_root)
        self.num_views = num_views
        self.max_seq_len = max_seq_len
        self.drop_stats: dict[str, int] = dict.fromkeys(DROP_REASONS, 0)

        self.lazy = lazy
        self._records = list(records)
        self._examples: list[VQAExample] = []
        if lazy:
            return
        for record in self._records:
            example = self._try_encode(record)
            if example is not None:
                self._examples.append(example)

    def _try_encode(self, record: DriveLMRecord) -> VQAExample | None:
        """The encoded example, or None (with the drop counted) if the record must be dropped."""
        try:
            example = self._encode(record)
        except AnswerBoundaryUnresolvable:
            self.drop_stats["answer_boundary_unresolvable"] += 1
            return None
        if example.input_ids.shape[0] > self.max_seq_len:
            self.drop_stats["seq_too_long"] += 1
            return None
        return example

    def __len__(self) -> int:
        return len(self._records) if self.lazy else len(self._examples)

    def __getitem__(self, index: int) -> VQAExample:
        if not self.lazy:
            return self._examples[index]
        n = len(self._records)
        for offset in range(n):
            example = self._try_encode(self._records[(index + offset) % n])
            if example is not None:
                return example
        raise RuntimeError(f"no usable QA pair in {n} records (drop_stats={self.drop_stats})")

    def _encode(self, record: DriveLMRecord) -> VQAExample:
        messages = build_messages(record, self.num_views)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        views = record.view_order[: self.num_views]
        images = [
            Image.open(self.nuscenes_root / record.image_paths[v]).convert("RGB") for v in views
        ]

        encoded = self.processor(text=[full_text], images=images, return_tensors="pt")
        expanded_prompt = expand_image_placeholders(
            self.processor, prompt_text, encoded["image_grid_thw"]
        )
        prompt_len = prompt_token_length(self.processor, expanded_prompt, encoded)
        labels = build_answer_labels(encoded, prompt_len)
        assert_answer_only_mask(self.processor, encoded, labels, record.answer, record.qa_id)

        return VQAExample(
            qa_id=record.qa_id,
            input_ids=encoded["input_ids"][0],
            attention_mask=encoded["attention_mask"][0],
            labels=labels[0],
            mm_token_type_ids=encoded["mm_token_type_ids"][0],
            pixel_values=encoded["pixel_values"],
            image_grid_thw=encoded["image_grid_thw"],
        )


class VQACollator:
    """Batches `VQAExample`s: right-pads the text side, `torch.cat`s the packed pixel side.

    Row order in the concatenated `pixel_values`/`image_grid_thw` is load-bearing (design decision
    6): `get_image_features` splits the vision tower's output by `image_grid_thw.prod(-1) //
    merge_size**2` and `get_rope_index` walks `image_grid_thw` as one iterator across the whole
    batch, so reordering either tensor silently pairs one example's patches with another's grid.
    `torch.cat` on dim 0, in example order, is the only correct operation here — never a stack,
    never a sort.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, examples: list[VQAExample]) -> dict[str, torch.Tensor]:
        max_len = max(example.input_ids.shape[0] for example in examples)

        input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
        mm_token_type_ids = torch.zeros((len(examples), max_len), dtype=torch.long)

        for row, example in enumerate(examples):
            n = example.input_ids.shape[0]
            input_ids[row, :n] = example.input_ids
            attention_mask[row, :n] = example.attention_mask
            labels[row, :n] = example.labels
            mm_token_type_ids[row, :n] = example.mm_token_type_ids

        pixel_values = torch.cat([example.pixel_values for example in examples], dim=0)
        image_grid_thw = torch.cat([example.image_grid_thw for example in examples], dim=0)

        merge_size = self.processor.image_processor.merge_size
        n_image_tokens = (input_ids == self.processor.image_token_id).sum()
        n_expected = (image_grid_thw.prod(-1) // merge_size**2).sum()
        if n_image_tokens != n_expected:
            raise ValueError(
                f"image-token/patch mismatch: {n_image_tokens} image tokens in input_ids, "
                f"{n_expected} expected from image_grid_thw — the get_placeholder_mask invariant "
                "(modeling_qwen2_5_vl.py) would fail mid-forward"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
