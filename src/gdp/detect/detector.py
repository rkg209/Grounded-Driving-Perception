"""Grounding-DINO model/processor loading, span→class assignment, and box-format conversion.

Grounding-DINO has no class head: each query's `logits` score the prompt's *text tokens*, not
class ids. The "token-span trap" (`detection-eval` skill) is decoding those logits to phrase
strings and matching them back to classes by string comparison — fragile, and exactly the path
this module avoids. Instead `assign_classes` aggregates each class's own tokens via
`PromptSpans.positive_map()` (a masked mean, the MDETR/Grounding-DINO convention) and takes the
argmax over classes. `assert_span_round_trip` is the runtime guard that catches a
processor/tokenizer mismatch loudly, before it would otherwise silently mislabel every box.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    PreTrainedTokenizerBase,
)

from gdp.config import DetectorConfig
from gdp.data.core import Sample
from gdp.data.prompt import PromptSpans
from gdp.detect.predictions import Detection
from gdp.seed import select_device


def _chunks(items: Sequence[Sample], size: int) -> Iterator[Sequence[Sample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def assert_span_round_trip(
    spans: PromptSpans, classes: Sequence[str], tokenizer: PreTrainedTokenizerBase
) -> None:
    """Fail loudly if decoding a class's token span doesn't return its name.

    Guards the alignment between `PromptSpans` (built with `tokenizer`) and whatever
    tokenization the model's processor actually feeds the model. If the processor pads or
    formats the prompt differently than `PromptSpans` assumed, this raises instead of silently
    mislabelling every prediction (the "token-span trap").
    """
    for idx, name in enumerate(classes):
        decoded = spans.decode_span(idx, tokenizer=tokenizer)
        if decoded != name:
            raise RuntimeError(
                f"span→class round-trip failed for class {idx} ({name!r}): decoded {decoded!r}. "
                "The processor's tokenizer produced a different tokenization than PromptSpans "
                "assumed."
            )


def assign_classes(logits: torch.Tensor, positive_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-query class assignment via masked-mean aggregation over each class's tokens.

    `logits`: `[num_queries, text_len]`, raw (pre-sigmoid) scores over the prompt's text tokens
    — Grounding-DINO's `outputs.logits[0]`.
    `positive_map`: `[num_classes, text_len]`, L1-normalized rows from `PromptSpans.positive_map()`
    — `positive_map[c]` is `1/span_len` over class `c`'s tokens, 0 elsewhere, so
    `scores @ positive_map.T` is exactly the masked mean of a query's token scores over each
    class's span (not the decoded-phrase-string path the token-span trap warns against).

    Returns `(class_ids, scores)`, each length `num_queries`: the argmax class per query and its
    aggregated score.
    """
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    text_len = min(scores.shape[1], positive_map.shape[1])
    class_scores = scores[:, :text_len] @ positive_map[:, :text_len].T  # [num_queries, num_classes]
    class_ids = class_scores.argmax(axis=1)
    box_scores = class_scores[np.arange(len(class_ids)), class_ids]
    return class_ids, box_scores


def convert_boxes_cxcywh_norm_to_xyxy_abs(
    boxes: torch.Tensor, width: int, height: int
) -> np.ndarray:
    """Grounding-DINO's `pred_boxes` (normalized cxcywh) → absolute xyxy pixel coordinates.

    Cheap guard: normalized coordinates must lie in `[0, 1]` — a `640.0` in this tensor means a
    caller passed already-absolute boxes, which would silently corrupt every downstream mAP
    (detection-eval skill §2: assert box format at every boundary).
    """
    arr = boxes.detach().cpu().numpy()
    if arr.size and (arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4):
        raise ValueError(
            f"expected normalized cxcywh boxes in [0, 1], got range [{arr.min()}, {arr.max()}]"
        )
    cx, cy, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    x0 = (cx - w / 2) * width
    y0 = (cy - h / 2) * height
    x1 = (cx + w / 2) * width
    y1 = (cy + h / 2) * height
    return np.stack([x0, y0, x1, y1], axis=-1)


class GroundingDinoDetector:
    """Loads `IDEA-Research/grounding-dino-tiny` (or config-specified checkpoint) for inference.

    `disable_custom_kernels` must be True before ONNX export (spec 05); Grounding-DINO's
    MS-deformable-attention CUDA kernels have no ONNX equivalent. Left False elsewhere for speed.

    `prompt_spans` is built from the *processor's own tokenizer* (not the default bert instance
    `gdp.data.prompt.get_tokenizer()` returns) and round-trip-asserted at construction time, per
    this spec's design decision 1: the token indices must match what the model actually sees.
    """

    def __init__(
        self, config: DetectorConfig, classes: Sequence[str], *, device: str = "auto"
    ) -> None:
        self.config = config
        self.classes = tuple(classes)
        self.device: torch.device = select_device(device)
        self.processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(config.model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            config.model_id, disable_custom_kernels=config.disable_custom_kernels
        )
        self.model.to(self.device)
        self.model.eval()

        self.prompt = config.prompt(list(self.classes))
        self.prompt_spans = PromptSpans.from_classes(
            self.classes, tokenizer=self.processor.tokenizer
        )
        assert_span_round_trip(self.prompt_spans, self.classes, self.processor.tokenizer)
        self._positive_map = self.prompt_spans.positive_map()

    def detect_images(
        self, samples: Sequence[Sample], *, box_threshold: float, batch_size: int = 8
    ) -> list[Detection]:
        """Batched inference: each `Sample`'s image → its kept `Detection`s.

        One forward pass per `batch_size`-sized chunk (the same fixed text prompt broadcast to
        every image in the chunk). Boxes below `box_threshold` are dropped here, at the earliest
        point they can be — everything downstream (predictions.json, mAP) only ever sees kept
        detections.
        """
        detections: list[Detection] = []
        for chunk in _chunks(samples, batch_size):
            images = [Image.open(s.image_path).convert("RGB") for s in chunk]
            inputs = self.processor(
                images=images, text=[self.prompt] * len(images), return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            for i, sample in enumerate(chunk):
                class_ids, scores = assign_classes(outputs.logits[i], self._positive_map)
                xyxy = convert_boxes_cxcywh_norm_to_xyxy_abs(
                    outputs.pred_boxes[i], sample.width, sample.height
                )
                for q in range(len(class_ids)):
                    if scores[q] < box_threshold:
                        continue
                    detections.append(
                        Detection(
                            image_id=sample.image_id,
                            class_id=int(class_ids[q]),
                            score=float(scores[q]),
                            xyxy=tuple(float(v) for v in xyxy[q]),
                        )
                    )
        return detections
