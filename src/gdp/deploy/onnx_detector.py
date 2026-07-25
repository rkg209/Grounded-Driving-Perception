"""ONNX Runtime inference behind the exact `detect_images` contract `GroundingDinoDetector` uses.

Same signature, same return type (`list[Detection]`), and — the point of this module — the same
`assign_classes`/`convert_boxes_cxcywh_norm_to_xyxy_abs` box-decoding path (`gdp.detect.detector`).
A second, ONNX-specific box-decoding implementation would be exactly the kind of drift the
detection-eval skill and H9 warn about: "mAP" has to mean the same computation for the PyTorch and
ONNX variants, or the accuracy-vs-latency curve (spec 05's deliverable) compares apples to oranges.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoProcessor

from gdp.config import DetectorConfig
from gdp.data.core import Sample
from gdp.data.prompt import PromptSpans
from gdp.deploy.export import INPUT_NAMES, OUTPUT_NAMES
from gdp.detect.detector import (
    assert_span_round_trip,
    assign_classes,
    convert_boxes_cxcywh_norm_to_xyxy_abs,
)
from gdp.detect.predictions import Detection


def _chunks(items: Sequence[Sample], size: int) -> Iterator[Sequence[Sample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class OnnxDetector:
    """Loads an exported `model.onnx` for inference, frozen to `config.prompt(classes)`.

    The processor/tokenizer are loaded fresh from `config.model_id` — a lightweight download, not
    the full model weights — and round-trip-asserted exactly as `GroundingDinoDetector` does,
    since the exported graph's `input_ids` only make sense under the same tokenization
    `PromptSpans` was built with.
    """

    def __init__(
        self, onnx_path: str | Path, config: DetectorConfig, classes: Sequence[str]
    ) -> None:
        self.config = config
        self.classes = tuple(classes)
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        self.prompt = config.prompt(list(self.classes))
        self.prompt_spans = PromptSpans.from_classes(
            self.classes, tokenizer=self.processor.tokenizer
        )
        assert_span_round_trip(self.prompt_spans, self.classes, self.processor.tokenizer)
        self._positive_map = self.prompt_spans.positive_map()

    def detect_images(
        self, samples: Sequence[Sample], *, box_threshold: float, batch_size: int = 8
    ) -> list[Detection]:
        """Batched inference, matching `GroundingDinoDetector.detect_images` exactly."""
        detections: list[Detection] = []
        for chunk in _chunks(samples, batch_size):
            images = [Image.open(s.image_path).convert("RGB") for s in chunk]
            inputs = self.processor(
                images=images, text=[self.prompt] * len(images), return_tensors="np"
            )
            ort_inputs = {name: inputs[name] for name in INPUT_NAMES}
            logits, pred_boxes = self.session.run(list(OUTPUT_NAMES), ort_inputs)

            for i, sample in enumerate(chunk):
                class_ids, scores = assign_classes(torch.from_numpy(logits[i]), self._positive_map)
                xyxy = convert_boxes_cxcywh_norm_to_xyxy_abs(
                    torch.from_numpy(pred_boxes[i]), sample.width, sample.height
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
