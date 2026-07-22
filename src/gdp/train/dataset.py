"""`Sample`s -> batched Grounding-DINO training inputs.

`GroundingDinoProcessor.__call__` forwards only `images` + `text` (verified against the installed
`transformers==5.13.1`: an `annotations=` kwarg is silently dropped with a warning, not an error —
exactly the kind of thing that would train a model with unsupervised boxes and no signal it
happened). The collator therefore drives the image processor and tokenizer separately, as spec 03
design decision 2 requires, and asserts the result before it ever reaches the loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import GroundingDinoProcessor

from gdp.data.core import Sample
from gdp.train.labels import assert_normalized_cxcywh, build_coco_target


class SampleDataset(TorchDataset):
    """Thin `torch.utils.data.Dataset` wrapper over a tuple of `gdp.data.core.Sample`."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class DetectionCollator:
    """Batches `Sample`s into Grounding-DINO training inputs: one fixed prompt, asserted boxes.

    Every image in a batch must share a size. `GroundingDinoImageProcessorPil` normalizes each
    annotation's boxes against that image's own *resized* size (a DETR-family convention); if two
    images in the same batch started at different original sizes, nothing in the processor would
    warn — the boxes would just be normalized against the wrong denominator. BDD100K is uniformly
    1280x720 and the fixture 640x360, so asserting same-size-per-batch (rather than handling mixed
    sizes correctly) is a deliberate scope cut, not an oversight.
    """

    def __init__(self, processor: GroundingDinoProcessor, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, samples: Sequence[Sample]) -> dict[str, Any]:
        sizes = {(s.width, s.height) for s in samples}
        if len(sizes) > 1:
            raise ValueError(
                f"DetectionCollator requires every image in a batch to share a size, got {sizes}"
            )

        images = [Image.open(s.image_path).convert("RGB") for s in samples]
        targets = [build_coco_target(s) for s in samples]

        image_inputs = self.processor.image_processor(
            images=images, annotations=targets, return_tensors="pt"
        )
        text_inputs = self.processor.tokenizer(
            [self.prompt] * len(samples), return_tensors="pt", padding=True
        )

        for label in image_inputs["labels"]:
            assert_normalized_cxcywh(label["boxes"])

        return {
            "pixel_values": image_inputs["pixel_values"],
            "pixel_mask": image_inputs["pixel_mask"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "token_type_ids": text_inputs["token_type_ids"],
            "labels": image_inputs["labels"],
        }
