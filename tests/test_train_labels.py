from __future__ import annotations

import pytest
import torch

from gdp.config import BDD100K_CLASSES
from gdp.data.core import Sample, load_dataset
from gdp.data.prompt import PromptSpans, get_tokenizer
from gdp.train.dataset import DetectionCollator
from gdp.train.labels import assert_label_map_alignment, assert_normalized_cxcywh, build_coco_target

FIXTURE_ANNOTATIONS = "tests/fixtures/mini_bdd/annotations.json"
FIXTURE_ROOT = "tests/fixtures/mini_bdd"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


@pytest.fixture(scope="module")
def prompt_spans(tokenizer):
    return PromptSpans.from_classes(BDD100K_CLASSES, tokenizer=tokenizer)


def test_label_map_alignment_holds_for_all_ten_classes(prompt_spans, tokenizer):
    """The spec's real risk: HF's own delimiter-splitting must agree with PromptSpans, for every
    class, not just the ones without multi-word names."""
    assert_label_map_alignment(prompt_spans, tokenizer)


def test_label_map_alignment_catches_a_shuffled_span(prompt_spans, tokenizer):
    """A deliberately wrong token span must be caught, not silently accepted."""
    bad_spans = PromptSpans(
        prompt=prompt_spans.prompt,
        classes=prompt_spans.classes,
        input_ids=prompt_spans.input_ids,
        char_spans=prompt_spans.char_spans,
        token_spans=(prompt_spans.token_spans[1], prompt_spans.token_spans[0])
        + prompt_spans.token_spans[2:],
    )
    with pytest.raises(RuntimeError, match="HF build_label_maps assigned tokens"):
        assert_label_map_alignment(bad_spans, tokenizer)


def test_build_coco_target_uses_class_id_as_category_id():
    ds = load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_ROOT)
    sample = ds.samples[0]
    target = build_coco_target(sample)
    assert target["image_id"] == sample.image_id
    assert [a["category_id"] for a in target["annotations"]] == [b.class_id for b in sample.boxes]
    assert all(a["iscrowd"] == 0 for a in target["annotations"])


def test_assert_normalized_cxcywh_accepts_valid_boxes():
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.1, 0.1, 0.05, 0.05]])
    assert_normalized_cxcywh(boxes)  # must not raise


def test_assert_normalized_cxcywh_rejects_absolute_xyxy():
    """An absolute-pixel box (e.g. 640.0) must raise, not train silently on garbage."""
    boxes = torch.tensor([[320.0, 180.0, 640.0, 360.0]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assert_normalized_cxcywh(boxes)


def test_assert_normalized_cxcywh_rejects_degenerate_box():
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.2]])
    with pytest.raises(ValueError, match="degenerate box"):
        assert_normalized_cxcywh(boxes)


def test_collator_rejects_mixed_size_batch():
    ds = load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_ROOT)
    s0 = ds.samples[0]
    mismatched = Sample(
        image_id=s0.image_id,
        image_path=s0.image_path,
        width=s0.width + 1,
        height=s0.height,
        boxes=s0.boxes,
    )
    collator = DetectionCollator(
        processor=None, prompt="pedestrian."
    )  # never reaches the processor
    with pytest.raises(ValueError, match="requires every image in a batch to share a size"):
        collator([s0, mismatched])


def test_collator_produces_asserted_normalized_boxes():
    from transformers import AutoProcessor

    ds = load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_ROOT)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    prompt = ds.prompt()
    collator = DetectionCollator(processor=processor, prompt=prompt)

    batch = collator(list(ds.samples[:2]))
    assert batch["pixel_values"].shape[0] == 2
    assert len(batch["labels"]) == 2
    for i, sample in enumerate(ds.samples[:2]):
        expected_class_ids = [b.class_id for b in sample.boxes]
        assert batch["labels"][i]["class_labels"].tolist() == expected_class_ids
        boxes = batch["labels"][i]["boxes"]
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0
