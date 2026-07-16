"""Spec 02's real risk (specs/02-zeroshot-baseline.md design decision 1): the token-span trap.

Grounding-DINO returns per-query logits over *text tokens*, not class ids. A wrong span→class
mapping runs cleanly and reports a plausible-but-wrong mAP. Every test here targets one guard
against that: the round-trip assertion, the masked-mean class assignment (acceptance 3 — a box
whose span decodes to "traffic light" must be assigned class index 8, not 0), and the box-format
conversion boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gdp.config import BDD100K_CLASSES, DetectorConfig
from gdp.data.prompt import PromptSpans, get_tokenizer
from gdp.detect.detector import (
    GroundingDinoDetector,
    assert_span_round_trip,
    assign_classes,
    convert_boxes_cxcywh_norm_to_xyxy_abs,
)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return get_tokenizer()
    except Exception as exc:  # noqa: BLE001 — unavailable tokenizer is a skip, not a failure
        pytest.skip(f"bert-base-uncased tokenizer unavailable (no cache, no network): {exc}")


@pytest.fixture(scope="module")
def spans(tokenizer):
    return PromptSpans.from_classes(BDD100K_CLASSES, tokenizer=tokenizer)


def test_assert_span_round_trip_passes_for_matching_classes(spans, tokenizer):
    assert_span_round_trip(spans, BDD100K_CLASSES, tokenizer)


def test_assert_span_round_trip_raises_on_mismatch(tokenizer):
    """A processor/tokenizer whose spans don't decode to the expected class name must fail
    loudly, not silently mislabel every prediction downstream."""
    spans = PromptSpans.from_classes(("car", "bus"), tokenizer=tokenizer)
    with pytest.raises(RuntimeError, match="round-trip failed"):
        assert_span_round_trip(spans, ("bus", "car"), tokenizer)


def test_assign_classes_traffic_light_not_class_zero(spans):
    """Acceptance 3: a box whose logits peak on 'traffic light''s tokens is assigned class
    index 8, not 0 — the exact failure mode a naive decoded-phrase-string mapping risks."""
    max_text_len = 64
    pmap = spans.positive_map(max_text_len=max_text_len)
    logits = torch.full((1, max_text_len), -10.0)
    start, end = spans.token_spans[8]  # "traffic light"
    logits[0, start:end] = 10.0

    class_ids, scores = assign_classes(logits, pmap)

    assert class_ids[0] == 8
    assert scores[0] > 0.9


def test_assign_classes_is_batched_over_queries(spans):
    max_text_len = 64
    pmap = spans.positive_map(max_text_len=max_text_len)
    logits = torch.full((3, max_text_len), -10.0)
    expected = [0, 2, 8]
    for query_idx, class_idx in enumerate(expected):
        start, end = spans.token_spans[class_idx]
        logits[query_idx, start:end] = 10.0

    class_ids, _scores = assign_classes(logits, pmap)

    assert class_ids.tolist() == expected


def test_convert_boxes_cxcywh_norm_to_xyxy_abs():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])  # cx, cy, w, h — all normalized
    xyxy = convert_boxes_cxcywh_norm_to_xyxy_abs(boxes, width=100, height=200)
    np.testing.assert_allclose(xyxy, [[40.0, 60.0, 60.0, 140.0]], atol=1e-4)


def test_convert_boxes_rejects_values_outside_zero_one():
    """A 640.0 in a supposedly-normalized box means a caller passed absolute coordinates —
    catch it here, not as a silently-corrupt mAP three layers downstream."""
    boxes = torch.tensor([[0.5, 0.5, 0.2, 640.0]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        convert_boxes_cxcywh_norm_to_xyxy_abs(boxes, width=100, height=200)


@pytest.fixture(scope="module")
def detector():
    try:
        return GroundingDinoDetector(DetectorConfig(), BDD100K_CLASSES, device="cpu")
    except Exception as exc:  # noqa: BLE001 — unavailable weights is a skip, not a failure
        pytest.skip(f"grounding-dino-tiny unavailable (no cache, no network): {exc}")


def test_detector_prompt_spans_round_trip_against_processors_own_tokenizer(detector):
    """Design decision 1's critical alignment guard: PromptSpans must be built with the
    processor's own tokenizer, and every class must round-trip against it at construction."""
    for idx, name in enumerate(BDD100K_CLASSES):
        assert (
            detector.prompt_spans.decode_span(idx, tokenizer=detector.processor.tokenizer) == name
        )


def test_detector_prompt_matches_config_prompt_format(detector):
    assert detector.prompt == DetectorConfig().prompt(list(BDD100K_CLASSES))
