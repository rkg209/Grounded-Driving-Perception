"""`draw_boxes` (spec 09 task 5). No model loaded — not `model_heavy`."""

from __future__ import annotations

from PIL import Image

from gdp.demo.overlay import draw_boxes
from gdp.detect.predictions import Detection

CLASSES = ("pedestrian", "car")


def _det(class_id: int = 1) -> Detection:
    return Detection(image_id=0, class_id=class_id, score=0.8, xyxy=(10.0, 10.0, 50.0, 50.0))


def test_draw_boxes_returns_a_new_image_same_size():
    img = Image.new("RGB", (100, 80), color="white")
    out = draw_boxes(img, [_det()], CLASSES)
    assert out is not img
    assert out.size == img.size


def test_draw_boxes_does_not_mutate_input():
    img = Image.new("RGB", (100, 80), color="white")
    original_bytes = img.tobytes()
    draw_boxes(img, [_det()], CLASSES)
    assert img.tobytes() == original_bytes


def test_draw_boxes_handles_empty_detections():
    img = Image.new("RGB", (100, 80), color="white")
    out = draw_boxes(img, [], CLASSES)
    assert out.size == img.size


def test_draw_boxes_accepts_highlight_indices():
    img = Image.new("RGB", (100, 80), color="white")
    out = draw_boxes(img, [_det(), _det(0)], CLASSES, highlight={0})
    assert out.size == img.size


def test_draw_boxes_unknown_class_id_does_not_raise():
    img = Image.new("RGB", (100, 80), color="white")
    det = Detection(image_id=0, class_id=99, score=0.5, xyxy=(0.0, 0.0, 5.0, 5.0))
    out = draw_boxes(img, [det], CLASSES)
    assert out.size == img.size
