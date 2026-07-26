"""`heuristic_highlights` (spec 09 task 5): geometry only, no score/level/number (H9). No model
loaded — not `model_heavy`."""

from __future__ import annotations

import pytest

from gdp.demo.risk import heuristic_highlights
from gdp.detect.predictions import Detection

WIDTH, HEIGHT = 300, 200


def _det(x0, y0, x1, y1) -> Detection:
    return Detection(image_id=0, class_id=0, score=0.9, xyxy=(x0, y0, x1, y1))


def test_returns_detection_objects_only():
    center_bottom = _det(120, 150, 180, 195)  # centre-x=150 (mid third), bottom=195 (>0.55*200)
    out = heuristic_highlights([center_bottom], width=WIDTH, height=HEIGHT)
    assert out == [center_bottom]
    assert all(isinstance(d, Detection) for d in out)


def test_excludes_boxes_outside_middle_third():
    left_edge = _det(0, 150, 20, 195)  # centre-x=10, far left
    out = heuristic_highlights([left_edge], width=WIDTH, height=HEIGHT)
    assert out == []


def test_excludes_boxes_above_bottom_threshold():
    high_box = _det(120, 10, 180, 50)  # centre in middle third but bottom edge near the top
    out = heuristic_highlights([high_box], width=WIDTH, height=HEIGHT)
    assert out == []


def test_empty_input_returns_empty_list():
    assert heuristic_highlights([], width=WIDTH, height=HEIGHT) == []


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        heuristic_highlights([], width=0, height=HEIGHT)
    with pytest.raises(ValueError):
        heuristic_highlights([], width=WIDTH, height=-1)


def test_returned_detections_carry_only_the_detectors_own_score():
    det = _det(120, 150, 180, 195)
    out = heuristic_highlights([det], width=WIDTH, height=HEIGHT)
    assert out[0].score == det.score
    # Detection has exactly these fields — nothing risk-shaped has been attached.
    assert set(vars(out[0])) <= {"image_id", "class_id", "score", "xyxy"}
