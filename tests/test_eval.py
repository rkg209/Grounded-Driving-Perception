"""Spec 02's evaluation math: the pycocotools mAP wrapper and the greedy operating-point matcher.

`detection-eval` skill: never hand-roll AP interpolation (H9) — `coco_map.py` only adapts
pycocotools' I/O. The operating-point matcher is home-grown but simple enough to hand-verify on
tiny synthetic boxes, which is exactly what these tests do.
"""

from __future__ import annotations

import json

import pytest

from gdp.eval.coco_map import evaluate_coco_map
from gdp.eval.operating_point import (
    GtBox,
    PredBox,
    box_iou,
    match_operating_point,
    sweep_threshold,
)


def test_box_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_disjoint_boxes_is_zero():
    assert box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_box_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 5x10=50, union 100+100-50=150
    assert box_iou(a, b) == pytest.approx(50.0 / 150.0)


def test_match_operating_point_perfect_prediction_is_all_tp():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredBox(image_id=0, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0))]
    per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 1 and overall.fp == 0 and overall.fn == 0
    assert overall.precision == 1.0 and overall.recall == 1.0
    assert per_class[2].tp == 1


def test_match_operating_point_extra_prediction_is_fp():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [
        PredBox(image_id=0, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0)),
        PredBox(image_id=0, class_id=2, score=0.5, xyxy=(50.0, 50.0, 60.0, 60.0)),
    ]
    _per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 1 and overall.fp == 1 and overall.fn == 0


def test_match_operating_point_missed_gt_is_fn():
    gt = [
        GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0)),
        GtBox(image_id=0, class_id=2, xyxy=(50.0, 50.0, 60.0, 60.0)),
    ]
    pred = [PredBox(image_id=0, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0))]
    _per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 1 and overall.fp == 0 and overall.fn == 1
    assert overall.recall == pytest.approx(0.5)


def test_match_operating_point_low_iou_is_fp_and_fn():
    """A prediction that only barely touches the GT box (IoU below threshold) counts as a false
    positive, and the GT box it failed to match counts as a false negative — not a match."""
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredBox(image_id=0, class_id=2, score=0.9, xyxy=(9.0, 9.0, 19.0, 19.0))]
    _per_class, overall = match_operating_point(pred, gt, iou_threshold=0.5)
    assert overall.tp == 0 and overall.fp == 1 and overall.fn == 1


def test_match_operating_point_greedy_prefers_higher_score_first():
    """Two predictions both overlap one GT box; only the higher-scoring one should win the
    match, leaving the lower-scoring one a false positive."""
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [
        PredBox(image_id=0, class_id=2, score=0.4, xyxy=(0.0, 0.0, 10.0, 10.0)),
        PredBox(image_id=0, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0)),
    ]
    _per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 1 and overall.fp == 1


def test_match_operating_point_different_images_do_not_match():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredBox(image_id=1, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0))]
    _per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 0 and overall.fp == 1 and overall.fn == 1


def test_match_operating_point_different_classes_do_not_match():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredBox(image_id=0, class_id=3, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0))]
    _per_class, overall = match_operating_point(pred, gt)
    assert overall.tp == 0 and overall.fp == 1 and overall.fn == 1


def test_sweep_threshold_picks_the_threshold_with_best_f1():
    gt = [
        GtBox(image_id=0, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0)),
        GtBox(image_id=1, class_id=2, xyxy=(0.0, 0.0, 10.0, 10.0)),
    ]
    pred = [
        PredBox(image_id=0, class_id=2, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0)),  # good, high score
        PredBox(image_id=1, class_id=2, score=0.6, xyxy=(0.0, 0.0, 10.0, 10.0)),  # good, mid score
        PredBox(
            image_id=0, class_id=2, score=0.3, xyxy=(50.0, 50.0, 60.0, 60.0)
        ),  # junk, low score
    ]
    # threshold 0.5 keeps both good predictions and drops the junk one -> perfect P/R.
    # threshold 0.1 keeps the junk prediction too -> a false positive.
    # threshold 0.95 drops everything -> zero recall.
    threshold, pr = sweep_threshold(pred, gt, candidates=[0.1, 0.5, 0.95])
    assert threshold == 0.5
    assert pr.precision == 1.0 and pr.recall == 1.0


def test_sweep_threshold_rejects_empty_candidates():
    with pytest.raises(ValueError, match="candidates"):
        sweep_threshold([], [], candidates=[])


@pytest.fixture
def tiny_coco_gt(tmp_path):
    gt = {
        "images": [
            {"id": 0, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [30, 30, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"},
        ],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    return path


def test_evaluate_coco_map_perfect_predictions_score_near_one(tiny_coco_gt):
    """Predictions that exactly match ground truth should score AP (and AP@50) close to 1.0 —
    not a hand-rolled computation, pycocotools' own.

    Also the regression test for the annotation-id-0 sentinel bug: `tiny_coco_gt`'s first
    annotation has `id: 0`, matching `gdp.data.bdd100k.convert_bdd_to_coco`'s 0-based ids
    (and every real converted split). Without `_load_coco_gt_with_safe_ids`'s +1 shift, this
    exact fixture reproduced mAP=0.5 instead of 1.0 — the "car" category's perfectly-matched
    box was silently counted as a false positive because pycocotools' `dtMatches` array can't
    distinguish "matched to gt id 0" from "unmatched"."""
    predictions = [
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 1, "category_id": 2, "bbox": [30, 30, 10, 10], "score": 0.99},
    ]
    result = evaluate_coco_map(tiny_coco_gt, predictions)
    assert result.map == pytest.approx(1.0, abs=1e-6)
    assert result.map50 == pytest.approx(1.0, abs=1e-6)
    assert result.per_class_ap["car"] == pytest.approx(1.0, abs=1e-6)
    assert result.per_class_ap["pedestrian"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_coco_map_no_predictions_is_all_zero_not_a_crash(tiny_coco_gt):
    result = evaluate_coco_map(tiny_coco_gt, [])
    assert result.map == 0.0
    assert result.map50 == 0.0
    assert result.per_class_ap == {"car": 0.0, "pedestrian": 0.0}


def test_evaluate_coco_map_accepts_a_predictions_file_path(tiny_coco_gt, tmp_path):
    predictions = [{"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99}]
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions))
    result = evaluate_coco_map(tiny_coco_gt, pred_path)
    assert result.per_class_ap["car"] == pytest.approx(1.0, abs=1e-6)
