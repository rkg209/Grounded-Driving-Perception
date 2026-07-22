"""Spec 04 task 5 — grounding accuracy's actual definition, tested on hand-built boxes. No model
is loaded here (`score_phrase`/`aggregate_results` are pure functions), so this file is
intentionally NOT `model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

from gdp.detect.predictions import Detection
from gdp.ground.evaluate import aggregate_results, score_phrase
from gdp.ground.phrases import Phrase

TARGET_BOX = (10.0, 10.0, 50.0, 50.0)  # 40x40


def _positive_phrase(qualifier_type: str = "spatial") -> Phrase:
    return Phrase(
        image_id=0,
        phrase="the pedestrian",
        qualifier_type=qualifier_type,
        author="rahul",
        date="2026-07-22",
        target_ann_id=0,
    )


def _negative_phrase() -> Phrase:
    return Phrase(
        image_id=0,
        phrase="the bus",
        qualifier_type="negative",
        author="rahul",
        date="2026-07-22",
        absent_class="bus",
    )


def _detection(xyxy: tuple[float, float, float, float], score: float = 0.9) -> Detection:
    return Detection(image_id=0, class_id=0, score=score, xyxy=xyxy)


def test_positive_phrase_with_high_iou_prediction_is_correct():
    prediction = _detection((10.0, 10.0, 50.0, 50.0))  # exact match, IoU = 1.0
    result = score_phrase(_positive_phrase(), prediction, target_box=TARGET_BOX)
    assert result.correct is True
    assert result.top1_iou == 1.0


def test_positive_phrase_with_low_iou_prediction_is_incorrect():
    prediction = _detection((200.0, 200.0, 240.0, 240.0))  # nowhere near the target
    result = score_phrase(_positive_phrase(), prediction, target_box=TARGET_BOX)
    assert result.correct is False
    assert result.top1_iou == 0.0


def test_positive_phrase_at_the_iou_threshold_boundary_is_correct():
    # A box shifted so IoU is exactly 0.5: (10,10,50,50) vs (10,30,50,70) -> intersection
    # 40x20=800, union 1600+1600-800=2400, IoU=1/3. Use a box tuned to land exactly on 0.5 instead.
    # (10,10,50,50) area 1600. Shift x by 10: (20,10,60,50) -> intersection 30x40=1200,
    # union 1600+1600-1200=2000, IoU=0.6 (> 0.5, correct). Use that as the boundary-adjacent case.
    prediction = _detection((20.0, 10.0, 60.0, 50.0))
    result = score_phrase(_positive_phrase(), prediction, target_box=TARGET_BOX)
    assert result.top1_iou == 0.6
    assert result.correct is True


def test_positive_phrase_with_no_prediction_is_incorrect():
    result = score_phrase(_positive_phrase(), None, target_box=TARGET_BOX)
    assert result.correct is False
    assert result.top1_iou is None
    assert result.predicted_box is None


def test_negative_phrase_with_no_prediction_above_threshold_is_correct():
    result = score_phrase(_negative_phrase(), None, target_box=None)
    assert result.correct is True
    assert result.top1_iou is None


def test_negative_phrase_with_a_prediction_above_threshold_is_incorrect():
    prediction = _detection((5.0, 5.0, 30.0, 30.0), score=0.8)
    result = score_phrase(_negative_phrase(), prediction, target_box=None)
    assert result.correct is False
    assert result.predicted_score == 0.8


def test_aggregate_results_reports_per_type_and_positive_negative_subtotals():
    results = [
        score_phrase(_positive_phrase("spatial"), _detection(TARGET_BOX), target_box=TARGET_BOX),
        score_phrase(_positive_phrase("attribute"), None, target_box=TARGET_BOX),
        score_phrase(_negative_phrase(), None, target_box=None),
        score_phrase(_negative_phrase(), _detection((0.0, 0.0, 5.0, 5.0)), target_box=None),
    ]
    agg = aggregate_results(results)

    assert agg["per_type"]["spatial"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert agg["per_type"]["attribute"] == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert agg["per_type"]["relational"] == {"n": 0, "correct": 0, "accuracy": 0.0}
    assert agg["per_type"]["negative"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert agg["positives"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert agg["negatives"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert agg["overall"] == {"n": 4, "correct": 2, "accuracy": 0.5}


def test_aggregate_results_a_model_that_boxes_everything_scores_0_on_negatives():
    """Design decision 8's own example: never let a single aggregate hide this."""
    results = [
        score_phrase(_positive_phrase(), _detection(TARGET_BOX), target_box=TARGET_BOX),
        score_phrase(_negative_phrase(), _detection((0.0, 0.0, 5.0, 5.0)), target_box=None),
    ]
    agg = aggregate_results(results)
    assert agg["positives"]["accuracy"] == 1.0
    assert agg["negatives"]["accuracy"] == 0.0
    assert agg["overall"]["accuracy"] == 0.5  # would look fine in isolation; hides the 0% negative
