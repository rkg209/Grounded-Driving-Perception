from __future__ import annotations

import math

import pytest

from gdp.eval.compare import build_comparison

BASE_FIELDS = {
    "dataset": "bdd100k",
    "split": "val",
    "num_images": 1000,
    "box_threshold": 0.25,
    "map_score_floor": 0.05,
    "model_id": "IDEA-Research/grounding-dino-tiny",
    "created": "2026-01-01T00:00:00+00:00",
}


def _metrics(**overrides) -> dict:
    base = dict(BASE_FIELDS)
    base.update(
        {
            "map": 0.20,
            "map50": 0.35,
            "per_class_ap": {"car": 0.30, "pedestrian": 0.10, "bus": 0.05},
        }
    )
    base.update(overrides)
    return base


def test_computes_map_and_map50_delta():
    zeroshot = _metrics(map=0.20, map50=0.35)
    finetuned = _metrics(map=0.28, map50=0.42, model_id="runs/03-finetune/20260101/checkpoint-500")

    result = build_comparison(zeroshot, finetuned)

    assert result["map_zeroshot"] == pytest.approx(0.20)
    assert result["map_finetuned"] == pytest.approx(0.28)
    assert result["map_delta"] == pytest.approx(0.08)
    assert result["map50_delta"] == pytest.approx(0.07)
    assert result["zeroshot_model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert result["finetuned_model_id"] == "runs/03-finetune/20260101/checkpoint-500"


def test_per_class_deltas_and_regressions():
    zeroshot = _metrics(per_class_ap={"car": 0.30, "pedestrian": 0.10, "bus": 0.05})
    finetuned = _metrics(per_class_ap={"car": 0.40, "pedestrian": 0.05, "bus": 0.05})

    result = build_comparison(zeroshot, finetuned)

    assert result["per_class_ap"]["car"] == {
        "before": 0.30,
        "after": 0.40,
        "delta": pytest.approx(0.10),
    }
    assert result["per_class_ap"]["pedestrian"]["delta"] == pytest.approx(-0.05)
    assert result["per_class_ap"]["bus"]["delta"] == pytest.approx(0.0)
    assert result["regressions"] == ["pedestrian"]


def test_nan_ap_class_is_not_a_regression_and_does_not_crash():
    """A class with no ground-truth boxes in this split scores NaN (pycocotools' convention) —
    a NaN delta must never be silently coerced into a false regression or a crash."""
    zeroshot = _metrics(per_class_ap={"car": 0.30, "train": float("nan")})
    finetuned = _metrics(per_class_ap={"car": 0.35, "train": float("nan")})

    result = build_comparison(zeroshot, finetuned)

    assert math.isnan(result["per_class_ap"]["train"]["delta"])
    assert "train" not in result["regressions"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("dataset", "mini_bdd"),
        ("split", "train"),
        ("num_images", 999),
        ("box_threshold", 0.3),
        # A higher floor truncates the PR curve: the mAPs would differ with no fine-tuning at all.
        ("map_score_floor", 0.25),
    ],
)
def test_refuses_mismatched_runs(field, value):
    zeroshot = _metrics()
    finetuned = _metrics(**{field: value})

    with pytest.raises(ValueError, match="refusing to compare mismatched runs"):
        build_comparison(zeroshot, finetuned)


def test_a_class_only_in_one_run_is_handled_with_nan_on_the_other_side():
    zeroshot = _metrics(per_class_ap={"car": 0.30})
    finetuned = _metrics(per_class_ap={"car": 0.35, "bicycle": 0.10})

    result = build_comparison(zeroshot, finetuned)

    assert math.isnan(result["per_class_ap"]["bicycle"]["before"])
    assert result["per_class_ap"]["bicycle"]["after"] == 0.10
