"""Spec 04 task 7 — `build_grounding_comparison`'s H8 guard and per-type delta arithmetic. Pure
dict-in, dict-out; no model, no dataset, not `model_heavy`."""

from __future__ import annotations

import math

import pytest

from gdp.ground.compare import build_grounding_comparison

BASE_FIELDS = {
    "phrases_sha256": "abc123",
    "num_phrases": 8,
    "box_threshold": 0.25,
    "model_id": "IDEA-Research/grounding-dino-tiny",
    "created": "2026-01-01T00:00:00+00:00",
}


def _metrics(**overrides) -> dict:
    base = dict(BASE_FIELDS)
    base.update(
        {
            "per_type": {
                "spatial": {"n": 2, "correct": 0, "accuracy": 0.0},
                "attribute": {"n": 2, "correct": 0, "accuracy": 0.0},
                "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
                "negative": {"n": 2, "correct": 0, "accuracy": 0.0},
            },
            "positives": {"n": 6, "correct": 0, "accuracy": 0.0},
            "negatives": {"n": 2, "correct": 0, "accuracy": 0.0},
            "overall": {"n": 8, "correct": 0, "accuracy": 0.0},
        }
    )
    base.update(overrides)
    return base


def test_computes_overall_accuracy_delta():
    zeroshot = _metrics(overall={"n": 8, "correct": 0, "accuracy": 0.0})
    finetuned = _metrics(
        overall={"n": 8, "correct": 3, "accuracy": 0.375},
        model_id="runs/03-finetune/20260101/checkpoint-500",
    )

    result = build_grounding_comparison(zeroshot, finetuned)

    assert result["accuracy_zeroshot"] == 0.0
    assert result["accuracy_finetuned"] == pytest.approx(0.375)
    assert result["accuracy_delta"] == pytest.approx(0.375)
    assert result["zeroshot_model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert result["finetuned_model_id"] == "runs/03-finetune/20260101/checkpoint-500"


def test_per_type_deltas_and_regressions():
    zeroshot = _metrics(
        per_type={
            "spatial": {"n": 2, "correct": 1, "accuracy": 0.5},
            "attribute": {"n": 2, "correct": 0, "accuracy": 0.0},
            "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
            "negative": {"n": 2, "correct": 2, "accuracy": 1.0},
        }
    )
    finetuned = _metrics(
        per_type={
            "spatial": {"n": 2, "correct": 2, "accuracy": 1.0},
            "attribute": {"n": 2, "correct": 1, "accuracy": 0.5},
            "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
            "negative": {"n": 2, "correct": 1, "accuracy": 0.5},  # regressed
        }
    )

    result = build_grounding_comparison(zeroshot, finetuned)

    assert result["per_type"]["spatial"]["delta"] == pytest.approx(0.5)
    assert result["per_type"]["attribute"]["delta"] == pytest.approx(0.5)
    assert result["per_type"]["relational"]["delta"] == pytest.approx(0.0)
    assert result["per_type"]["negative"]["delta"] == pytest.approx(-0.5)
    assert result["regressions"] == ["negative"]


def test_nan_accuracy_type_is_not_a_regression_and_does_not_crash():
    zeroshot = _metrics(
        per_type={
            "spatial": {"n": 0, "correct": 0, "accuracy": float("nan")},
            "attribute": {"n": 2, "correct": 0, "accuracy": 0.0},
            "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
            "negative": {"n": 2, "correct": 0, "accuracy": 0.0},
        }
    )
    finetuned = _metrics(
        per_type={
            "spatial": {"n": 0, "correct": 0, "accuracy": float("nan")},
            "attribute": {"n": 2, "correct": 1, "accuracy": 0.5},
            "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
            "negative": {"n": 2, "correct": 0, "accuracy": 0.0},
        }
    )

    result = build_grounding_comparison(zeroshot, finetuned)

    assert math.isnan(result["per_type"]["spatial"]["delta"])
    assert "spatial" not in result["regressions"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("phrases_sha256", "different"),
        ("num_phrases", 999),
        ("box_threshold", 0.3),
    ],
)
def test_refuses_mismatched_runs(field, value):
    zeroshot = _metrics()
    finetuned = _metrics(**{field: value})

    with pytest.raises(ValueError, match="refusing to compare mismatched grounding runs"):
        build_grounding_comparison(zeroshot, finetuned)


def test_stamps_h9_labels_on_the_output():
    result = build_grounding_comparison(_metrics(), _metrics())
    assert result["is_self_built_benchmark"] is True
    assert "caveat" in result and result["caveat"]
