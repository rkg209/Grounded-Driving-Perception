"""Hallucination rate — a self-defined diagnostic (H9), never dressed up as an official metric.

Acceptance criterion 5 wants a number; H9 forbids scoring our own rule against our own labels as
if it were DriveLM's metric. `CAVEAT` travels beside every number this module produces, exactly as
`gdp.ground.evaluate.CAVEAT` does for the grounding-accuracy set — it never appears in the same
table as `gdp.vqa.score`'s official sub-metrics.
"""

from __future__ import annotations

from typing import Any

from gdp.config import DRIVELM_CATEGORIES
from gdp.data.drivelm import parse_object_tags

CAVEAT = "self-defined diagnostic, not part of DriveLM's official metric"


def ungrounded_tags(prediction: str, gt_answer: str) -> list[dict[str, Any]]:
    """Object-reference tags (`<c1,CAM_FRONT,x,y>`) the model emitted in `prediction` whose `ref`
    (e.g. `"c1"`) has no referent among the ones DriveLM's own ground-truth answer names — an
    ungrounded object reference. Only `ref` is compared (not camera/coordinates): the model is
    hallucinating an *object*, not necessarily mis-locating a real one, and a coordinate-exact
    match would miss a real object cited with slightly different numbers."""
    valid_refs = {tag["ref"] for tag in parse_object_tags(gt_answer)}
    return [tag for tag in parse_object_tags(prediction) if tag["ref"] not in valid_refs]


def _stats_for_subset(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total_emitted = 0
    total_ungrounded = 0
    items_with_hallucination = 0

    for row in predictions:
        tags = ungrounded_tags(row["prediction"], row["gt_answer"])
        emitted = parse_object_tags(row["prediction"])
        total_emitted += len(emitted)
        total_ungrounded += len(tags)
        if tags:
            items_with_hallucination += 1

    return {
        "n_items": len(predictions),
        "n_emitted_tags": total_emitted,
        "n_ungrounded_tags": total_ungrounded,
        "ungrounded_tag_rate": (total_ungrounded / total_emitted) if total_emitted else None,
        "items_with_hallucination": items_with_hallucination,
        "items_with_hallucination_rate": (
            items_with_hallucination / len(predictions) if predictions else None
        ),
    }


def hallucination_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall + per-category ungrounded-tag rates (design decision 9). `None`, never `0.0`, when
    a subset emits no object-reference tags at all — mirroring `gdp.vqa.score`'s zero-item
    handling, since a rate over zero denominator is not a rate of zero."""
    return {
        "caveat": CAVEAT,
        "overall": _stats_for_subset(predictions),
        "per_category": {
            category: _stats_for_subset([p for p in predictions if p["category"] == category])
            for category in DRIVELM_CATEGORIES
        },
    }
