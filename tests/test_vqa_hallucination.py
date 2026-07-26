"""Spec 08 task 6: `gdp.vqa.hallucination` — the self-defined ungrounded-object-tag diagnostic.

No model dependency; exercises `tests/fixtures/vqa_predictions/base_mini.jsonl`'s deliberately
planted ungrounded tag (`<c9,CAM_BACK,...>` on the perception item, task 3) and its valid one
(`<c1,CAM_FRONT,...>`, present in both prediction and GT).
"""

from __future__ import annotations

import json
from pathlib import Path

from gdp.vqa.hallucination import CAVEAT, hallucination_stats, ungrounded_tags


def _load(name: str) -> list[dict]:
    path = Path("tests/fixtures/vqa_predictions") / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ungrounded_tags_catches_the_planted_hallucination():
    prediction = (
        "There is a pedestrian <c1,CAM_FRONT,110.0,50.0> and a cyclist "
        "<c9,CAM_BACK,900.0,200.0> near the crosswalk."
    )
    gt_answer = "There is a pedestrian <c1,CAM_FRONT,110.0,50.0> near the crosswalk."
    tags = ungrounded_tags(prediction, gt_answer)
    assert len(tags) == 1
    assert tags[0]["ref"] == "c9"


def test_ungrounded_tags_empty_when_all_refs_match_gt():
    prediction = "There is a pedestrian <c1,CAM_FRONT,110.0,50.0> near the crosswalk."
    gt_answer = "There is a pedestrian <c1,CAM_FRONT,110.0,50.0> near the crosswalk."
    assert ungrounded_tags(prediction, gt_answer) == []


def test_ungrounded_tags_empty_when_prediction_has_no_tags():
    assert ungrounded_tags("Decelerating.", "Decelerating.") == []


def test_hallucination_stats_caveat_present():
    stats = hallucination_stats(_load("base_mini.jsonl"))
    assert stats["caveat"] == CAVEAT


def test_hallucination_stats_finds_the_planted_tag_in_base_predictions():
    base = _load("base_mini.jsonl")
    stats = hallucination_stats(base)
    overall = stats["overall"]
    assert overall["n_ungrounded_tags"] == 1
    assert overall["items_with_hallucination"] == 1
    assert 0.0 < overall["ungrounded_tag_rate"] < 1.0

    perception = stats["per_category"]["perception"]
    assert perception["n_ungrounded_tags"] == 1
    assert perception["items_with_hallucination"] == 1


def test_hallucination_stats_clean_on_finetuned_predictions():
    """The finetuned fixture has no planted ungrounded tag — the diagnostic must report zero,
    not just skip silently."""
    finetuned = _load("finetuned_mini.jsonl")
    stats = hallucination_stats(finetuned)
    assert stats["overall"]["n_ungrounded_tags"] == 0
    assert stats["overall"]["items_with_hallucination"] == 0


def test_hallucination_stats_none_not_zero_for_categories_with_no_emitted_tags():
    """behavior/planning/prediction emit no object-reference tags at all in the fixture — a rate
    over zero denominator must be None, never 0.0."""
    base = _load("base_mini.jsonl")
    stats = hallucination_stats(base)
    behavior = stats["per_category"]["behavior"]
    assert behavior["n_emitted_tags"] == 0
    assert behavior["ungrounded_tag_rate"] is None
