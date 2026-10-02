"""The majority-answer baseline printed beside every Stage-2 accuracy (a dataset statistic, H9).

Pure logic over prediction rows — no scorer, no model."""

from __future__ import annotations

import pytest

from gdp.vqa.score import _majority_answer_baseline


def _row(answer: str, tag: list[int]) -> dict:
    return {"gt_answer": answer, "tag": tag}


def test_counts_only_tag_zero_items_the_accuracy_metric_scores():
    rows = [
        _row("No.", [0]),
        _row("No.", [0]),
        _row("Yes.", [0]),
        _row("a long free-text answer", [2]),  # language item: not an accuracy item
    ]
    result = _majority_answer_baseline(rows)
    assert result == {"n_items": 3, "answer": "No.", "rate": pytest.approx(2 / 3)}


def test_none_when_there_are_no_accuracy_items():
    assert _majority_answer_baseline([_row("anything", [1])]) is None
    assert _majority_answer_baseline([]) is None


def test_whitespace_around_answers_does_not_split_the_majority():
    result = _majority_answer_baseline([_row("No. ", [0]), _row("No.", [0])])
    assert result["rate"] == 1.0
