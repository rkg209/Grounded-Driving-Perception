"""Spec 08 tasks 1 + 4: the vendored DriveLM scorer's identity/weighting (no model or network
dependency), plus `to_official_format`/`run_official_scorer` proven against the vendored entry
point on the hand-built fixture predictions — "the spec's real risk", per the plan.

`run_official_scorer`'s round-trip test needs `language_evaluation` + `openai` installed (neither
is a hard `pyproject.toml` dependency — see `third_party/drivelm/PROVENANCE.md`); it skips itself
with a clear reason when they aren't, so the shape-only tests above it still run by default on a
fresh clone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gdp.vqa.official import (
    EVALUATION_PY,
    OfficialScorerUnavailable,
    scorer_composition,
    scorer_sha256,
    to_official_format,
)


def test_vendored_scorer_file_exists():
    assert EVALUATION_PY.is_file()


def test_scorer_sha256_matches_pinned_file():
    expected = hashlib.sha256(EVALUATION_PY.read_bytes()).hexdigest()
    assert scorer_sha256() == expected


def test_scorer_sha256_is_stable_across_calls():
    assert scorer_sha256() == scorer_sha256()


def test_scorer_composition_names_all_four_weighted_components():
    composition = scorer_composition()
    for component in ("chatgpt", "language", "match", "accuracy"):
        assert component in composition


def test_scorer_composition_names_the_pinned_commit():
    assert "266b570f1c746a4cad7f8f0317eb3c2f99141512" in scorer_composition()


def test_to_official_format_reshapes_rows():
    predictions = [
        {"qa_id": "a", "tag": [0], "gt_answer": "yes", "prediction": "yes"},
        {"qa_id": "b", "tag": [2], "gt_answer": "GT text", "prediction": "pred text"},
    ]
    assert to_official_format(predictions) == [
        {"qa_id": "a", "tag": [0], "answer": "yes", "GT": "yes"},
        {"qa_id": "b", "tag": [2], "answer": "pred text", "GT": "GT text"},
    ]


def test_to_official_format_raises_on_missing_tag():
    with pytest.raises(KeyError, match="tag"):
        to_official_format([{"qa_id": "a", "gt_answer": "x", "prediction": "y"}])


def test_to_official_format_raises_on_empty_tag_list():
    with pytest.raises(KeyError, match="tag"):
        to_official_format([{"qa_id": "a", "tag": [], "gt_answer": "x", "prediction": "y"}])


def _load_fixture_predictions(name: str) -> list[dict]:
    path = Path("tests/fixtures/vqa_predictions") / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_official_scorer_end_to_end_on_fixture_predictions():
    """The real risk: feed our reshaped predictions into the actual vendored `evaluation_suit`
    and check its raw output, not a value we invented ourselves."""
    try:
        from gdp.vqa.official import run_official_scorer
    except ImportError:
        pytest.skip("gdp.vqa.official failed to import")

    predictions = _load_fixture_predictions("base_mini.jsonl")
    items = to_official_format(predictions)
    try:
        result = run_official_scorer(items)
    except OfficialScorerUnavailable as exc:
        pytest.skip(f"vendored scorer deps not installed: {exc}")

    # fixture has 2 behavior (tag [0]) items, 2 perception (tag [2]), 2 planning (tag [1]),
    # 2 prediction (tag [3]) — see tests/fixtures/vqa_predictions/base_mini.jsonl.
    assert result["n_accuracy_items"] == 2
    assert result["n_language_items"] == 2
    assert result["n_chatgpt_items"] == 2
    assert result["n_match_items"] == 2
    assert result["accuracy"] is not None
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["language"] is not None
    assert "val/Bleu_1" in result["language"]


def test_run_official_scorer_returns_none_for_zero_item_buckets():
    """Design decision 7: a bucket with zero items is `None`, never `0.0` — proven against the
    real vendored `eval_acc`/`eval_language`, which would ZeroDivisionError on an empty list."""
    try:
        from gdp.vqa.official import run_official_scorer
    except ImportError:
        pytest.skip("gdp.vqa.official failed to import")

    items = to_official_format(
        [{"qa_id": "only-chatgpt", "tag": [1], "gt_answer": "GT", "prediction": "pred"}]
    )
    try:
        result = run_official_scorer(items)
    except OfficialScorerUnavailable as exc:
        pytest.skip(f"vendored scorer deps not installed: {exc}")

    assert result["accuracy"] is None
    assert result["language"] is None
    assert result["n_accuracy_items"] == 0
    assert result["n_language_items"] == 0
    assert result["n_chatgpt_items"] == 1
