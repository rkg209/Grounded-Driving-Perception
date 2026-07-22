"""Spec 04 task 2 — the phrase schema is this spec's real risk: an orphan target, a false
negative, or a broken round-trip must all fail loudly, never silently score wrong."""

from __future__ import annotations

import json
import pathlib

import pytest

from gdp.ground.phrases import (
    Phrase,
    PhraseSet,
    assert_frozen,
    ensure_valid,
    freeze,
    load_phrases,
    phrases_sha256,
    validate_phrases,
)

FIXTURE_PHRASES = "tests/fixtures/grounding/phrases_mini.json"
FIXTURE_ANNOTATIONS = "tests/fixtures/mini_bdd/annotations.json"


def _base_phrase(**overrides) -> Phrase:
    base = dict(
        image_id=0,
        phrase="the pedestrian near the curb",
        qualifier_type="spatial",
        author="rahul",
        date="2026-07-22",
        target_ann_id=0,
    )
    base.update(overrides)
    return Phrase(**base)


def test_fixture_phrase_set_loads_and_validates_clean():
    phrase_set = load_phrases(FIXTURE_PHRASES)
    assert len(phrase_set) == 8
    assert phrase_set.is_synthetic is True

    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=150)
    assert result.ok, result.errors
    assert not result.ambiguous
    assert result.type_counts == {
        "spatial": 2,
        "attribute": 2,
        "relational": 2,
        "negative": 2,
    }


def test_orphan_target_ann_id_is_rejected():
    phrase_set = PhraseSet(phrases=(_base_phrase(target_ann_id=9999),), is_synthetic=True)
    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=1)
    assert not result.ok
    assert any("does not exist" in e for e in result.errors)


def test_negative_naming_a_class_that_is_present_is_rejected():
    """image_id 0 has a real pedestrian (ann_id 0) — claiming it's absent must fail."""
    phrase_set = PhraseSet(
        phrases=(
            _base_phrase(
                phrase="the pedestrian",
                qualifier_type="negative",
                target_ann_id=None,
                absent_class="pedestrian",
            ),
        ),
        is_synthetic=True,
    )
    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=1)
    assert not result.ok
    assert any("IS present" in e for e in result.errors)


def test_missing_qualifier_type_coverage_is_rejected():
    phrase_set = PhraseSet(phrases=(_base_phrase(),), is_synthetic=True)
    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=1)
    assert not result.ok
    assert any("missing qualifier_type coverage" in e for e in result.errors)


def test_real_set_below_min_phrases_is_rejected():
    phrase_set = load_phrases(FIXTURE_PHRASES)
    non_synthetic = PhraseSet(phrases=phrase_set.phrases, is_synthetic=False)
    result = validate_phrases(non_synthetic, FIXTURE_ANNOTATIONS, min_phrases=150)
    assert not result.ok
    assert any("need >= 150" in e for e in result.errors)


def test_round_trip_failure_is_reported():
    """A phrase with characters the tokenizer round-trips differently (design decision 2) must
    be flagged, not silently mis-scored at eval time."""
    phrase_set = PhraseSet(
        phrases=(_base_phrase(phrase="the pedestrian--near the curb!!"),), is_synthetic=True
    )
    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=1)
    assert not result.ok
    assert any("round-trip failed" in e for e in result.errors)


def test_ensure_valid_raises_on_errors():
    phrase_set = PhraseSet(phrases=(_base_phrase(target_ann_id=9999),), is_synthetic=True)
    result = validate_phrases(phrase_set, FIXTURE_ANNOTATIONS, min_phrases=1)
    with pytest.raises(ValueError, match="phrase validation failed"):
        ensure_valid(result)


def test_near_duplicate_same_class_gt_boxes_are_flagged_as_ambiguous():
    """Design decision 4's mechanical half: a target with another same-class GT box at
    IoU >= 0.5 in the same frame is unresolvable by construction."""
    anns = json.loads(pathlib.Path(FIXTURE_ANNOTATIONS).read_text())
    anns["annotations"].append(
        {
            "id": 100,
            "image_id": 0,
            "category_id": 0,  # same class (pedestrian) as ann_id 0
            "bbox": [12, 203, 96, 37],  # identical box -> IoU 1.0
            "area": 3552,
            "iscrowd": 0,
        }
    )
    path = pathlib.Path("tests/fixtures/grounding/_tmp_dup_annotations.json")
    path.write_text(json.dumps(anns))
    try:
        phrase_set = PhraseSet(
            phrases=(
                _base_phrase(),
                _base_phrase(phrase="the small car", qualifier_type="attribute", target_ann_id=2),
                _base_phrase(
                    phrase="the rider behind the pedestrian",
                    qualifier_type="relational",
                    target_ann_id=1,
                ),
                _base_phrase(
                    phrase="the bus",
                    qualifier_type="negative",
                    target_ann_id=None,
                    absent_class="bus",
                ),
            ),
            is_synthetic=True,
        )
        result = validate_phrases(phrase_set, path, min_phrases=1)
        assert result.ok
        assert any("near-duplicate" in a for a in result.ambiguous)
    finally:
        path.unlink()


def _copy_fixture_phrases(dest: pathlib.Path) -> None:
    dest.write_text(pathlib.Path(FIXTURE_PHRASES).read_text())


def test_freeze_then_edit_makes_assert_frozen_raise(tmp_path):
    phrases_path = tmp_path / "phrases.json"
    lock_path = tmp_path / "phrases.lock.json"
    _copy_fixture_phrases(phrases_path)

    lock = freeze(phrases_path, lock_path=lock_path, author="rahul")
    assert lock["num_phrases"] == 8
    assert lock["sha256"] == phrases_sha256(phrases_path)

    # Untouched: assert_frozen passes.
    assert assert_frozen(phrases_path, lock_path)["sha256"] == lock["sha256"]

    # Edit one character -> must raise.
    phrases_path.write_text(phrases_path.read_text().replace("curb", "curb2"))
    with pytest.raises(RuntimeError, match="has changed since it was frozen"):
        assert_frozen(phrases_path, lock_path)


def test_assert_frozen_without_a_lock_file_raises(tmp_path):
    phrases_path = tmp_path / "phrases.json"
    _copy_fixture_phrases(phrases_path)
    with pytest.raises(RuntimeError, match="no lock file"):
        assert_frozen(phrases_path, tmp_path / "missing.lock.json")
