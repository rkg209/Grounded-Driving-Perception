"""The split-preservation guarantee at the heart of spec 06 (H2). nuScenes samples at 2 Hz from a
continuous drive, so a frame- or QA-level shuffle would put near-duplicate images in both train and
val and inflate Stage-2 accuracy. Everything here keeps the sampling unit the *scene*.
"""

from __future__ import annotations

import json

import pytest

from gdp.data.drivelm import DriveLMRecord, assert_no_scene_overlap
from gdp.data.splits import load_official_splits, load_scene_meta, resolve_split


def _record(scene_token: str, split: str) -> DriveLMRecord:
    return DriveLMRecord(
        qa_id=f"{scene_token}-qa",
        scene_token=scene_token,
        frame_token="frame",
        category="perception",
        question="Q?",
        answer="A.",
        object_tags=[],
        image_paths={},
        view_order=[],
        official_split=split,
        is_synthetic=True,
    )


def test_vendored_split_counts():
    splits = load_official_splits()
    assert len(splits.train) == 700
    assert len(splits.val) == 150
    assert splits.train.isdisjoint(splits.val)


def test_vendored_split_has_provenance():
    splits = load_official_splits()
    assert splits.provenance["upstream_package"] == "nuscenes-devkit"
    assert "content_sha256" in splits.provenance


def test_devkit_cross_check():
    """If nuscenes-devkit happens to be installed, our vendored copy must match it exactly.

    Not a required dependency (spec 06 design decision 1) — this test is skipped, not failed,
    when the package is absent.
    """
    devkit = pytest.importorskip("nuscenes.utils.splits")
    splits = load_official_splits()
    assert splits.train == frozenset(devkit.train)
    assert splits.val == frozenset(devkit.val)


def test_load_scene_meta(tmp_path):
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(
        json.dumps(
            [
                {"token": "aaa", "name": "scene-0001"},
                {"token": "bbb", "name": "scene-0003"},
            ]
        )
    )
    meta = load_scene_meta(scene_json)
    assert meta == {"aaa": "scene-0001", "bbb": "scene-0003"}


def test_load_scene_meta_rejects_conflicting_names(tmp_path):
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(
        json.dumps(
            [
                {"token": "aaa", "name": "scene-0001"},
                {"token": "aaa", "name": "scene-0002"},
            ]
        )
    )
    with pytest.raises(ValueError, match="conflicting names"):
        load_scene_meta(scene_json)


def test_resolve_split_train_and_val():
    splits = load_official_splits()
    train_name = next(iter(splits.train))
    val_name = next(iter(splits.val))
    scene_meta = {"tok-train": train_name, "tok-val": val_name}

    assert resolve_split("tok-train", scene_meta=scene_meta, splits=splits) == "train"
    assert resolve_split("tok-val", scene_meta=scene_meta, splits=splits) == "val"


def test_resolve_split_unknown_token_is_hard_error():
    splits = load_official_splits()
    with pytest.raises(KeyError, match="not found in scene.json"):
        resolve_split("missing-token", scene_meta={}, splits=splits)


def test_resolve_split_unknown_scene_name_is_hard_error():
    splits = load_official_splits()
    scene_meta = {"tok": "scene-9999"}  # not in either official list
    with pytest.raises(KeyError, match="neither the official"):
        resolve_split("tok", scene_meta=scene_meta, splits=splits)


def test_assert_no_scene_overlap_passes_on_disjoint():
    train = [_record("tok-a", "train"), _record("tok-b", "train")]
    val = [_record("tok-c", "val")]
    assert_no_scene_overlap(train, val)  # must not raise


def test_assert_no_scene_overlap_raises_on_hand_crafted_overlap():
    """The non-negotiable gate: a scene token in both splits must raise, listing the token."""
    train = [_record("leaky-token", "train"), _record("tok-b", "train")]
    val = [_record("leaky-token", "val")]
    with pytest.raises(ValueError, match="leaky-token"):
        assert_no_scene_overlap(train, val)
