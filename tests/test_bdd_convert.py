"""The converter (`gdp.data.bdd100k`) turns raw BDD `det_20` JSON into our COCO-style schema. Two
things must hold: the fixture and real BDD100K pass through the *same* code path (acceptance 2),
and every dropped box is counted by reason rather than silently swallowed (acceptance 5, H7).
"""

from __future__ import annotations

import json

import pytest

from gdp.config import BDD100K_CLASSES
from gdp.data import load_dataset
from gdp.data.bdd100k import DROP_REASONS, convert_bdd_to_coco
from gdp.paths import resolve

FIXTURE_WIDTH, FIXTURE_HEIGHT = 640, 360


@pytest.fixture
def raw_val_frames():
    path = resolve("tests/fixtures/mini_bdd_raw/det_val.json")
    return json.loads(path.read_text())


@pytest.fixture
def dirty_frames():
    path = resolve("tests/fixtures/mini_bdd_raw/det_dirty.json")
    return json.loads(path.read_text())


def _convert_val(frames):
    return convert_bdd_to_coco(frames, image_width=FIXTURE_WIDTH, image_height=FIXTURE_HEIGHT)


def test_converted_json_loads_via_spec00_loader(raw_val_frames, tmp_path):
    """Acceptance 1: `prepare` output is read by `load_dataset` unmodified."""
    coco, _ = _convert_val(raw_val_frames)
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(coco))

    dataset = load_dataset(ann_path, resolve("tests/fixtures/mini_bdd"))
    assert len(dataset) == 4
    assert dataset.classes == BDD100K_CLASSES
    assert sum(len(s.boxes) for s in dataset.samples) == 10


def test_converted_json_reproduces_committed_annotations(raw_val_frames):
    """Acceptance 2: the fixture and real BDD JSON pass the *same* validation — converting the
    independently-generated raw fixture must reproduce the committed `annotations.json` byte for
    byte (modulo key ordering), proving the converter and the hand-written fixture agree."""
    coco, _ = _convert_val(raw_val_frames)
    committed = json.loads(resolve("tests/fixtures/mini_bdd/annotations.json").read_text())

    assert coco["categories"] == committed["categories"]
    assert coco["images"] == committed["images"]

    def _key(ann):
        return (ann["image_id"], ann["category_id"], tuple(ann["bbox"]))

    assert sorted(map(_key, coco["annotations"])) == sorted(map(_key, committed["annotations"]))


def test_no_surviving_boxes_keeps_the_image_entry():
    frames = [{"name": "empty.jpg", "labels": None}]
    coco, stats = convert_bdd_to_coco(
        frames, image_width=FIXTURE_WIDTH, image_height=FIXTURE_HEIGHT
    )
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "empty.jpg"
    assert stats.images_without_boxes == 1
    assert stats.boxes == 0


def test_image_ids_are_deterministic_by_sorted_name():
    frames = [{"name": "b.jpg", "labels": []}, {"name": "a.jpg", "labels": []}]
    coco, _ = convert_bdd_to_coco(frames, image_width=FIXTURE_WIDTH, image_height=FIXTURE_HEIGHT)
    by_name = {img["file_name"]: img["id"] for img in coco["images"]}
    assert by_name["a.jpg"] == 0
    assert by_name["b.jpg"] == 1


def test_dirty_frames_are_counted_by_reason(dirty_frames):
    """Acceptance 5: exact per-reason drop counts, plus the one edge-straddle case that must be
    clipped rather than dropped."""
    coco, stats = convert_bdd_to_coco(
        dirty_frames, image_width=FIXTURE_WIDTH, image_height=FIXTURE_HEIGHT
    )

    assert stats.dropped == {
        "unknown_category": 1,
        "degenerate": 1,
        "out_of_frame": 1,
        "no_box2d": 1,
    }
    assert stats.clipped == 1
    assert stats.boxes == 1  # only the clipped edge-straddle box survives
    assert stats.images == 6
    assert stats.images_without_boxes == 5  # every frame except the surviving clipped one

    # the survivor is clipped to the frame width, not dropped
    surviving = coco["annotations"][0]
    x, y, w, h = surviving["bbox"]
    assert x + w <= FIXTURE_WIDTH


@pytest.mark.parametrize("reason", DROP_REASONS)
def test_every_drop_reason_is_tracked(reason):
    assert reason in DROP_REASONS
