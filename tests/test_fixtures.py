"""The synthetic mini-BDD fixture is the thing that keeps this repo testable with no dataset and
no cluster. If these tests break, every later spec loses its offline verification path.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from gdp.config import BDD100K_CLASSES, load_config
from gdp.data import Box, load_dataset


@pytest.fixture(scope="module")
def dataset():
    cfg = load_config("configs/default.yaml")
    return load_dataset(cfg.dataset.annotations, cfg.dataset.root)


def test_fixture_loads(dataset):
    assert len(dataset) == 4
    assert dataset.classes == BDD100K_CLASSES


def test_every_image_exists_and_matches_declared_size(dataset):
    for s in dataset.samples:
        assert s.image_path.is_file()
        with Image.open(s.image_path) as im:
            assert im.size == (s.width, s.height)


def test_every_image_has_boxes_and_they_are_inside_the_image(dataset):
    for s in dataset.samples:
        assert s.boxes, f"{s.image_path.name} has no boxes"
        for b in s.boxes:
            assert 0 <= b.x0 < b.x1 <= s.width
            assert 0 <= b.y0 < b.y1 <= s.height
            assert 0 <= b.class_id < len(dataset.classes)
            assert b.area > 0


def test_prompt_is_period_separated(dataset):
    prompt = dataset.prompt()
    assert prompt.startswith("pedestrian. rider. car.")
    assert prompt.endswith(".")
    # one class per period-separated segment
    assert len([p for p in prompt.split(".") if p.strip()]) == len(dataset.classes)


def test_degenerate_box_is_rejected():
    with pytest.raises(ValueError, match="degenerate box"):
        Box(x0=10, y0=10, x1=10, y1=20, class_id=0)


def test_unknown_category_is_rejected(tmp_path):
    bad = tmp_path / "annotations.json"
    bad.write_text(
        json.dumps(
            {
                "categories": [{"id": 0, "name": "car"}],
                "images": [{"id": 0, "file_name": "x.jpg", "width": 10, "height": 10}],
                "annotations": [{"id": 0, "image_id": 0, "category_id": 99, "bbox": [0, 0, 5, 5]}],
            }
        )
    )
    with pytest.raises(ValueError, match="unknown category_id"):
        load_dataset(bad, tmp_path)


def test_missing_image_is_rejected(tmp_path):
    bad = tmp_path / "annotations.json"
    bad.write_text(
        json.dumps(
            {
                "categories": [{"id": 0, "name": "car"}],
                "images": [{"id": 0, "file_name": "gone.jpg", "width": 10, "height": 10}],
                "annotations": [],
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        load_dataset(bad, tmp_path)


def test_fixture_is_labelled_as_synthetic(dataset):
    """Guard against H7: nobody may mistake this for real BDD100K."""
    cfg = load_config("configs/default.yaml")
    raw = json.loads(cfg.dataset.annotations_path().read_text())
    assert "NOT real BDD100K" in raw["info"]["description"]
