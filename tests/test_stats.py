"""The sanity report is dataset statistics, not a metric (H9) — these tests check the report is
built and written correctly, not that any number in it is good."""

from __future__ import annotations

import json

import pytest

from gdp.config import BDD100K_CLASSES
from gdp.data.bdd100k import convert_bdd_to_coco
from gdp.data.stats import build_stats, write_stats
from gdp.paths import resolve, run_dir


@pytest.fixture
def raw_val_frames():
    path = resolve("tests/fixtures/mini_bdd_raw/det_val.json")
    return json.loads(path.read_text())


def test_stats_reports_all_ten_classes(raw_val_frames):
    """Acceptance 4: class-count report is written; counts are non-zero for all 10 — a fixture-only
    assertion, since real BDD100K may genuinely have zero `train` boxes in val (H7)."""
    coco, conv_stats = convert_bdd_to_coco(raw_val_frames, image_width=640, image_height=360)
    stats = build_stats(
        coco,
        conv_stats,
        split="val",
        source_file=resolve("tests/fixtures/mini_bdd_raw/det_val.json"),
    )
    assert set(stats["per_class"]) == set(BDD100K_CLASSES)
    for name in BDD100K_CLASSES:
        assert stats["per_class"][name] > 0, f"{name} has zero boxes in the fixture"


def test_stats_flags_rare_classes_below_threshold(raw_val_frames):
    coco, conv_stats = convert_bdd_to_coco(raw_val_frames, image_width=640, image_height=360)
    stats = build_stats(
        coco,
        conv_stats,
        split="val",
        source_file=resolve("tests/fixtures/mini_bdd_raw/det_val.json"),
        min_boxes_for_eval=2,
    )
    # each fixture class has exactly 1 box, so all are below a threshold of 2
    assert set(stats["rare_classes"]) == set(BDD100K_CLASSES)


def test_stats_records_drop_and_clip_counts(raw_val_frames):
    coco, conv_stats = convert_bdd_to_coco(raw_val_frames, image_width=640, image_height=360)
    stats = build_stats(
        coco,
        conv_stats,
        split="val",
        source_file=resolve("tests/fixtures/mini_bdd_raw/det_val.json"),
    )
    assert stats["dropped"] == {
        "unknown_category": 0,
        "degenerate": 0,
        "out_of_frame": 0,
        "no_box2d": 0,
    }
    assert stats["clipped"] == 0
    assert stats["images"] == 4
    assert stats["boxes"] == 10


def test_stats_is_never_presented_as_a_metric(raw_val_frames):
    """H9 guard: stats.json is dataset statistics, and must not carry a field that looks like an
    accuracy/mAP number."""
    coco, conv_stats = convert_bdd_to_coco(raw_val_frames, image_width=640, image_height=360)
    stats = build_stats(
        coco,
        conv_stats,
        split="val",
        source_file=resolve("tests/fixtures/mini_bdd_raw/det_val.json"),
    )
    forbidden = {"map", "accuracy", "precision", "recall", "score"}
    assert not (forbidden & set(stats))


def test_write_stats_writes_to_runs_01_data(raw_val_frames):
    coco, conv_stats = convert_bdd_to_coco(raw_val_frames, image_width=640, image_height=360)
    stats = build_stats(
        coco,
        conv_stats,
        split="val",
        source_file=resolve("tests/fixtures/mini_bdd_raw/det_val.json"),
    )
    out_path = write_stats(stats, spec="01-data")
    try:
        assert out_path.is_file()
        assert out_path.parent.parent == run_dir("01-data").parent
        written = json.loads(out_path.read_text())
        assert written["boxes"] == 10
        assert written["source"]["file"] == "det_val.json"
        assert "git_sha" in written and "created" in written
    finally:
        out_path.unlink()
        try:
            out_path.parent.rmdir()
        except OSError:
            pass
