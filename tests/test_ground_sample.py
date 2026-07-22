"""Spec 04 task 4 — the seeded sampler and overlay renderer. No model is loaded here (both are
pure image/JSON I/O), so this file is intentionally NOT `model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

import json

from gdp.data.core import load_dataset
from gdp.ground.phrases import load_gt_index
from gdp.ground.sample import render_overlay, sample_and_render, sample_frames, write_frame_sample

FIXTURE_ANNOTATIONS = "tests/fixtures/mini_bdd/annotations.json"
FIXTURE_IMAGES = "tests/fixtures/mini_bdd"


def _dataset():
    return load_dataset(FIXTURE_ANNOTATIONS, FIXTURE_IMAGES)


def test_sample_frames_is_deterministic_for_a_fixed_seed():
    ds = _dataset()
    a = sample_frames(ds, n=3, seed=42)
    b = sample_frames(ds, n=3, seed=42)
    assert a.image_ids == b.image_ids
    assert len(a.image_ids) == 3


def test_sample_frames_different_seeds_can_differ():
    ds = _dataset()
    a = sample_frames(ds, n=2, seed=1)
    b = sample_frames(ds, n=2, seed=2)
    # Not a strict inequality assertion (small sample space could coincide), just that both are
    # valid, in-range subsets.
    assert set(a.image_ids) <= {s.image_id for s in ds.samples}
    assert set(b.image_ids) <= {s.image_id for s in ds.samples}


def test_sample_frames_caps_n_to_dataset_size():
    ds = _dataset()
    sample = sample_frames(ds, n=999, seed=0)
    assert len(sample.image_ids) == len(ds.samples)


def test_write_frame_sample_records_seed_and_timestamp(tmp_path):
    ds = _dataset()
    sample = sample_frames(ds, n=2, seed=7)
    out_path = write_frame_sample(sample, out_path=tmp_path / "frames.json")

    payload = json.loads(out_path.read_text())
    assert payload["seed"] == 7
    assert payload["image_ids"] == list(sample.image_ids)
    assert "sampled_at" in payload
    assert "git_sha" in payload


def test_render_overlay_draws_a_labelled_box_per_gt_annotation(tmp_path):
    ds = _dataset()
    sample = next(s for s in ds.samples if s.image_id == 0)
    anns_by_id, ann_ids_by_image, _ = load_gt_index(FIXTURE_ANNOTATIONS)
    gt_boxes = [anns_by_id[aid] for aid in ann_ids_by_image[0]]
    assert len(gt_boxes) == 3  # image_id 0 has 3 GT boxes in the fixture

    out_path = render_overlay(sample.image_path, gt_boxes, out_path=tmp_path / "overlay.png")
    assert out_path.is_file()
    # Overlay must not alter the underlying image dimensions.
    from PIL import Image

    with Image.open(out_path) as img:
        assert img.size == (sample.width, sample.height)


def test_sample_and_render_writes_frames_json_and_one_overlay_per_frame(tmp_path):
    ds = _dataset()
    frames_json_path = tmp_path / "frames.json"
    overlays_dir = tmp_path / "overlays"

    frame_sample, overlay_paths = sample_and_render(
        ds,
        FIXTURE_ANNOTATIONS,
        n=4,
        seed=42,
        frames_json_path=frames_json_path,
        overlays_dir=overlays_dir,
    )

    assert frames_json_path.is_file()
    assert len(overlay_paths) == len(frame_sample.image_ids) == 4
    for path in overlay_paths:
        assert path.is_file()


def test_sample_and_render_same_seed_reproduces_the_same_frame_list(tmp_path):
    ds = _dataset()
    first, _ = sample_and_render(
        ds,
        FIXTURE_ANNOTATIONS,
        n=4,
        seed=42,
        frames_json_path=tmp_path / "a" / "frames.json",
        overlays_dir=tmp_path / "a" / "overlays",
    )
    second, _ = sample_and_render(
        ds,
        FIXTURE_ANNOTATIONS,
        n=4,
        seed=42,
        frames_json_path=tmp_path / "b" / "frames.json",
        overlays_dir=tmp_path / "b" / "overlays",
    )
    assert first.image_ids == second.image_ids
