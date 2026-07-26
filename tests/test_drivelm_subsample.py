"""The scene-level stratified subsampler (spec 06 design decision 7). Sampling unit is the scene,
never a partial scene; val is never touched by this module at all — there is no `val_records`
parameter to misuse.
"""

from __future__ import annotations

import json

import pytest

from gdp.config import load_config
from gdp.data.drivelm import (
    convert_drivelm,
    subsample_scenes,
    write_subsample_json,
)
from gdp.data.splits import load_official_splits, load_scene_meta


@pytest.fixture(scope="module")
def train_records():
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    train, _, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    return train


def test_fraction_one_keeps_every_scene(train_records):
    result = subsample_scenes(train_records, fraction=1.0, seed=42)
    assert result.scenes_after == result.scenes_before
    assert result.qa_after == result.qa_before
    assert set(result.selected_scene_tokens) == {r.scene_token for r in train_records}


def test_same_seed_is_reproducible(train_records):
    r1 = subsample_scenes(train_records, fraction=0.5, seed=7)
    r2 = subsample_scenes(train_records, fraction=0.5, seed=7)
    assert r1.selected_scene_tokens == r2.selected_scene_tokens
    assert r1.to_json() == {**r2.to_json(), "created": r1.to_json()["created"]}


def test_sampling_unit_is_whole_scenes_never_partial(train_records):
    """A selected scene contributes *every* one of its QA records, or none at all — never some."""
    result = subsample_scenes(train_records, fraction=0.5, seed=7)
    selected = set(result.selected_scene_tokens)
    assert selected <= {r.scene_token for r in train_records}

    records_by_scene: dict[str, int] = {}
    for r in train_records:
        records_by_scene[r.scene_token] = records_by_scene.get(r.scene_token, 0) + 1
    kept_by_scene: dict[str, int] = {}
    for r in train_records:
        if r.scene_token in selected:
            kept_by_scene[r.scene_token] = kept_by_scene.get(r.scene_token, 0) + 1

    for token in selected:
        assert kept_by_scene[token] == records_by_scene[token]


def test_qa_after_matches_selected_scenes_exactly(train_records):
    result = subsample_scenes(train_records, fraction=0.5, seed=7)
    selected = set(result.selected_scene_tokens)
    expected = sum(1 for r in train_records if r.scene_token in selected)
    assert result.qa_after == expected


def test_category_distribution_and_l1_drift_fields_present(train_records):
    result = subsample_scenes(train_records, fraction=0.5, seed=7)
    assert result.category_distribution_before
    assert result.category_distribution_after
    assert result.l1_drift >= 0.0
    assert result.unit == "scene"


def test_invalid_fraction_rejected(train_records):
    with pytest.raises(ValueError, match="fraction must be in"):
        subsample_scenes(train_records, fraction=0.0, seed=1)
    with pytest.raises(ValueError, match="fraction must be in"):
        subsample_scenes(train_records, fraction=1.5, seed=1)


def test_write_subsample_json(tmp_path, train_records):
    result = subsample_scenes(train_records, fraction=1.0, seed=42)
    path = write_subsample_json(result, tmp_path / "subsample.json")
    written = json.loads(path.read_text())
    assert written["unit"] == "scene"
    assert written["seed"] == 42
    assert written["selected_scene_tokens"] == result.selected_scene_tokens
