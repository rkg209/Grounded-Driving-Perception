"""`Scene`/`load_scenes`/`scene_badge` (spec 09 task 4). No model loaded — not `model_heavy`."""

from __future__ import annotations

from gdp.demo.honesty import SYNTHETIC_SCENE
from gdp.demo.scenes import UPLOADED_SCENE_BADGE, load_scenes, scene_badge
from gdp.paths import resolve

FIXTURES_DIR = resolve("tests/fixtures/mini_bdd")


def test_load_scenes_finds_fixture_images():
    scenes = load_scenes(FIXTURES_DIR)
    assert len(scenes) >= 4
    names = {s.name for s in scenes}
    assert "scene_000" in names


def test_load_scenes_returns_empty_for_missing_dir(tmp_path):
    assert load_scenes(tmp_path / "does_not_exist") == []


def test_fixture_scenes_are_badged_synthetic():
    scenes = load_scenes(FIXTURES_DIR)
    for scene in scenes:
        assert scene.is_synthetic
        assert scene_badge(scene) == SYNTHETIC_SCENE


def test_non_fixture_scene_is_not_badged_synthetic(tmp_path):
    from PIL import Image

    img_path = tmp_path / "upload.jpg"
    Image.new("RGB", (16, 16)).save(img_path)
    scene = load_scenes(tmp_path)[0]
    assert not scene.is_synthetic
    assert scene_badge(scene) == UPLOADED_SCENE_BADGE
