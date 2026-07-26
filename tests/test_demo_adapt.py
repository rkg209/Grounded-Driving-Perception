"""`sample_from_image`/`record_from_image` shims (spec 09 task 4). No model loaded — not
`model_heavy`."""

from __future__ import annotations

from gdp.demo.adapt import record_from_image, sample_from_image
from gdp.paths import resolve

FIXTURES_DIR = resolve("tests/fixtures/mini_bdd")
SCENE_PATH = FIXTURES_DIR / "scene_000.jpg"


def test_sample_from_image_has_empty_boxes():
    sample = sample_from_image(SCENE_PATH)
    assert sample.boxes == ()
    assert sample.image_id == 0
    assert sample.width > 0 and sample.height > 0


def test_sample_from_image_custom_image_id():
    sample = sample_from_image(SCENE_PATH, image_id=7)
    assert sample.image_id == 7


def test_record_from_image_never_carries_a_real_answer():
    record = record_from_image(SCENE_PATH, "Is it safe to proceed?", scenes_root=FIXTURES_DIR)
    assert record.answer == ""
    assert record.tag == []
    assert record.object_tags == []


def test_record_from_image_single_cam_front_view():
    record = record_from_image(SCENE_PATH, "What is ahead?", scenes_root=FIXTURES_DIR)
    assert record.view_order == ["CAM_FRONT"]
    assert set(record.image_paths) == {"CAM_FRONT"}
    assert record.image_paths["CAM_FRONT"] == "scene_000.jpg"


def test_record_from_image_marks_fixture_scenes_synthetic():
    record = record_from_image(SCENE_PATH, "What is ahead?", scenes_root=FIXTURES_DIR)
    assert record.is_synthetic is True


def test_record_from_image_outside_scenes_root_uses_absolute_path(tmp_path):
    from PIL import Image

    upload_path = tmp_path / "upload.jpg"
    Image.new("RGB", (16, 16)).save(upload_path)
    record = record_from_image(upload_path, "What is ahead?", scenes_root=FIXTURES_DIR)
    assert record.image_paths["CAM_FRONT"] == str(upload_path.resolve())
    assert record.is_synthetic is False
