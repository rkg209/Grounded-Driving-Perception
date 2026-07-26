"""The converter: DriveLM's raw nested JSON -> flat `DriveLMRecord`s, drop accounting exercised
against every malformed case `scripts/make_fixtures.py` bakes into `mini_drivelm`.
"""

from __future__ import annotations

import json

import pytest

from gdp.config import load_config
from gdp.data.drivelm import (
    DROP_REASONS,
    VIEW_ORDER,
    convert_drivelm,
    parse_object_tags,
    write_jsonl,
)
from gdp.data.splits import load_official_splits, load_scene_meta


@pytest.fixture(scope="module")
def drivelm_cfg():
    return load_config("configs/default.yaml").drivelm


@pytest.fixture(scope="module")
def raw(drivelm_cfg):
    return json.loads(drivelm_cfg.annotations_path().read_text())


@pytest.fixture(scope="module")
def converted(drivelm_cfg, raw):
    scene_meta = load_scene_meta(drivelm_cfg.scene_meta_path())
    splits = load_official_splits()
    return convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=drivelm_cfg.nuscenes_root_path(),
        categories=tuple(drivelm_cfg.categories),
        is_synthetic=True,
    )


def test_parse_object_tags():
    tags = parse_object_tags("There is a pedestrian <c1,CAM_FRONT,1088.3,497.5> crossing.")
    assert tags == [{"ref": "c1", "camera": "CAM_FRONT", "x": 1088.3, "y": 497.5}]


def test_parse_object_tags_none_present():
    assert parse_object_tags("Nothing tagged here.") == []


def test_parse_object_tags_multiple():
    text = "See <c1,CAM_FRONT,1.0,2.0> and <c2,CAM_BACK,3.5,4.5>."
    tags = parse_object_tags(text)
    assert len(tags) == 2
    assert tags[0]["camera"] == "CAM_FRONT"
    assert tags[1]["ref"] == "c2"


def test_scenes_and_frames_counted(converted, raw):
    _, _, stats = converted
    assert stats.scenes == len(raw)
    assert stats.frames == sum(len(s["key_frames"]) for s in raw.values())


def test_every_drop_reason_is_exercised(converted):
    _, _, stats = converted
    for reason in DROP_REASONS:
        assert stats.dropped[reason] > 0, f"drop reason {reason!r} never triggered by the fixture"


def test_qa_total_equals_kept_plus_dropped(converted):
    _, _, stats = converted
    assert stats.qa_total == stats.qa_kept + sum(stats.dropped.values())


def test_train_val_records_go_to_correct_split(converted):
    train_records, val_records, _ = converted
    assert train_records, "expected at least one surviving train record"
    assert val_records, "expected at least one surviving val record"
    assert all(r.official_split == "train" for r in train_records)
    assert all(r.official_split == "val" for r in val_records)
    assert {r.scene_token for r in train_records}.issubset(
        {
            s["token"]
            for s in json.loads(
                (load_config("configs/default.yaml").drivelm.scene_meta_path()).read_text()
            )
        }
    )


def test_question_and_answer_are_byte_verbatim(converted, raw):
    train_records, val_records, _ = converted
    for record in train_records + val_records:
        scene = raw[record.scene_token]
        frame = scene["key_frames"][record.frame_token]
        qa_list = frame["QA"][record.category]
        matched = [qa for qa in qa_list if qa["Q"] == record.question and qa["A"] == record.answer]
        assert matched, f"record {record.qa_id} not found verbatim in raw QA"


def test_object_tags_parsed_on_surviving_record(converted):
    train_records, val_records, _ = converted
    tagged = [r for r in train_records + val_records if r.object_tags]
    assert tagged
    assert tagged[0].object_tags[0]["camera"] == "CAM_FRONT"


def test_image_paths_carry_all_six_views_on_kept_records(converted):
    train_records, val_records, _ = converted
    for record in train_records + val_records:
        assert set(record.image_paths) == set(VIEW_ORDER)
        assert record.view_order == list(VIEW_ORDER)


def test_qa_ids_are_deterministic_and_unique(converted):
    train_records, val_records, _ = converted
    ids = [r.qa_id for r in train_records + val_records]
    assert len(ids) == len(set(ids))


def test_rerun_is_byte_identical(drivelm_cfg, raw):
    scene_meta = load_scene_meta(drivelm_cfg.scene_meta_path())
    splits = load_official_splits()
    run1 = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=drivelm_cfg.nuscenes_root_path(),
        categories=tuple(drivelm_cfg.categories),
        is_synthetic=True,
    )
    run2 = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=drivelm_cfg.nuscenes_root_path(),
        categories=tuple(drivelm_cfg.categories),
        is_synthetic=True,
    )
    assert [r.to_json() for r in run1[0]] == [r.to_json() for r in run2[0]]
    assert [r.to_json() for r in run1[1]] == [r.to_json() for r in run2[1]]
    assert run1[2].to_json() == run2[2].to_json()


def test_fixture_split_is_scene_clean(converted):
    """`convert_drivelm` calls `assert_no_scene_overlap` internally — reaching this point without
    raising already proves the fixture's split is scene-clean, but assert it explicitly too."""
    train_records, val_records, _ = converted
    train_tokens = {r.scene_token for r in train_records}
    val_tokens = {r.scene_token for r in val_records}
    assert train_tokens.isdisjoint(val_tokens)


def test_convert_drivelm_unknown_scene_token_is_hard_error(drivelm_cfg):
    splits = load_official_splits()
    raw = {
        "unknown-token": {
            "key_frames": {
                "frame-1": {
                    "image_paths": {},
                    "QA": {"perception": [{"Q": "Q?", "A": "A."}]},
                }
            }
        }
    }
    with pytest.raises(KeyError, match="not found in scene.json"):
        convert_drivelm(
            raw,
            scene_meta={},
            splits=splits,
            images_root=drivelm_cfg.nuscenes_root_path(),
            categories=tuple(drivelm_cfg.categories),
            is_synthetic=True,
        )


def test_write_jsonl_is_sorted_and_byte_identical(tmp_path, converted):
    train_records, _, _ = converted
    p1 = write_jsonl(train_records, tmp_path / "a.jsonl")
    p2 = write_jsonl(list(reversed(train_records)), tmp_path / "b.jsonl")
    assert p1.read_text() == p2.read_text()

    lines = p1.read_text().splitlines()
    ids = [json.loads(line)["qa_id"] for line in lines]
    assert ids == sorted(ids)
