"""Chat formatting for Qwen2.5-VL (spec 06 design decision 8). `build_messages` is pure and cheap;
the real-processor round-trip is `model_heavy` (CLAUDE.md §4) — run deliberately with:

    uv run pytest -m model_heavy tests/test_vqa_chat.py
"""

from __future__ import annotations

import json

import pytest

from gdp.config import load_config
from gdp.data.drivelm import DriveLMRecord, convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.chat import build_messages, format_example, load_jsonl


def _sample_record() -> DriveLMRecord:
    return DriveLMRecord(
        qa_id="tok-frame-perception-000",
        scene_token="tok",
        frame_token="frame",
        category="perception",
        question="What objects are visible?",
        answer="There is a pedestrian <c1,CAM_FRONT,100.0,50.0> near the crosswalk.",
        object_tags=[{"ref": "c1", "camera": "CAM_FRONT", "x": 100.0, "y": 50.0}],
        tag=[2],
        image_paths={
            "CAM_FRONT": "samples/CAM_FRONT/a.jpg",
            "CAM_FRONT_LEFT": "samples/CAM_FRONT_LEFT/a.jpg",
            "CAM_FRONT_RIGHT": "samples/CAM_FRONT_RIGHT/a.jpg",
            "CAM_BACK": "samples/CAM_BACK/a.jpg",
            "CAM_BACK_LEFT": "samples/CAM_BACK_LEFT/a.jpg",
            "CAM_BACK_RIGHT": "samples/CAM_BACK_RIGHT/a.jpg",
        },
        view_order=[
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        official_split="train",
        is_synthetic=True,
    )


def test_build_messages_default_num_views_is_cam_front_only():
    record = _sample_record()
    messages = build_messages(record)
    user_content = messages[0]["content"]
    images = [c for c in user_content if c["type"] == "image"]
    assert len(images) == 1
    assert images[0]["image"] == "samples/CAM_FRONT/a.jpg"


def test_build_messages_honours_num_views():
    record = _sample_record()
    messages = build_messages(record, num_views=3)
    images = [c for c in messages[0]["content"] if c["type"] == "image"]
    assert [c["image"] for c in images] == [
        "samples/CAM_FRONT/a.jpg",
        "samples/CAM_FRONT_LEFT/a.jpg",
        "samples/CAM_FRONT_RIGHT/a.jpg",
    ]


def test_build_messages_question_and_answer_are_verbatim():
    record = _sample_record()
    messages = build_messages(record, num_views=1)
    user_text = next(c["text"] for c in messages[0]["content"] if c["type"] == "text")
    assistant_text = next(c["text"] for c in messages[1]["content"] if c["type"] == "text")
    assert user_text == record.question
    assert assistant_text == record.answer
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_build_messages_rejects_out_of_range_num_views():
    record = _sample_record()
    with pytest.raises(ValueError, match="num_views must be in"):
        build_messages(record, num_views=0)
    with pytest.raises(ValueError, match="num_views must be in"):
        build_messages(record, num_views=7)


def test_load_jsonl_round_trips_write_jsonl(tmp_path):
    from gdp.data.drivelm import write_jsonl

    record = _sample_record()
    path = write_jsonl([record], tmp_path / "r.jsonl")
    loaded = load_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0] == record


@pytest.fixture(scope="module")
def qwen_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


@pytest.mark.model_heavy
def test_format_example_round_trips_real_processor(qwen_processor):
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    train_records, _, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    record = train_records[0]

    example = format_example(qwen_processor, record, num_views=1)
    assert example["qa_id"] == record.qa_id
    assert "<|im_start|>" in example["text"]
    assert "<|vision_start|>" in example["text"]
    assert example["image_paths"] == [record.image_paths["CAM_FRONT"]]
