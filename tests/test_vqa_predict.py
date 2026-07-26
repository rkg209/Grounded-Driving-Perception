"""Spec 08 task 3: `gdp.vqa.predict` — resumable prediction generation and its completeness gate.

The resumability tests need no model at all (an already-complete `out_path` must never touch
`processor`/`model`). The one test that actually calls `generate_with_stats` is `model_heavy`
(CLAUDE.md §4), using the same few-layer randomly-initialized Qwen2.5-VL as
`tests/test_vqa_lora.py`/`tests/test_vqa_train_loop.py` — real architecture, megabytes not
gigabytes.
"""

from __future__ import annotations

import json

import pytest

from gdp.config import VQAEvalConfig, load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.predict import already_predicted, assert_complete, load_predictions, predict_split


@pytest.fixture(scope="module")
def val_records():
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    _, val, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    return val


def test_load_predictions_missing_file_is_empty(tmp_path):
    assert load_predictions(tmp_path / "nope.jsonl") == []


def test_already_predicted_returns_qa_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"qa_id": "a"}\n{"qa_id": "b"}\n')
    assert already_predicted(path) == {"a", "b"}


def test_assert_complete_passes_when_sets_match(val_records):
    predictions = [{"qa_id": r.qa_id} for r in val_records]
    assert_complete(predictions, val_records)  # no raise


def test_assert_complete_raises_on_missing(val_records):
    predictions = [{"qa_id": r.qa_id} for r in val_records[:-1]]
    with pytest.raises(ValueError, match="missing qa_id"):
        assert_complete(predictions, val_records)


def test_assert_complete_raises_on_extra(val_records):
    predictions = [{"qa_id": r.qa_id} for r in val_records] + [{"qa_id": "not-in-split"}]
    with pytest.raises(ValueError, match="extra qa_id"):
        assert_complete(predictions, val_records)


def test_predict_split_skips_generation_when_all_already_present(tmp_path, val_records):
    """The resumability contract: an already-complete file must never touch processor/model —
    proven by passing `None` for both and asserting no crash and no new lines."""
    out_path = tmp_path / "predictions.jsonl"
    with out_path.open("w") as f:
        for r in val_records:
            f.write(json.dumps({"qa_id": r.qa_id, "prediction": "already done"}) + "\n")

    result_path = predict_split(
        val_records,
        None,
        None,
        cfg=VQAEvalConfig(),
        nuscenes_root="unused",
        device="cpu",
        model_role="base",
        out_path=out_path,
    )
    assert result_path == out_path
    rows = load_predictions(out_path)
    assert len(rows) == len(val_records)
    assert all(row["prediction"] == "already done" for row in rows)


@pytest.mark.model_heavy
def test_predict_split_generates_and_resumes(tmp_path, val_records):
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

    from gdp.config import load_config

    text_config = Qwen2_5_VLTextConfig(
        vocab_size=152064,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=151643,
        eos_token_id=151645,
        rope_parameters={
            "type": "mrope",
            "mrope_section": [2, 1, 1],
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        spatial_merge_size=2,
        fullatt_block_indexes=(0, 1),
    )
    config = Qwen2_5_VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    drivelm_cfg = load_config("configs/default.yaml").drivelm
    records = val_records[:2]
    out_path = tmp_path / "predictions.jsonl"

    predict_split(
        records,
        processor,
        model,
        cfg=VQAEvalConfig(max_new_tokens=4),
        nuscenes_root=drivelm_cfg.nuscenes_root_path(),
        device=torch.device("cpu"),
        model_role="base",
        out_path=out_path,
    )
    rows = load_predictions(out_path)
    assert len(rows) == 2
    assert {r["qa_id"] for r in rows} == {r.qa_id for r in records}
    for row in rows:
        assert row["model_role"] == "base"
        assert row["num_input_tokens"] > 0
        assert row["seconds"] >= 0

    # resuming against the same out_path must not regenerate already-done items
    predict_split(
        records,
        processor,
        model,
        cfg=VQAEvalConfig(max_new_tokens=4),
        nuscenes_root=drivelm_cfg.nuscenes_root_path(),
        device=torch.device("cpu"),
        model_role="base",
        out_path=out_path,
    )
    assert len(load_predictions(out_path)) == 2
