"""Dataset + collator coverage for `gdp.vqa.dataset` (spec 07 task 3).

Every test drives the real Qwen2.5-VL-3B-Instruct processor against the synthetic `mini_drivelm`
fixture — `model_heavy` (CLAUDE.md §4). Run deliberately with:

    uv run pytest -m model_heavy tests/test_vqa_dataset.py
"""

from __future__ import annotations

import json

import pytest
import torch

from gdp.config import load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.dataset import DriveLMVQADataset, VQACollator
from gdp.vqa.labels import processor_pixel_budget

pytestmark = pytest.mark.model_heavy


@pytest.fixture(scope="module")
def qwen_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


@pytest.fixture(scope="module")
def train_records():
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    records, _, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    return records


@pytest.fixture(scope="module")
def nuscenes_root():
    return load_config("configs/default.yaml").drivelm.nuscenes_root_path()


@pytest.fixture(scope="module")
def mixed_batch(qwen_processor, train_records, nuscenes_root):
    """Two examples at num_views=1, one at num_views=2 — guarantees mixed image counts and, since
    a second image adds prompt tokens, mixed sequence lengths too (group G)."""
    ds1 = DriveLMVQADataset(train_records[:2], qwen_processor, nuscenes_root, num_views=1)
    ds2 = DriveLMVQADataset(train_records[2:3], qwen_processor, nuscenes_root, num_views=2)
    return [ds1[0], ds1[1], ds2[0]]


# Group G — mixed-length, mixed-image-count batch shapes


def test_batch_has_mixed_lengths_and_image_counts(mixed_batch):
    lengths = {e.input_ids.shape[0] for e in mixed_batch}
    image_counts = {e.image_grid_thw.shape[0] for e in mixed_batch}
    assert len(lengths) > 1
    assert image_counts == {1, 2}


def test_collated_batch_shape_is_right_padded_to_max(qwen_processor, mixed_batch):
    batch = VQACollator(qwen_processor)(mixed_batch)
    max_len = max(e.input_ids.shape[0] for e in mixed_batch)
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        assert batch[key].shape == (len(mixed_batch), max_len)


# Group H — packed pixel_values shape and order preservation


def test_packed_pixel_values_shape(qwen_processor, mixed_batch):
    batch = VQACollator(qwen_processor)(mixed_batch)
    assert batch["pixel_values"].shape[1] == 1176
    assert batch["pixel_values"].shape[0] == sum(e.pixel_values.shape[0] for e in mixed_batch)
    assert batch["image_grid_thw"].shape == (sum(e.image_grid_thw.shape[0] for e in mixed_batch), 3)


def test_packed_pixel_values_preserve_example_order(qwen_processor, mixed_batch):
    """Row order is load-bearing (design decision 6): `get_image_features` splits the vision
    tower's output by `image_grid_thw.prod(-1) // merge_size**2` in row order, so a reordering here
    would silently pair one example's patches with another's grid."""
    batch = VQACollator(qwen_processor)(mixed_batch)
    offset = 0
    for example in mixed_batch:
        n = example.pixel_values.shape[0]
        assert torch.equal(batch["pixel_values"][offset : offset + n], example.pixel_values)
        offset += n


# Group I — pad values, presence checks, and the assistant_masks regression guard


def test_pad_values_per_key(qwen_processor, mixed_batch):
    batch = VQACollator(qwen_processor)(mixed_batch)
    max_len = batch["input_ids"].shape[1]
    for row, example in enumerate(mixed_batch):
        n = example.input_ids.shape[0]
        assert (batch["attention_mask"][row, :n] == 1).all()  # right padding: content is a prefix
        if n == max_len:
            continue
        assert (batch["input_ids"][row, n:] == qwen_processor.tokenizer.pad_token_id).all()
        assert (batch["attention_mask"][row, n:] == 0).all()
        assert (batch["labels"][row, n:] == -100).all()
        assert (batch["mm_token_type_ids"][row, n:] == 0).all()


def test_mm_token_type_ids_present_in_batch(qwen_processor, mixed_batch):
    """`mm_token_type_ids` gates `compute_3d_position_ids`'s branch selection in the installed
    `transformers==5.13.1` (`modeling_qwen2_5_vl.py`, the `else` branch sets `position_ids = None`
    with no warning when this key is absent) — dropping it silently degrades mRoPE. The collator
    must always carry it through."""
    batch = VQACollator(qwen_processor)(mixed_batch)
    assert "mm_token_type_ids" in batch
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape


def test_no_row_is_entirely_ignored(qwen_processor, mixed_batch):
    batch = VQACollator(qwen_processor)(mixed_batch)
    assert (batch["labels"] != -100).any(dim=1).all()


def test_pixel_budget_readback_matches_processor(qwen_processor):
    budget = processor_pixel_budget(qwen_processor)
    assert budget == qwen_processor.image_processor.size["longest_edge"]
    assert budget > 0


def test_assistant_tokens_mask_is_currently_broken_for_qwen_template(qwen_processor):
    """Regression guard for trap 1 (`gdp.vqa.labels`'s module docstring): Qwen2.5-VL's chat
    template has no `{% generation %}` block, so `assistant_masks` comes back all zeros with a
    warning, not an error. If HF or Qwen ever adds one, this test starts failing — forcing a
    deliberate migration of `build_answer_labels` instead of a silent divergence."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Q?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A."}]},
    ]
    out = qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    assert "assistant_masks" in out
    assert sum(out["assistant_masks"][0]) == 0


def test_seq_too_long_and_answer_boundary_drops_are_counted(
    qwen_processor, train_records, nuscenes_root
):
    ds = DriveLMVQADataset(
        train_records[:3], qwen_processor, nuscenes_root, num_views=1, max_seq_len=5
    )
    assert ds.drop_stats["seq_too_long"] == 3
    assert len(ds) == 0
