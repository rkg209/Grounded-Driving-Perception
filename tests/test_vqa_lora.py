"""Coverage for `gdp.vqa.lora` (spec 07 task 5): the anchored LoRA target regex and the frozen-tower
assertion (trap 4). Uses a few-layer, randomly-initialized `Qwen2_5_VLForConditionalGeneration` —
real architecture, real LoRA attach, megabytes not gigabytes (design decision 11). Marked
`model_heavy` because it constructs a real (if shrunken) HF model class (CLAUDE.md §4). Run
deliberately with:

    uv run pytest -m model_heavy tests/test_vqa_lora.py
"""

from __future__ import annotations

import pytest

from gdp.config import Config
from gdp.vqa.lora import (
    LM_TARGET_RE,
    assert_vision_tower_frozen,
    attach_lora,
    trainable_parameter_summary,
)

pytestmark = pytest.mark.model_heavy


def make_tiny_qwen_vl():
    """A few-layer Qwen2.5-VL with the real special-token ids, real module names, real forward
    pass — but ~10M parameters instead of ~3B. `vocab_size` is kept at the real tokenizer's size
    because the special image/video/vision-boundary token ids (~151656) must be valid embedding
    indices; everything else is shrunk."""
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

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
        # mRoPE requires mrope_section to sum to head_dim // 2 (hidden_size / num_attention_heads
        # / 2 = 32 / 4 / 2 = 4 here) — proportionally shrunk from the real config's [16, 24, 24]
        # (sum 64, head_dim 128 // 2), or `Qwen2_5_VLAttention.forward` raises `KeyError` looking
        # it up.
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
    return Qwen2_5_VLForConditionalGeneration(config)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config()


def test_attach_lora_reports_under_two_percent_trainable(cfg):
    model = attach_lora(make_tiny_qwen_vl(), cfg)
    summary = trainable_parameter_summary(model)
    assert summary["trainable_pct"] < 2.0
    assert summary["trainable_params"] > 0


def test_attach_lora_leaves_vision_tower_frozen(cfg):
    model = attach_lora(make_tiny_qwen_vl(), cfg)
    assert_vision_tower_frozen(model)  # must not raise


def test_lm_target_regex_matches_language_model_projections_only():
    """PEFT matches `target_modules` against `named_modules()` keys (no `.weight`/`.bias` suffix),
    not `named_parameters()` — this test checks the regex against what PEFT actually sees."""
    import re

    names = [n for n, _ in make_tiny_qwen_vl().named_modules()]
    matched = [n for n in names if re.fullmatch(LM_TARGET_RE, n)]
    assert matched  # something matched
    assert all(".visual." not in n for n in matched)
    assert all(n.startswith("model.language_model.") for n in matched)


def test_bare_short_name_target_modules_would_leak_into_vision_tower():
    """Proves trap 4 is real: a bare `target_modules=["gate_proj"]` list, via PEFT's own
    `endswith`-based matching, attaches LoRA under `model.visual.` too.
    `assert_vision_tower_frozen` must raise when that happens — the check that protects
    `attach_lora`'s real regex path."""
    from peft import LoraConfig, get_peft_model

    model = make_tiny_qwen_vl()
    bare_config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["gate_proj"], bias="none", task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, bare_config)
    with pytest.raises(RuntimeError, match="vision tower"):
        assert_vision_tower_frozen(peft_model)
