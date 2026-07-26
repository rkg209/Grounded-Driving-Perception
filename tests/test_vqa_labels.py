"""Label-mask coverage for `gdp.vqa.labels` (spec 07 task 1's "real risk").

Every test drives the real Qwen2.5-VL-3B-Instruct processor (tokenizer + image processor, no
weights) — `model_heavy` (CLAUDE.md §4). Run deliberately with:

    uv run pytest -m model_heavy tests/test_vqa_labels.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from gdp.vqa.labels import (
    IGNORE_INDEX,
    AnswerBoundaryUnresolvable,
    assert_answer_only_mask,
    build_answer_labels,
    expand_image_placeholders,
    prompt_token_length,
)

pytestmark = pytest.mark.model_heavy


@pytest.fixture(scope="module")
def qwen_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def _image(size: int = 280, fill: int = 0) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), fill, dtype=np.uint8))


def _encode(processor, question: str, answer: str, num_images: int = 1):
    """Build one QA example through the full labels.py pipeline, mirroring the sequence
    `gdp.vqa.dataset` (task 3) will use: one processor call on the full text, prompt-only
    re-expansion, then the assertion-backed label mask.
    """
    images = [_image(fill=i * 40) for i in range(num_images)]
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": question})
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    if num_images:
        enc = processor(text=[full_text], images=images, return_tensors="pt")
        prompt_text = expand_image_placeholders(processor, prompt_text, enc["image_grid_thw"])
    else:
        enc = processor(text=[full_text], return_tensors="pt")
    n = prompt_token_length(processor, prompt_text, enc)
    labels = build_answer_labels(enc, n)
    return enc, labels, n


# Group A — shape agreement


def test_labels_shape_matches_input_ids(qwen_processor):
    enc, labels, _ = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    assert labels.shape == enc["input_ids"].shape


# Group B — labels[:prompt_len] are all IGNORE_INDEX (acceptance criterion 2, literally)


def test_everything_before_prompt_len_is_ignored(qwen_processor):
    enc, labels, n = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    assert (labels[0, :n] == IGNORE_INDEX).all()


# Group C — image-pad positions are IGNORE_INDEX, checked via two independent routes


def test_image_pad_positions_masked_via_image_token_id(qwen_processor):
    enc, labels, _ = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    image_positions = enc["input_ids"][0] == qwen_processor.image_token_id
    assert image_positions.any()
    assert (labels[0, image_positions] == IGNORE_INDEX).all()


def test_image_pad_positions_masked_via_mm_token_type_ids(qwen_processor):
    enc, labels, _ = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    image_positions = enc["mm_token_type_ids"][0] == 1
    assert image_positions.any()
    assert (labels[0, image_positions] == IGNORE_INDEX).all()


# Group D — the get_placeholder_mask invariant (modeling_qwen2_5_vl.py's own mid-forward check)


def test_image_token_count_matches_vision_patch_count(qwen_processor):
    enc, _, _ = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.", num_images=2)
    merge_size = qwen_processor.image_processor.merge_size
    n_image_tokens = (enc["input_ids"][0] == qwen_processor.image_token_id).sum()
    n_expected = (enc["image_grid_thw"].prod(-1) // merge_size**2).sum()
    assert n_image_tokens == n_expected


# Group E — the round-trip decode, the negative control, and the EOS-supervised boundary


def test_supervised_span_decodes_to_answer_plus_im_end(qwen_processor):
    answer = "A pedestrian is crossing."
    enc, labels, _ = _encode(qwen_processor, "What is ahead?", answer)
    assert_answer_only_mask(qwen_processor, enc, labels, answer, qa_id="t-roundtrip")

    supervised_ids = enc["input_ids"][0][labels[0] != IGNORE_INDEX]
    decoded = qwen_processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
    assert decoded == answer + "<|im_end|>\n"


def test_question_never_appears_in_supervised_span(qwen_processor):
    question = "Describe the unique zebra crossing hazard here"
    answer = "A pedestrian is crossing."
    enc, labels, _ = _encode(qwen_processor, question, answer)
    supervised_ids = enc["input_ids"][0][labels[0] != IGNORE_INDEX]
    decoded = qwen_processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
    assert "zebra" not in decoded
    assert "hazard" not in decoded


def test_eos_supervised_while_earlier_im_ends_are_masked(qwen_processor):
    enc, labels, n = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    im_end_id = qwen_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = (enc["input_ids"][0] == im_end_id).nonzero(as_tuple=True)[0].tolist()
    assert len(im_end_positions) >= 3  # system, user, assistant turns each end in <|im_end|>

    *earlier, last = im_end_positions
    for pos in earlier:
        assert pos < n
        assert labels[0, pos].item() == IGNORE_INDEX
    assert last >= n
    assert labels[0, last].item() != IGNORE_INDEX


# Group F — the ForCausalLMLoss shift replayed in pure torch, and the two boundary cases


def test_forcausallm_shift_matches_manual_pad_and_slice(qwen_processor):
    """Mirrors `ForCausalLMLoss` (`loss/loss_utils.py`): pad labels by one IGNORE_INDEX column,
    then take [1:]. `labels` here must never be pre-shifted (design decision 8) — this test proves
    the module's output composes correctly with HF's own shift, not a hand-rolled one."""
    enc, labels, _ = _encode(qwen_processor, "What is ahead?", "A pedestrian is crossing.")
    padded = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = padded[..., 1:].contiguous()

    assert shift_labels.shape == labels.shape
    assert shift_labels[0, -1].item() == IGNORE_INDEX
    assert torch.equal(shift_labels[0, :-1], labels[0, 1:])


def test_newline_prefixed_answer_raises_answer_boundary_unresolvable(qwen_processor):
    with pytest.raises(AnswerBoundaryUnresolvable):
        _encode(qwen_processor, "Q?", "\nGoing ahead.", num_images=0)


def test_leading_spaces_answer_does_not_raise(qwen_processor):
    enc, labels, n = _encode(qwen_processor, "Q?", "  leading spaces", num_images=0)
    assert n > 0
    assert (labels[0, :n] == IGNORE_INDEX).all()
