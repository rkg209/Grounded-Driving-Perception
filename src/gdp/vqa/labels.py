"""The label path: one rendered chat turn -> Qwen2.5-VL's answer-only loss mask.

Verified against the installed `transformers==5.13.1`. Three independent traps, all load-bearing:

1. **`apply_chat_template(..., return_assistant_tokens_mask=True)` silently produces NaN.** It
   requires a `{% generation %}` block in the chat template. Qwen2.5-VL's template has none, so HF
   logs `logger.warning_once("...chat template does not contain `{% generation %}` keyword.")`
   (`chat_template_utils.py`) instead of raising — `assistant_masks` comes back **all zeros**, every
   label becomes `-100`, and cross-entropy over zero valid targets is NaN. `build_answer_labels`
   below never uses this path; `tests/test_vqa_labels.py`'s test I-25 pins that it is still broken.
2. **The prompt/answer token boundary can genuinely not exist.** Qwen2's BPE pre-tokenizer merges
   consecutive newlines, so an answer starting with `\\n` fuses with the chat template's trailing
   `\\n` (`Ċ` id 198 -> `ĊĊ` id 271) and the prompt is no longer a token-id prefix of the full
   sequence. `prompt_token_length` asserts the prefix match explicitly instead of assuming it, and
   raises `AnswerBoundaryUnresolvable` when it fails — the caller (`gdp.vqa.dataset`) drops the
   record with a counted reason (H7) rather than silently stripping the answer, which would edit
   ground truth (H2, DriveLM answers are byte-verbatim by policy).
3. **`(input_ids == image_token_id).sum() == (image_grid_thw.prod(-1) // merge_size**2).sum()`** is
   exactly the invariant `Qwen2_5_VLForConditionalGeneration.get_placeholder_mask`
   (`modeling_qwen2_5_vl.py:1113-1143`) enforces mid-forward, by summing `special_image_mask` and
   comparing to the vision tower's feature count. `expand_image_placeholders` reconstructs the same
   per-image token counts so the prompt-only re-tokenization in trap 2's assertion lines up; getting
   the count wrong here would surface as this exact assertion failing on the collator, not 40
   minutes into a cluster run.
"""

from __future__ import annotations

from typing import Any

import torch

IGNORE_INDEX = -100


class AnswerBoundaryUnresolvable(ValueError):
    """Raised when the prompt text is not an exact token-id prefix of the full rendered sequence
    (trap 2). The caller counts and drops the record; it must never strip or re-tokenize the
    answer to force a match — that would edit ground truth (H2)."""


def expand_image_placeholders(processor: Any, text: str, image_grid_thw: torch.Tensor) -> str:
    """Replace each occurrence of `processor.image_token` in `text` with the number of copies the
    processor's own `__call__` would expand it to for that image (`processing_utils.py`'s
    `replace_image_token`: `image_grid_thw[i].prod() // merge_size**2`).

    `text` must contain exactly one placeholder per row of `image_grid_thw`, in the same order —
    true of a chat-template-rendered prompt before it has been through the processor, since the
    template emits one `<|image_pad|>` per image regardless of resolution.
    """
    merge_size = processor.image_processor.merge_size
    counts = (image_grid_thw.prod(dim=-1) // merge_size**2).tolist()
    parts = text.split(processor.image_token)
    if len(parts) - 1 != len(counts):
        raise ValueError(
            f"expected {len(counts)} occurrences of {processor.image_token!r} in text, "
            f"found {len(parts) - 1}"
        )
    expanded = parts[0]
    for count, tail in zip(counts, parts[1:], strict=True):
        expanded += processor.image_token * count + tail
    return expanded


def prompt_token_length(processor: Any, prompt_text: str, encoded: dict[str, torch.Tensor]) -> int:
    """Tokenize `prompt_text` independently (the tokenizer, not the processor - no images to
    re-encode) and assert it is an exact token-id prefix of `encoded["input_ids"][0]`.

    Returns the prefix length `n`; `labels[:n]` must be masked. Raises
    `AnswerBoundaryUnresolvable` if the prefix does not match byte-for-byte in token-id space
    (trap 2) — the load-bearing check in this module, per the design note it does not assume the
    boundary exists, it proves it.
    """
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    full_ids = encoded["input_ids"][0].tolist()
    if full_ids[:n] != prompt_ids:
        raise AnswerBoundaryUnresolvable(
            "prompt is not a token-id prefix of the full sequence (Qwen2 BPE merged a boundary "
            "token, e.g. consecutive newlines) — cannot locate the answer-only span"
        )
    return n


def build_answer_labels(encoded: dict[str, torch.Tensor], prompt_len: int) -> torch.Tensor:
    """`labels` = `input_ids` with everything before `prompt_len` set to `IGNORE_INDEX`.

    Unshifted and 1:1 with `input_ids` (design decision 8): `ForCausalLMLoss` does its own
    pad+shift, so pre-shifting here would double-shift. `<|im_end|>` at the end of the assistant
    turn falls inside the unmasked region because it is part of `encoded["input_ids"]` past
    `prompt_len` — required, or the model never learns to stop generating.
    """
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_len] = IGNORE_INDEX
    return labels


def assert_answer_only_mask(
    processor: Any,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    answer: str,
    qa_id: str,
) -> None:
    """Decode the supervised span (`labels != IGNORE_INDEX`) and assert it equals exactly the
    answer text plus the chat template's trailing `<|im_end|>\\n` — nothing from the question,
    nothing from the image tokens, nothing truncated off the end.

    This is the round-trip that would catch a `build_answer_labels` regression even if
    `prompt_token_length`'s prefix check happened to still pass (e.g. an off-by-one that shifts
    the mask but keeps the same length).
    """
    input_ids = encoded["input_ids"][0]
    row_labels = labels[0]
    supervised_ids = input_ids[row_labels != IGNORE_INDEX]
    decoded = processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
    expected = answer + "<|im_end|>\n"
    if decoded != expected:
        raise RuntimeError(
            f"qa_id={qa_id}: supervised span decoded to {decoded!r}, expected {expected!r} — "
            "the answer-only label mask is wrong"
        )


def processor_pixel_budget(processor: Any) -> int:
    """The processor's actual per-image pixel cap, read back off `image_processor.size` rather
    than asserted from config (design decision 10) — reflects `max_pixels`/`min_pixels` as folded
    into `size` at `AutoProcessor.from_pretrained(...)` construction time, whatever was actually
    passed. The shipped default is ~12.8 MP, i.e. up to 16384 image tokens for one image at
    `merge_size=2`; this number belongs in every run's `train_config.json` (acceptance criterion 5).
    """
    return int(processor.image_processor.size["longest_edge"])
