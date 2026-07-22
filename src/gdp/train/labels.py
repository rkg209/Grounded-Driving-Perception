"""The label path: `Sample` boxes -> Grounding-DINO's training targets, asserted at every step.

Grounding-DINO has no class head (`gdp.detect.detector` module docstring). At train time this cuts
the other way from inference: `GroundingDinoForObjectDetection.forward` calls
`build_label_maps(logits, input_ids)` internally, which derives each class's token group by
splitting `input_ids` on delimiter tokens (`SPECIAL_TOKENS` + `[PAD]`) — *not* from anything we
pass in. So `class_labels[i]` must be the index of the class **in prompt order**, and the only way
to be sure our assumption matches HF's actual splitting is to call HF's own `build_label_maps` and
compare, rather than trust the two implementations agree by construction. That is
`assert_label_map_alignment` below — the training-time twin of
`detect.detector.assert_span_round_trip`.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

from gdp.data.core import Sample
from gdp.data.prompt import PromptSpans


def build_coco_target(sample: Sample) -> dict:
    """One image's boxes -> the COCO-detection dict `GroundingDinoImageProcessor` expects.

    `category_id` is `Box.class_id` directly — already the prompt-order index (spec 01: categories
    are loaded sorted by id, and `configs/default.yaml` lists classes in that same prompt order) —
    not a raw BDD100K category id needing remapping.
    """
    return {
        "image_id": sample.image_id,
        "annotations": [
            {
                "bbox": [b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0],
                "category_id": b.class_id,
                "area": (b.x1 - b.x0) * (b.y1 - b.y0),
                "iscrowd": 0,
            }
            for b in sample.boxes
        ],
    }


def assert_normalized_cxcywh(boxes: torch.Tensor) -> None:
    """Fail loudly on absolute-pixel boxes (e.g. a `640.0`) reaching the loss.

    Mirrors `detect.detector.convert_boxes_cxcywh_norm_to_xyxy_abs`'s guard: an absolute box here
    trains cleanly (falling loss, plausible-looking logs) while learning nothing, because the
    bipartite matcher's L1/GIoU terms assume `[0, 1]`-normalized coordinates.
    """
    if boxes.numel() == 0:
        return
    arr = boxes.detach().cpu()
    if arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
        raise ValueError(
            f"expected normalized cxcywh boxes in [0, 1], got range [{arr.min().item()}, "
            f"{arr.max().item()}]"
        )
    w, h = arr[..., 2], arr[..., 3]
    if (w <= 0).any() or (h <= 0).any():
        raise ValueError(f"degenerate box: w or h <= 0 in {arr[(w <= 0) | (h <= 0)].tolist()}")


def assert_label_map_alignment(
    prompt_spans: PromptSpans, tokenizer: PreTrainedTokenizerBase
) -> None:
    """Assert HF's `build_label_maps` splits `prompt_spans.input_ids` exactly how we assume.

    Calls HF's own delimiter-splitting logic on the real prompt token ids (a dummy `logits` tensor
    supplies only the shape `build_label_maps` reads) and checks, for every class, that the token
    indices HF's label map assigns to that group are exactly `prompt_spans.token_spans[class_idx]`
    — same count, same identity, same order. If HF's delimiter splitting ever disagrees (a prompt
    format change, a tokenizer special-token change, an off-by-one in `PromptSpans`), this raises
    instead of silently training every class against the wrong tokens.
    """
    input_ids = torch.tensor([list(prompt_spans.input_ids)])
    seq_len = input_ids.shape[1]
    dummy_logits = torch.zeros(1, 1, seq_len)
    (label_map,) = build_label_maps(dummy_logits, input_ids)

    if label_map.shape[0] != len(prompt_spans.classes):
        raise RuntimeError(
            f"HF build_label_maps found {label_map.shape[0]} label groups in prompt "
            f"{prompt_spans.prompt!r}, expected {len(prompt_spans.classes)} "
            f"(one per class: {prompt_spans.classes})"
        )

    for class_idx, name in enumerate(prompt_spans.classes):
        start, end = prompt_spans.token_spans[class_idx]
        expected = set(range(start, end))
        actual = set(torch.nonzero(label_map[class_idx], as_tuple=False).flatten().tolist())
        if actual != expected:
            raise RuntimeError(
                f"class {name!r} (index {class_idx}): HF build_label_maps assigned tokens "
                f"{sorted(actual)}, but PromptSpans.token_spans says {sorted(expected)}. "
                "class_labels would be trained against the wrong text tokens."
            )
