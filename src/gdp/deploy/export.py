"""ONNX export of the fine-tuned detector, frozen to the 10-class prompt.

Grounding-DINO's text branch is dynamic-vocabulary by design — arbitrary free text in, per-token
logits out. Deploying it as a fixed graph means the prompt has to stop being an input and start
being a contract: the exported ONNX artifact answers "where are these 10 BDD classes", never
"where is X" for an arbitrary X. `fixed_prompt: true` on `ExportResult` carries that limitation as
data (H3), not just prose that a later rewrite could drop.

Two real export blockers, both from HF's `generate_masks_with_special_tokens_and_transfer_map`
(called once per forward pass to build the intra-prompt attention mask from `input_ids`):
`torch.isin` and `torch.cummax`/`torch.cummin` have no legacy-tracer ONNX opset-17 mapping.
Since the prompt is frozen, that function's *output* never changes — so `_generate_masks_traceable`
below is a tensor-op-only reimplementation (equality/where/triangular-mask reductions instead of
isin/cummax), verified bit-for-bit identical to HF's version before being swapped in for the
export only, then torn down. This is not a hand-rolled model change — it is the same function,
written so a tracer can follow it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from gdp.data.core import Sample
from gdp.detect.detector import GroundingDinoDetector
from gdp.paths import resolve

INPUT_NAMES: tuple[str, ...] = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
OUTPUT_NAMES: tuple[str, ...] = ("logits", "pred_boxes")


@dataclass(frozen=True)
class ExportResult:
    """What got exported and how — carried alongside the artifact, not just printed."""

    onnx_path: Path
    export_path: Literal["dynamo", "legacy"]
    fixed_prompt: bool
    prompt: str
    opset: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


def _generate_masks_traceable(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-op-only reimplementation of HF's special-token attention-mask builder.

    Replaces `torch.isin` (test-elements membership) with a broadcast equality + `any`, and
    `torch.cummax`/`torch.cummin` with triangular-masked `max`/`min` reductions — both ONNX
    opset-17-legal, unlike the originals. Numerically verified identical to HF's function for
    this project's fixed prompt (see `tests/test_deploy_export.py`).
    """
    import transformers.models.grounding_dino.modeling_grounding_dino as gd_module

    special_tokens = torch.tensor(gd_module.SPECIAL_TOKENS, device=input_ids.device)

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    special_mask = (input_ids.unsqueeze(-1) == special_tokens.view(1, 1, -1)).any(dim=-1)
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    tri_le = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    prev_vals = torch.where(special_mask, indices, torch.full_like(indices, -1))
    prev_expanded = prev_vals.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    prev_masked = torch.where(
        tri_le.unsqueeze(0), prev_expanded, torch.full_like(prev_expanded, -1)
    )
    prev_special = prev_masked.max(dim=2).values

    tri_ge = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    next_vals = torch.where(special_mask, indices, torch.full_like(indices, seq_len))
    next_expanded = next_vals.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    next_masked = torch.where(
        tri_ge.unsqueeze(0), next_expanded, torch.full_like(next_expanded, seq_len)
    )
    next_special = next_masked.min(dim=2).values

    valid_block = (next_special != 0) & (next_special != seq_len - 1) & (next_special != seq_len)
    attention_mask = (
        next_special.unsqueeze(2) == next_special.unsqueeze(1)
    ) & valid_block.unsqueeze(1)

    idx = torch.arange(seq_len, device=device)
    identity = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).expand(batch_size, -1, -1)
    attention_mask = identity | attention_mask

    position_ids = indices - prev_special - 1
    position_ids = torch.where(valid_block, position_ids, torch.zeros_like(position_ids))
    position_ids = torch.clamp(position_ids, min=0).to(torch.long)

    return attention_mask, position_ids


@contextlib.contextmanager
def _traceable_special_token_masks():
    import transformers.models.grounding_dino.modeling_grounding_dino as gd_module

    original = gd_module.generate_masks_with_special_tokens_and_transfer_map
    gd_module.generate_masks_with_special_tokens_and_transfer_map = _generate_masks_traceable
    try:
        yield
    finally:
        gd_module.generate_masks_with_special_tokens_and_transfer_map = original


@contextlib.contextmanager
def _force_disabled_custom_kernels(model: torch.nn.Module):
    """Grounding-DINO's MS-deformable-attention custom kernels have no ONNX equivalent.

    `disable_custom_kernels` is read per-module at forward time (set from config at each
    submodule's `__init__`), not just at construction — so it can be toggled after the fact by
    walking every submodule that carries the attribute, without needing to reload the model.
    """
    touched: list[tuple[torch.nn.Module, bool]] = []
    for module in model.modules():
        if hasattr(module, "disable_custom_kernels"):
            touched.append((module, module.disable_custom_kernels))
            module.disable_custom_kernels = True
    try:
        yield
    finally:
        for module, original in touched:
            module.disable_custom_kernels = original


class _LogitsBoxesWrapper(torch.nn.Module):
    """Exports exactly `outputs.logits, outputs.pred_boxes` — the two tensors
    `assign_classes`/`convert_boxes_cxcywh_norm_to_xyxy_abs` (`gdp.detect.detector`) consume, in
    that order, so `OnnxDetector` (task 4) can feed them through the same box-decoding path."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask):
        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
        )
        return out.logits, out.pred_boxes


def export_to_onnx(
    detector: GroundingDinoDetector, sample: Sample, out_path: str | Path, *, opset: int = 17
) -> ExportResult:
    """Export `detector.model` to ONNX, frozen to `detector.prompt`, dynamic only over batch.

    Tries `dynamo=True` first (torch's current default exporter); Grounding-DINO's Swin
    backbone does not currently survive dynamo's shape-inference pass (a `view` on a
    non-contiguous window-attention tensor), so this falls back to the legacy TorchScript
    tracer — which needs `_traceable_special_token_masks` to route around `isin`/`cummax`.
    Whichever path succeeds is recorded in `ExportResult.export_path`, not silently assumed.
    """
    out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(sample.image_path).convert("RGB")
    inputs = detector.processor(images=[image], text=[detector.prompt], return_tensors="pt")
    args = tuple(inputs[name] for name in INPUT_NAMES)
    dynamic_axes = {name: {0: "batch"} for name in (*INPUT_NAMES, *OUTPUT_NAMES)}

    wrapper = _LogitsBoxesWrapper(detector.model)
    wrapper.eval()

    original_device = next(detector.model.parameters()).device
    detector.model.to("cpu")
    try:
        with _force_disabled_custom_kernels(detector.model):
            try:
                torch.onnx.export(
                    wrapper,
                    args,
                    str(out_path),
                    input_names=list(INPUT_NAMES),
                    output_names=list(OUTPUT_NAMES),
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                    dynamo=True,
                )
                export_path: Literal["dynamo", "legacy"] = "dynamo"
            except Exception:  # noqa: BLE001 — any dynamo failure falls back to the legacy tracer
                with _traceable_special_token_masks():
                    torch.onnx.export(
                        wrapper,
                        args,
                        str(out_path),
                        input_names=list(INPUT_NAMES),
                        output_names=list(OUTPUT_NAMES),
                        dynamic_axes=dynamic_axes,
                        opset_version=opset,
                        dynamo=False,
                    )
                export_path = "legacy"
    finally:
        detector.model.to(original_device)

    return ExportResult(
        onnx_path=out_path,
        export_path=export_path,
        fixed_prompt=True,
        prompt=detector.prompt,
        opset=opset,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )
