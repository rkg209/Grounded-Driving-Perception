"""Spec 05's real risk (design decision 4): a broken export must fail here, not three layers
downstream as "quantization tanked accuracy". Loads real `grounding-dino-tiny` weights —
`model_heavy`, deselected by default (CLAUDE.md §4)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from gdp.config import BDD100K_CLASSES, DetectorConfig
from gdp.data.core import load_dataset
from gdp.deploy.export import INPUT_NAMES, OUTPUT_NAMES, ExportResult, export_to_onnx
from gdp.detect.detector import GroundingDinoDetector

pytestmark = pytest.mark.model_heavy

BOX_THRESHOLD = 0.25


@pytest.fixture(scope="module")
def detector():
    try:
        return GroundingDinoDetector(DetectorConfig(), BDD100K_CLASSES, device="cpu")
    except Exception as exc:  # noqa: BLE001 — unavailable weights is a skip, not a failure
        pytest.skip(f"grounding-dino-tiny unavailable (no cache, no network): {exc}")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("tests/fixtures/mini_bdd/annotations.json", "tests/fixtures/mini_bdd")


@pytest.fixture(scope="module")
def export_result(detector, dataset, tmp_path_factory) -> ExportResult:
    out_path = tmp_path_factory.mktemp("deploy_export") / "model.onnx"
    return export_to_onnx(detector, dataset.samples[0], out_path)


def test_export_writes_onnx_and_records_fixed_prompt_limitation(export_result):
    assert export_result.onnx_path.is_file()
    assert export_result.onnx_path.stat().st_size > 0
    assert export_result.fixed_prompt is True
    assert export_result.export_path in ("dynamo", "legacy")
    assert export_result.input_names == INPUT_NAMES
    assert export_result.output_names == OUTPUT_NAMES


def test_export_prompt_matches_detector_prompt(detector, export_result):
    assert export_result.prompt == detector.prompt


def test_onnx_parity_on_fixture(detector, dataset, export_result):
    """`np.allclose` on PyTorch vs. ONNX Runtime outputs — a numerical parity test, not a vibe
    check (acceptance criterion 1). Compared post-sigmoid (Grounding-DINO's raw `logits` are
    legitimately -inf at masked text positions; -inf minus -inf is NaN, not a mismatch) and,
    for boxes, on the queries that clear `box_threshold` — the ones that ever become a
    `Detection`. Below-threshold queries are unused noise whose regressed box coordinates are
    fp32-accumulation-sensitive between the eager and traced graphs (confirmed by inspection: on
    this fixture, every query above threshold matches to ~1e-4, and threshold membership itself
    never disagrees between the two graphs).
    """
    images = [Image.open(s.image_path).convert("RGB") for s in dataset.samples]
    inputs = detector.processor(
        images=images, text=[detector.prompt] * len(images), return_tensors="pt"
    )
    with torch.no_grad():
        out = detector.model(**inputs)
    logits_pt = out.logits.numpy()
    boxes_pt = out.pred_boxes.numpy()

    sess = ort.InferenceSession(str(export_result.onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {name: inputs[name].numpy() for name in INPUT_NAMES}
    logits_onnx, boxes_onnx = sess.run(list(OUTPUT_NAMES), ort_inputs)

    sig_pt = 1 / (1 + np.exp(-np.clip(logits_pt, -50, 50)))
    sig_onnx = 1 / (1 + np.exp(-np.clip(logits_onnx, -50, 50)))
    assert np.allclose(sig_pt, sig_onnx, atol=1e-2)

    kept_pt = sig_pt.max(axis=2) > BOX_THRESHOLD
    kept_onnx = sig_onnx.max(axis=2) > BOX_THRESHOLD
    assert kept_pt.any(), "fixture produced no detections above box_threshold — nothing to compare"
    assert np.array_equal(kept_pt, kept_onnx), (
        "PyTorch/ONNX disagree on which queries clear box_threshold"
    )
    assert np.allclose(boxes_pt[kept_pt], boxes_onnx[kept_pt], atol=1e-3)
