"""INT8 quantization must actually shrink the model and the result must still infer. Loads real
`grounding-dino-tiny` weights to produce a real `model.onnx` to quantize — `model_heavy`
(CLAUDE.md §4)."""

from __future__ import annotations

import pytest

from gdp.config import BDD100K_CLASSES, DetectorConfig
from gdp.data.core import load_dataset
from gdp.deploy.export import export_to_onnx
from gdp.deploy.onnx_detector import OnnxDetector
from gdp.deploy.quantize import QuantizeResult, quantize_dynamic_int8
from gdp.detect.detector import GroundingDinoDetector

pytestmark = pytest.mark.model_heavy


@pytest.fixture(scope="module")
def pt_detector():
    try:
        return GroundingDinoDetector(DetectorConfig(), BDD100K_CLASSES, device="cpu")
    except Exception as exc:  # noqa: BLE001 — unavailable weights is a skip, not a failure
        pytest.skip(f"grounding-dino-tiny unavailable (no cache, no network): {exc}")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("tests/fixtures/mini_bdd/annotations.json", "tests/fixtures/mini_bdd")


@pytest.fixture(scope="module")
def fp32_onnx_path(pt_detector, dataset, tmp_path_factory):
    out_path = tmp_path_factory.mktemp("quantize") / "model.onnx"
    export_to_onnx(pt_detector, dataset.samples[0], out_path)
    return out_path


@pytest.fixture(scope="module")
def quantize_result(fp32_onnx_path, tmp_path_factory) -> QuantizeResult:
    out_path = tmp_path_factory.mktemp("quantize_out") / "model.int8.onnx"
    return quantize_dynamic_int8(fp32_onnx_path, out_path)


def test_int8_model_is_smaller_on_disk(quantize_result):
    assert quantize_result.onnx_path.is_file()
    assert quantize_result.int8_size_bytes > 0
    assert quantize_result.int8_size_bytes < quantize_result.fp32_size_bytes


def test_int8_model_still_loads_and_infers(quantize_result, dataset):
    detector = OnnxDetector(quantize_result.onnx_path, DetectorConfig(), BDD100K_CLASSES)
    detections = detector.detect_images(dataset.samples, box_threshold=0.25)
    assert isinstance(detections, list)
