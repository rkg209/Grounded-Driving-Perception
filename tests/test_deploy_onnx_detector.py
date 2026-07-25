"""`OnnxDetector` must produce the same `Detection`s as `GroundingDinoDetector` on the fixture —
proof that reusing `assign_classes`/box conversion (not a second decoding path) actually works
end to end. Loads real `grounding-dino-tiny` weights — `model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

import pytest

from gdp.config import BDD100K_CLASSES, DetectorConfig
from gdp.data.core import load_dataset
from gdp.deploy.export import export_to_onnx
from gdp.deploy.onnx_detector import OnnxDetector
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
def onnx_detector(pt_detector, dataset, tmp_path_factory) -> OnnxDetector:
    out_path = tmp_path_factory.mktemp("onnx_detector") / "model.onnx"
    export_to_onnx(pt_detector, dataset.samples[0], out_path)
    return OnnxDetector(out_path, DetectorConfig(), BDD100K_CLASSES)


def test_onnx_detector_matches_pytorch_detector_on_fixture(pt_detector, onnx_detector, dataset):
    box_threshold = DetectorConfig().box_threshold
    pt_detections = pt_detector.detect_images(dataset.samples, box_threshold=box_threshold)
    onnx_detections = onnx_detector.detect_images(dataset.samples, box_threshold=box_threshold)

    assert len(pt_detections) == len(onnx_detections)

    def sort_key(d):
        return (d.image_id, d.class_id, round(d.score, 4))

    pt_sorted = sorted(pt_detections, key=sort_key)
    onnx_sorted = sorted(onnx_detections, key=sort_key)

    for pt_det, onnx_det in zip(pt_sorted, onnx_sorted, strict=True):
        assert pt_det.image_id == onnx_det.image_id
        assert pt_det.class_id == onnx_det.class_id
        assert pt_det.score == pytest.approx(onnx_det.score, abs=1e-2)
        for pt_coord, onnx_coord in zip(pt_det.xyxy, onnx_det.xyxy, strict=True):
            assert pt_coord == pytest.approx(onnx_coord, abs=2.0)
