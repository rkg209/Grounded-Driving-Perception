"""mAP re-evaluation wiring for all three variants (fp32-PyTorch, fp32-ONNX, INT8-ONNX) against
the fixture — three `predictions.json`s, three mAP numbers, via the identical `evaluate_coco_map`
call specs 02-04 use. Loads real `grounding-dino-tiny` weights — `model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

import pytest

from gdp.config import BDD100K_CLASSES, DetectorConfig
from gdp.data.core import load_dataset
from gdp.deploy.evaluate import evaluate_variant
from gdp.deploy.export import export_to_onnx
from gdp.deploy.onnx_detector import OnnxDetector
from gdp.deploy.quantize import quantize_dynamic_int8
from gdp.detect.detector import GroundingDinoDetector

pytestmark = pytest.mark.model_heavy

GT_PATH = "tests/fixtures/mini_bdd/annotations.json"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(GT_PATH, "tests/fixtures/mini_bdd")


@pytest.fixture(scope="module")
def variants(dataset, tmp_path_factory):
    try:
        pt_detector = GroundingDinoDetector(DetectorConfig(), BDD100K_CLASSES, device="cpu")
    except Exception as exc:  # noqa: BLE001 — unavailable weights is a skip, not a failure
        pytest.skip(f"grounding-dino-tiny unavailable (no cache, no network): {exc}")

    onnx_dir = tmp_path_factory.mktemp("evaluate_variants")
    export_result = export_to_onnx(pt_detector, dataset.samples[0], onnx_dir / "model.onnx")
    quantize_result = quantize_dynamic_int8(export_result.onnx_path, onnx_dir / "model.int8.onnx")

    fp32_onnx_detector = OnnxDetector(export_result.onnx_path, DetectorConfig(), BDD100K_CLASSES)
    int8_onnx_detector = OnnxDetector(quantize_result.onnx_path, DetectorConfig(), BDD100K_CLASSES)

    return {
        "fp32-pt": pt_detector,
        "fp32-onnx": fp32_onnx_detector,
        "int8-onnx": int8_onnx_detector,
    }


def test_evaluate_variant_produces_map_and_predictions_for_all_three(dataset, variants, tmp_path):
    box_threshold = DetectorConfig().box_threshold
    results = {}
    for name, detector in variants.items():
        results[name] = evaluate_variant(
            detector,
            dataset.samples,
            gt_path=GT_PATH,
            box_threshold=box_threshold,
            out_dir=tmp_path,
            variant=name,
        )

    assert set(results) == {"fp32-pt", "fp32-onnx", "int8-onnx"}
    for name, result in results.items():
        predictions_path = tmp_path / f"predictions_{name}.json"
        assert predictions_path.is_file()
        assert isinstance(result.map, float)
        assert isinstance(result.map50, float)
        assert isinstance(result.per_class_ap, dict)
