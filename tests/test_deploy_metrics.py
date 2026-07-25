"""`build_deploy_metrics` stamps `fixed_prompt`/`export_path`/`is_synthetic`, and refuses to
build a record missing any of the three variants (H8: no 2-of-3 ship). No model needed —
everything here is fed fake, typed results."""

from __future__ import annotations

from pathlib import Path

import pytest

from gdp.config import Config
from gdp.deploy.bench import BenchResult
from gdp.deploy.export import ExportResult
from gdp.deploy.metrics import (
    REQUIRED_VARIANTS,
    build_deploy_metrics,
    write_deploy_metrics,
    write_latency,
)
from gdp.eval.coco_map import CocoMapResult


def _export_result() -> ExportResult:
    return ExportResult(
        onnx_path=Path("model.onnx"),
        export_path="legacy",
        fixed_prompt=True,
        prompt="car. pedestrian.",
        opset=17,
        input_names=("pixel_values", "input_ids", "token_type_ids", "attention_mask", "pixel_mask"),
        output_names=("logits", "pred_boxes"),
    )


def _coco_result(map_value: float) -> CocoMapResult:
    return CocoMapResult(map=map_value, map50=map_value + 0.1, per_class_ap={"car": map_value})


def test_build_deploy_metrics_stamps_fixed_prompt_and_export_path():
    variant_results = {name: _coco_result(0.4) for name in REQUIRED_VARIANTS}

    metrics = build_deploy_metrics(
        cfg=Config(),
        export_result=_export_result(),
        quantization="dynamic",
        variant_results=variant_results,
        dataset="mini_bdd",
        num_images=4,
        is_synthetic=True,
    )

    assert metrics["fixed_prompt"] is True
    assert metrics["export_path"] == "legacy"
    assert metrics["is_synthetic"] is True
    assert metrics["quantization"] == "dynamic"
    assert set(metrics["variants"]) == set(REQUIRED_VARIANTS)
    for name in REQUIRED_VARIANTS:
        assert metrics["variants"][name]["map"] == 0.4


def test_build_deploy_metrics_rejects_missing_variant():
    incomplete = {name: _coco_result(0.4) for name in list(REQUIRED_VARIANTS)[:2]}

    with pytest.raises(ValueError, match="requires mAP for all of"):
        build_deploy_metrics(
            cfg=Config(),
            export_result=_export_result(),
            quantization="dynamic",
            variant_results=incomplete,
            dataset="mini_bdd",
            num_images=4,
            is_synthetic=True,
        )


def test_write_deploy_metrics_writes_json(tmp_path):
    metrics = build_deploy_metrics(
        cfg=Config(),
        export_result=_export_result(),
        quantization="dynamic",
        variant_results={name: _coco_result(0.4) for name in REQUIRED_VARIANTS},
        dataset="mini_bdd",
        num_images=4,
        is_synthetic=True,
    )
    path = write_deploy_metrics(metrics, out_dir=tmp_path)
    assert path == tmp_path / "metrics.json"
    assert path.is_file()


def test_write_latency_writes_json(tmp_path):
    variant_results = {
        "fp32-pt": BenchResult(
            variant="fp32-pt",
            p50=0.1,
            p95=0.12,
            p99=0.15,
            fps=10.0,
            peak_rss_bytes=1_000_000,
            warmup_iters=50,
            timed_iters=200,
            interleaved=True,
        )
    }
    path = write_latency(variant_results, hardware={"processor": "arm"}, out_dir=tmp_path)
    assert path == tmp_path / "latency.json"
    assert path.is_file()
