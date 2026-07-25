"""Assembling `metrics.json` and `latency.json` — spec 05's provenance record.

Mirrors `gdp.eval.metrics_io.build_metrics`'s shape (config, git_sha, created, `is_synthetic`)
but kept as a separate file from `latency.json` (design decision 10): an accuracy record and a
speed record answer different questions, and a future reader may want one without the other.

`build_deploy_metrics` refuses to build a metrics record missing any of the three variants
(H8) — a latency win with an incomplete accuracy picture beside it is exactly the half-truth H8
exists to block, so this is enforced here rather than left to the caller to remember.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.config import Config
from gdp.deploy.bench import BenchResult
from gdp.deploy.export import ExportResult
from gdp.eval.coco_map import CocoMapResult
from gdp.paths import git_sha

REQUIRED_VARIANTS: tuple[str, ...] = ("fp32-pt", "fp32-onnx", "int8-onnx")


def build_deploy_metrics(
    *,
    cfg: Config,
    export_result: ExportResult,
    quantization: str,
    variant_results: dict[str, CocoMapResult],
    dataset: str,
    num_images: int,
    is_synthetic: bool,
) -> dict[str, Any]:
    missing = set(REQUIRED_VARIANTS) - set(variant_results)
    if missing:
        raise ValueError(
            f"metrics.json requires mAP for all of {REQUIRED_VARIANTS}; missing {sorted(missing)}"
        )

    return {
        "dataset": dataset,
        "num_images": num_images,
        "is_synthetic": is_synthetic,
        "fixed_prompt": export_result.fixed_prompt,
        "export_path": export_result.export_path,
        "quantization": quantization,
        "prompt": export_result.prompt,
        "variants": {
            name: {
                "map": result.map,
                "map50": result.map50,
                "per_class_ap": result.per_class_ap,
            }
            for name, result in variant_results.items()
        },
        "config": asdict(cfg),
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_deploy_metrics(metrics: dict[str, Any], *, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def write_latency(
    variant_results: dict[str, BenchResult], *, hardware: dict[str, Any], out_dir: str | Path
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latency.json"
    payload = {
        "hardware": hardware,
        "variants": {name: asdict(result) for name, result in variant_results.items()},
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
