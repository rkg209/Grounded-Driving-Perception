"""Assembling `metrics.json` — a metric without provenance is a rumour (CLAUDE.md §8).

Every number this writes carries the split, image count, checkpoint id, the resolved config, the
operating threshold and how it was chosen ("chosen_on"), the git SHA, and whether the run was
against the synthetic fixture (`is_synthetic` — H7: a fixture metric is a smoke-test artifact,
never a reported result).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.config import Config
from gdp.eval.coco_map import CocoMapResult
from gdp.eval.operating_point import PrecisionRecall
from gdp.paths import git_sha


def build_metrics(
    *,
    coco_result: CocoMapResult,
    per_class_pr: dict[str, PrecisionRecall],
    overall_pr: PrecisionRecall,
    box_threshold: float,
    chosen_on: str,
    cfg: Config,
    dataset: str,
    split: str,
    num_images: int,
    is_synthetic: bool,
) -> dict[str, Any]:
    """Assemble the full provenance record. `chosen_on` is `"train"` once task 6's sweep picks
    the threshold, `"config_default"` otherwise — recorded either way (spec 02 acceptance 4)."""
    return {
        "dataset": dataset,
        "split": split,
        "num_images": num_images,
        "is_synthetic": is_synthetic,
        "model_id": cfg.detector.model_id,
        "map": coco_result.map,
        "map50": coco_result.map50,
        "per_class_ap": coco_result.per_class_ap,
        "box_threshold": box_threshold,
        "chosen_on": chosen_on,
        "precision": overall_pr.precision,
        "recall": overall_pr.recall,
        "tp": overall_pr.tp,
        "fp": overall_pr.fp,
        "fn": overall_pr.fn,
        "per_class_precision_recall": {name: asdict(pr) for name, pr in per_class_pr.items()},
        "config": asdict(cfg),
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_metrics(metrics: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path
