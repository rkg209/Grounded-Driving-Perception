"""mAP re-evaluation, applied identically to all three detector variants.

`evaluate_variant` is the one place spec 05 touches accuracy — and it touches it by calling the
same `Detection.to_coco()` / `write_predictions` / `evaluate_coco_map` path specs 02–04 already
use, so "mAP" means the identical computation for fp32-PyTorch, fp32-ONNX, and INT8-ONNX. A
second, ONNX-specific evaluation path would make the accuracy-vs-latency curve (task 7) compare
apples to oranges — exactly what H9 exists to block.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from gdp.data.core import Sample
from gdp.detect.predictions import Detection, write_predictions
from gdp.eval.coco_map import CocoMapResult, evaluate_coco_map


class Detector(Protocol):
    """What `GroundingDinoDetector` and `OnnxDetector` both are — duck-typed, not a shared base
    class, since they intentionally have no other coupling."""

    def detect_images(
        self, samples: Sequence[Sample], *, box_threshold: float, batch_size: int = 8
    ) -> list[Detection]: ...


def evaluate_variant(
    detector: Detector,
    samples: Sequence[Sample],
    *,
    gt_path: str | Path,
    box_threshold: float,
    out_dir: str | Path,
    variant: str,
) -> CocoMapResult:
    """Run `detector` over `samples`, write its predictions, and score them against `gt_path`.

    `image_ids=[s.image_id for s in samples]` restricts `evaluate_coco_map` to exactly the images
    this variant ran on, the same discipline `gdp detect`/`gdp evaluate` use for `--limit`ed runs.
    """
    detections = detector.detect_images(samples, box_threshold=box_threshold)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"predictions_{variant}.json"
    write_predictions(detections, predictions_path)

    image_ids = [s.image_id for s in samples]
    return evaluate_coco_map(gt_path, predictions_path, image_ids=image_ids)
