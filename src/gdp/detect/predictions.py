"""The detector's output shape: a single `Detection` and its COCO-detection serialization.

Boxes here are always **absolute xyxy** (CLAUDE.md §8, detection-eval skill §2) — the detector
converts from Grounding-DINO's normalized cxcywh at the model boundary, not here, so every
`Detection` downstream of that boundary is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One predicted box: absolute xyxy pixel coordinates, a class index, and a score."""

    image_id: int
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate box {self.xyxy}: x1/y1 must exceed x0/y0")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    def to_coco(self) -> dict[str, Any]:
        """COCO-detection-result format: xywh, not xyxy — pycocotools' convention."""
        x0, y0, x1, y1 = self.xyxy
        return {
            "image_id": self.image_id,
            "category_id": self.class_id,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "score": self.score,
        }


def write_predictions(detections: list[Detection], path: Any) -> None:
    """Serialize detections to a COCO-detection-results JSON list at `path`."""
    import json

    path.write_text(json.dumps([d.to_coco() for d in detections], indent=2) + "\n")


PREDICTIONS_META_SUFFIX = "_meta.json"


def meta_path_for(predictions_path: Any) -> Any:
    """`predictions.json` -> `predictions_meta.json`, its sidecar."""
    return predictions_path.with_name(predictions_path.stem + PREDICTIONS_META_SUFFIX)


def write_predictions_meta(image_ids: list[int], path: Any) -> Any:
    """Record which images `gdp detect` actually ran on, beside its predictions.

    `predictions.json` is COCO-detection-results format, which has no room for this: an image the
    detector processed but found nothing in is byte-identical, in that file, to an image the
    detector never opened. `gdp evaluate` must tell them apart — the first is a set of false
    negatives, the second is not evaluable at all. Without this sidecar a `--limit`ed detect run
    scored against the full split silently reports mAP over thousands of never-processed images
    (every one of them a phantom miss), and `--sweep`'s F1 argmax is dragged toward lower
    thresholds by the same phantom recall penalty. That is a corrupted H8 baseline, produced
    without a single error message.
    """
    import json

    meta = {"image_ids": sorted(image_ids), "num_images": len(image_ids)}
    meta_path = meta_path_for(path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def read_evaluated_image_ids(predictions_path: Any) -> list[int] | None:
    """The image ids `gdp detect` processed, or `None` if there is no sidecar.

    `None` means "unknown provenance" — a hand-written or third-party predictions file. The caller
    decides what to do about it; this function never guesses an image set from the predictions
    themselves, which would silently drop every true negative image.
    """
    import json

    meta_path = meta_path_for(predictions_path)
    if not meta_path.is_file():
        return None
    return list(json.loads(meta_path.read_text())["image_ids"])
