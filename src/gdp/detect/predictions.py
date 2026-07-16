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
