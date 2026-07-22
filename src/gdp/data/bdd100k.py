"""Raw BDD100K `det_20` JSON -> the project's COCO-style schema.

`box2d` is absolute xyxy; our schema is absolute xywh (spec-00's contract, `Box`/`load_dataset`).
Category names map to `BDD100K_CLASSES` **by index** — that order is load-bearing, and the emitted
`categories[].id` must be `0..9` to match the fixture schema exactly.

Every dropped box is counted **by reason**, never silently swallowed (CLAUDE.md H7): a class-count
report built on top of a converter that silently drops boxes would flatter later mAP without anyone
noticing. Boxes that merely straddle the image edge are *clipped* and counted as clipped, not
dropped — clipping loses less ground truth than dropping outright.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from gdp.config import BDD100K_CLASSES

# BDD100K's 100k split images are uniformly this size. Real-data callers should verify this against
# a sample of actual files (`verify_image_size`) rather than trusting the assumption blindly.
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720

DROP_REASONS = ("unknown_category", "degenerate", "out_of_frame", "no_box2d")


@dataclass
class ConversionStats:
    """Drop/clip accounting for one `convert_bdd_to_coco` call. Dataset statistics, not a metric
    (H9) — never present this as a result."""

    images: int = 0
    images_without_boxes: int = 0
    boxes: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: {r: 0 for r in DROP_REASONS})
    clipped: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "images_without_boxes": self.images_without_boxes,
            "boxes": self.boxes,
            "dropped": dict(self.dropped),
            "clipped": self.clipped,
        }


def convert_bdd_to_coco(
    frames: list[dict[str, Any]],
    *,
    classes: tuple[str, ...] = BDD100K_CLASSES,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
) -> tuple[dict[str, Any], ConversionStats]:
    """Convert raw BDD `det_20` frames into our COCO-style annotations dict.

    `frames` is the raw `det_20` JSON: a list of `{"name": ..., "labels": [...] | None}`.
    Deterministic: `image_id` is the index into `sorted(names)`; `annotation_id` is a running
    counter over surviving boxes in frame order. Same input -> same output, always.
    """
    name_to_class_id = {name: i for i, name in enumerate(classes)}
    stats = ConversionStats()

    sorted_names = sorted(f["name"] for f in frames)
    name_to_image_id = {name: i for i, name in enumerate(sorted_names)}
    frames_by_name = {f["name"]: f for f in frames}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0

    for name in sorted_names:
        frame = frames_by_name[name]
        image_id = name_to_image_id[name]
        images.append(
            {"id": image_id, "file_name": name, "width": image_width, "height": image_height}
        )
        stats.images += 1

        labels = frame.get("labels") or []
        surviving = 0
        for label in labels:
            category = label.get("category")
            if category not in name_to_class_id:
                stats.dropped["unknown_category"] += 1
                continue

            box2d = label.get("box2d")
            if not box2d:
                stats.dropped["no_box2d"] += 1
                continue

            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            if x2 <= x1 or y2 <= y1:
                stats.dropped["degenerate"] += 1
                continue
            if x2 <= 0 or x1 >= image_width or y2 <= 0 or y1 >= image_height:
                stats.dropped["out_of_frame"] += 1
                continue

            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(image_width), x2), min(float(image_height), y2)
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                stats.clipped += 1

            w, h = cx2 - cx1, cy2 - cy1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": name_to_class_id[category],
                    "bbox": [cx1, cy1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            surviving += 1
            stats.boxes += 1

        # A frame with zero surviving boxes keeps its image entry — dropping it would silently
        # shrink the val split that every later mAP is computed on (H8).
        if surviving == 0:
            stats.images_without_boxes += 1

    categories = [{"id": i, "name": n} for i, n in enumerate(classes)]
    coco = {
        "info": {
            "description": "BDD100K det_20, converted to COCO-style schema.",
            "classes_follow": "BDD100K 10-class detection taxonomy",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return coco, stats


def verify_image_size(
    images_root: str | Path,
    file_names: list[str],
    *,
    expected_width: int = DEFAULT_IMAGE_WIDTH,
    expected_height: int = DEFAULT_IMAGE_HEIGHT,
    sample_n: int = 20,
) -> None:
    """Check a sample of real images against the assumed uniform size, erroring on mismatch.

    Never trust "BDD100K is uniformly 1280x720" blindly — an unexpected class or a truncated
    download shows up here before it shows up as a silently-wrong mAP downstream.
    """
    root = Path(images_root)
    sample = random.Random(0).sample(file_names, k=min(sample_n, len(file_names)))
    for name in sample:
        path = root / name
        with Image.open(path) as im:
            if im.size != (expected_width, expected_height):
                raise ValueError(
                    f"{path}: expected size ({expected_width}, {expected_height}), got {im.size}"
                )
