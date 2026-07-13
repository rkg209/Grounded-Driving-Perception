"""Minimal COCO-style annotation loading.

Spec 00 ships only what the synthetic fixture needs: enough to prove the data contract end-to-end
on the laptop with zero downloads. Spec 01 replaces the *source* (real BDD100K) but must keep this
contract, so every later spec stays testable offline (CLAUDE.md §4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gdp.paths import resolve


@dataclass(frozen=True)
class Box:
    """One annotated object. Coordinates are absolute pixels, xyxy."""

    x0: float
    y0: float
    x1: float
    y1: float
    class_id: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"degenerate box: {self}")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class Sample:
    """One image plus its ground-truth boxes."""

    image_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class Dataset:
    classes: tuple[str, ...]
    samples: tuple[Sample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def prompt(self) -> str:
        """The fixed class prompt Grounding-DINO is queried with: 'car. pedestrian. ...'"""
        return ". ".join(self.classes) + "."


def load_dataset(annotations: str | Path, images_root: str | Path) -> Dataset:
    """Load a COCO-style annotations JSON into a Dataset, validating as we go.

    Raises on any image referencing an unknown category or a missing file: a dataset that loads
    but is quietly wrong is worse than one that fails.
    """
    ann_path = resolve(annotations)
    root = resolve(images_root)
    if not ann_path.is_file():
        raise FileNotFoundError(f"annotations not found: {ann_path}")

    raw = json.loads(ann_path.read_text())
    for key in ("categories", "images", "annotations"):
        if key not in raw:
            raise ValueError(f"{ann_path}: missing required key {key!r}")

    # category id → contiguous class index, ordered by id so the mapping is stable across loads.
    cats = sorted(raw["categories"], key=lambda c: c["id"])
    classes = tuple(c["name"] for c in cats)
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    by_image: dict[int, list[Box]] = {}
    for a in raw["annotations"]:
        if a["category_id"] not in cat_to_idx:
            raise ValueError(f"annotation {a.get('id')} has unknown category_id {a['category_id']}")
        x, y, w, h = a["bbox"]  # COCO stores xywh
        by_image.setdefault(a["image_id"], []).append(
            Box(x0=x, y0=y, x1=x + w, y1=y + h, class_id=cat_to_idx[a["category_id"]])
        )

    samples = []
    for img in raw["images"]:
        path = root / img["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"image referenced by annotations is missing: {path}")
        samples.append(
            Sample(
                image_path=path,
                width=img["width"],
                height=img["height"],
                boxes=tuple(by_image.get(img["id"], [])),
            )
        )

    return Dataset(classes=classes, samples=tuple(samples))
