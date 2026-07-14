"""Generate the synthetic mini-BDD fixture.

Why synthetic: BDD100K needs registration and is ~7 GB, and the M4 laptop must be able to run the
whole repo — tests, CLI, later the ONNX export — with no downloads and no cluster (CLAUDE.md §4).
So we draw a handful of crude "driving scenes" whose ground truth we know *exactly*, because we
drew it. The fixture is a data-contract test, not a perception test: it proves the pipeline plumbs
images and boxes correctly. It proves nothing about model accuracy, and no metric computed on it
may ever be reported as a result (H7).

Deterministic: same bytes every run. Regenerate with `make fixtures`.

Spec 01 extends this fixture two ways, both required to exercise its acceptance criteria offline:

- **All 10 classes are drawn** (not just 4), split deterministically across the 4 scenes, so
  "counts non-zero for all 10" (acceptance 4) is a real assertion rather than one that can only be
  checked on real BDD100K.
- **The same boxes are also emitted in raw BDD `det_20` format** (`mini_bdd_raw/det_val.json`),
  giving the converter (`gdp/data/bdd100k.py`) an offline input, plus a deliberately dirty raw file
  (`mini_bdd_raw/det_dirty.json`) covering every drop/clip case the converter must handle.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from gdp.config import BDD100K_CLASSES
from gdp.paths import resolve

OUT = resolve("tests/fixtures/mini_bdd")
OUT_RAW = resolve("tests/fixtures/mini_bdd_raw")
W, H = 640, 360
N_IMAGES = 4

# One colour per BDD100K class, so all 10 render distinguishably.
PALETTE = {
    "pedestrian": (200, 120, 90),
    "rider": (170, 90, 140),
    "car": (70, 90, 160),
    "truck": (110, 70, 60),
    "bus": (180, 160, 60),
    "train": (90, 60, 130),
    "motorcycle": (60, 130, 140),
    "bicycle": (150, 150, 90),
    "traffic light": (60, 160, 90),
    "traffic sign": (200, 60, 60),
}

# Classes hanging above the horizon (traffic infrastructure) vs. sitting on the road (everything
# with wheels or feet). Deterministic per-scene assignment guarantees every one of the 10 classes
# is drawn at least once across the 4 scenes — a random sample could miss one by chance.
ABOVE_HORIZON = {"traffic light", "traffic sign"}
SCENE_CLASSES: list[list[str]] = [
    ["pedestrian", "rider", "car"],
    ["truck", "bus", "train"],
    ["motorcycle", "bicycle"],
    ["traffic light", "traffic sign"],
]


def _scene(rng: random.Random, idx: int) -> tuple[Image.Image, list[tuple[str, list[int]]]]:
    """One crude road scene. Returns the image and the boxes we actually drew (exact GT)."""
    img = Image.new("RGB", (W, H), (135, 170, 200))  # sky
    d = ImageDraw.Draw(img)
    horizon = H // 2
    d.rectangle([0, horizon, W, H], fill=(85, 85, 90))  # road
    d.line([(W // 2, horizon), (W // 2, H)], fill=(230, 230, 230), width=3)

    boxes: list[tuple[str, list[int]]] = []
    for name in SCENE_CLASSES[idx]:
        bw = rng.randint(40, 110)
        bh = rng.randint(30, 90)
        x0 = rng.randint(5, W - bw - 5)
        y0 = (
            rng.randint(20, horizon - bh - 5)
            if name in ABOVE_HORIZON
            else rng.randint(horizon, H - bh - 5)
        )
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=PALETTE[name], outline=(20, 20, 20), width=2)
        boxes.append((name, [x0, y0, bw, bh]))  # COCO xywh

    d.text((8, 8), f"synthetic scene {idx} - NOT REAL DATA", fill=(255, 255, 255))
    return img, boxes


def _write_coco(images: list[dict], annotations: list[dict]) -> None:
    categories = [{"id": i, "name": n} for i, n in enumerate(BDD100K_CLASSES)]
    payload = {
        "info": {
            "description": "Synthetic mini-BDD fixture. NOT real BDD100K. "
            "For pipeline tests only — never report a metric computed on this (H7).",
            "classes_follow": "BDD100K 10-class detection taxonomy",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    (OUT / "annotations.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_raw(images: list[dict], boxes_by_image: dict[int, list[tuple[str, list[int]]]]) -> None:
    """Raw BDD `det_20` format: xyxy `box2d`, one frame per image, `labels` per box."""
    frames = []
    for img in images:
        labels = []
        for name, (x, y, w, h) in boxes_by_image.get(img["id"], []):
            labels.append(
                {
                    "category": name,
                    "box2d": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                }
            )
        frames.append({"name": img["file_name"], "labels": labels})
    (OUT_RAW / "det_val.json").write_text(json.dumps(frames, indent=2) + "\n")


def _write_dirty() -> None:
    """One instance of every nasty case the converter (`bdd100k.py`) must count and drop, plus
    the one it must clip rather than drop. Used by `test_dirty_frames_are_counted_by_reason`."""
    frames = [
        {"name": "dirty_labels_null.jpg", "labels": None},
        {
            "name": "dirty_unknown_category.jpg",
            "labels": [{"category": "trailer", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
        },
        {
            "name": "dirty_degenerate.jpg",
            "labels": [{"category": "car", "box2d": {"x1": 30, "y1": 30, "x2": 30, "y2": 60}}],
        },
        {
            "name": "dirty_out_of_frame.jpg",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 1000, "y1": 1000, "x2": 1100, "y2": 1100},
                }
            ],
        },
        {
            "name": "dirty_edge_straddle.jpg",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": W - 20, "y1": 100, "x2": W + 40, "y2": 160},
                }
            ],
        },
        {
            "name": "dirty_no_box2d.jpg",
            "labels": [{"category": "car", "attributes": {"occluded": True}}],
        },
    ]
    (OUT_RAW / "det_dirty.json").write_text(json.dumps(frames, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)  # fixed: the fixture must be byte-stable across runs

    name_to_id = {n: i for i, n in enumerate(BDD100K_CLASSES)}

    images: list[dict] = []
    annotations: list[dict] = []
    boxes_by_image: dict[int, list[tuple[str, list[int]]]] = {}
    ann_id = 0

    for i in range(N_IMAGES):
        img, boxes = _scene(rng, i)
        fname = f"scene_{i:03d}.jpg"
        img.save(OUT / fname, quality=90)
        images.append({"id": i, "file_name": fname, "width": W, "height": H})
        boxes_by_image[i] = boxes
        for name, bbox in boxes:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": name_to_id[name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    _write_coco(images, annotations)
    _write_raw(images, boxes_by_image)
    _write_dirty()

    print(f"wrote {len(images)} images + {len(annotations)} boxes to {Path(OUT)}")
    print(f"wrote raw fixtures to {Path(OUT_RAW)}")


if __name__ == "__main__":
    main()
