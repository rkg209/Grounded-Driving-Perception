"""Generate the synthetic mini-BDD fixture.

Why synthetic: BDD100K needs registration and is ~7 GB, and the M4 laptop must be able to run the
whole repo — tests, CLI, later the ONNX export — with no downloads and no cluster (CLAUDE.md §4).
So we draw a handful of crude "driving scenes" whose ground truth we know *exactly*, because we
drew it. The fixture is a data-contract test, not a perception test: it proves the pipeline plumbs
images and boxes correctly. It proves nothing about model accuracy, and no metric computed on it
may ever be reported as a result (H7).

Deterministic: same bytes every run. Regenerate with `make fixtures`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from gdp.config import BDD100K_CLASSES
from gdp.paths import resolve

OUT = resolve("tests/fixtures/mini_bdd")
W, H = 640, 360
N_IMAGES = 4

# (class name, RGB) for the handful of classes the fixture actually draws.
PALETTE = {
    "car": (70, 90, 160),
    "pedestrian": (200, 120, 90),
    "bus": (180, 160, 60),
    "traffic light": (60, 160, 90),
}


def _scene(rng: random.Random, idx: int) -> tuple[Image.Image, list[tuple[str, list[int]]]]:
    """One crude road scene. Returns the image and the boxes we actually drew (exact GT)."""
    img = Image.new("RGB", (W, H), (135, 170, 200))  # sky
    d = ImageDraw.Draw(img)
    horizon = H // 2
    d.rectangle([0, horizon, W, H], fill=(85, 85, 90))  # road
    d.line([(W // 2, horizon), (W // 2, H)], fill=(230, 230, 230), width=3)

    boxes: list[tuple[str, list[int]]] = []
    for name in rng.sample(sorted(PALETTE), k=rng.randint(2, 4)):
        bw = rng.randint(40, 110)
        bh = rng.randint(30, 90)
        x0 = rng.randint(5, W - bw - 5)
        # traffic lights hang above the horizon; everything else sits on the road
        y0 = rng.randint(20, horizon - bh - 5) if name == "traffic light" else rng.randint(
            horizon, H - bh - 5
        )
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=PALETTE[name], outline=(20, 20, 20), width=2)
        boxes.append((name, [x0, y0, bw, bh]))  # COCO xywh

    d.text((8, 8), f"synthetic scene {idx} - NOT REAL DATA", fill=(255, 255, 255))
    return img, boxes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)  # fixed: the fixture must be byte-stable across runs

    categories = [{"id": i, "name": n} for i, n in enumerate(BDD100K_CLASSES)]
    name_to_id = {c["name"]: c["id"] for c in categories}

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0

    for i in range(N_IMAGES):
        img, boxes = _scene(rng, i)
        fname = f"scene_{i:03d}.jpg"
        img.save(OUT / fname, quality=90)
        images.append({"id": i, "file_name": fname, "width": W, "height": H})
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

    print(f"wrote {len(images)} images + {len(annotations)} boxes to {Path(OUT)}")


if __name__ == "__main__":
    main()
