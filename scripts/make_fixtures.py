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

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from gdp.config import BDD100K_CLASSES
from gdp.paths import resolve

OUT = resolve("tests/fixtures/mini_bdd")
OUT_RAW = resolve("tests/fixtures/mini_bdd_raw")
W, H = 640, 360
N_IMAGES = 4

OUT_DRIVELM = resolve("tests/fixtures/mini_drivelm")
DRIVELM_W, DRIVELM_H = 64, 36
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
# Real official nuScenes v1.0-trainval scene *names* (verified against src/gdp/data/
# nuscenes_splits.json) — using genuine names means gdp.data.splits.resolve_split resolves the
# fixture's scenes against the real vendored split list, with no parallel "fixture split" concept
# to keep honest separately (H9).
DRIVELM_TRAIN_SCENES = ("scene-0001", "scene-0002")
DRIVELM_VAL_SCENES = ("scene-0003", "scene-0012")

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


def _token(name: str) -> str:
    """Deterministic 32-hex nuScenes-style token, derived from the scene/frame name — no RNG
    needed for byte-stability."""
    return hashlib.md5(name.encode()).hexdigest()


def _drivelm_image(scene_name: str, camera: str) -> Image.Image:
    img = Image.new("RGB", (DRIVELM_W, DRIVELM_H), (120, 140, 170))
    d = ImageDraw.Draw(img)
    d.rectangle([0, DRIVELM_H // 2, DRIVELM_W, DRIVELM_H], fill=(80, 80, 85))
    d.text((2, 2), f"{scene_name[-2:]}/{camera[4:6]}", fill=(255, 255, 255))
    return img


def _qa(question: str, answer: str) -> dict[str, Any]:
    """One QA item in the real `v1_1_train_nus.json` schema: exactly these seven keys and **no**
    `tag` — the real file has none; the vendored extract_data.py assigns them ([SEQ-0143], where a
    fixture that invented `tag` hid a converter bug that dropped all 377,956 real QA)."""
    return {
        "Q": question,
        "A": answer,
        "C": None,
        "con_up": None,
        "con_down": None,
        "cluster": None,
        "layer": None,
    }


def _object_ref(scene_name: str) -> str:
    return f"<c1,CAM_FRONT,{100 + len(scene_name)}.0,50.0>"


def _drivelm_qa(scene_name: str, *, rich: bool) -> dict[str, list[dict[str, Any]]]:
    """QA for one clean frame, worded so DriveLM's own extract_data.py selects (and tags) exactly:
    perception[0] [2] (answer names the object class), prediction[0] [3] (answer contains the
    object's location key), planning[0] [1] ("What actions could the ego vehicle take"),
    behavior[0] [0]. perception[1] is selected by none of its rules, so on a val scene it is a
    `val_not_in_official_eval` drop. `rich` (train scenes) adds the other rule types — a "moving
    status" question [0], a yes/no prediction [0], collision and safe-action planning [1] — so the
    fixture exercises every extract_data branch. `<cX,CAM,x,y>` object refs stay byte-verbatim."""
    obj = _object_ref(scene_name)
    qa = {
        "perception": [
            _qa(
                "What are the important objects in the current scene?",
                f"There is one pedestrian near the crosswalk {obj}.",
            ),
            _qa(
                "What are objects to the front right of the ego car?",
                "There is one barrier to the front right of the ego car.",
            ),
        ],
        "prediction": [
            _qa("What will the pedestrian do next?", f"The pedestrian {obj} will keep walking."),
        ],
        "planning": [
            _qa(
                f"What actions could the ego vehicle take based on {obj}? Why take this action?",
                "Slow down and yield, high probability.",
            ),
        ],
        "behavior": [
            _qa(
                "Predict the behavior of the ego vehicle.",
                "The ego vehicle is going straight. The ego vehicle is driving slowly.",
            ),
        ],
    }
    if rich:
        qa["perception"].append(_qa(f"What is the moving status of object {obj}?", "Going ahead."))
        qa["prediction"].append(_qa(f"Is {obj} a traffic sign or a road barrier?", "No."))
        qa["planning"].append(
            _qa(
                f"What actions taken by the ego vehicle can lead to a collision with {obj}?",
                "Accelerating straight ahead.",
            )
        )
        qa["planning"].append(
            _qa(
                "In this scenario, what are safe actions to take for the ego vehicle?",
                "Brake gently to a stop.",
            )
        )
    return qa


def _key_object_infos(scene_name: str) -> dict[str, dict[str, str]]:
    # extract_data reads the class as `Visual_description.split('.')[0]`.
    return {
        _object_ref(scene_name): {
            "Category": "Vulnerable road user",
            "Status": "Moving",
            "Visual_description": "Pedestrian near the crosswalk.",
            "2d_bbox": [95.0, 30.0, 125.0, 70.0],
        }
    }


def _make_mini_drivelm() -> None:
    """Synthetic mini-DriveLM: 4 scenes under real official nuScenes scene *names* (2 train, 2
    val — see DRIVELM_TRAIN_SCENES/DRIVELM_VAL_SCENES), six tiny camera JPEGs each, all four
    DriveLM categories present, and one deliberately malformed QA/record per drop reason so
    `gdp.data.drivelm`'s drop accounting is genuinely exercised rather than merely plumbed."""
    OUT_DRIVELM.mkdir(parents=True, exist_ok=True)
    samples_dir = OUT_DRIVELM / "samples"

    scenes = {}
    scene_meta = []
    all_scene_names = list(DRIVELM_TRAIN_SCENES) + list(DRIVELM_VAL_SCENES)

    for scene_name in all_scene_names:
        scene_token = _token(scene_name)
        clean_frame_token = _token(scene_name + "_frame")
        scene_meta.append({"token": scene_token, "name": scene_name})

        def _full_image_paths(scene_name: str = scene_name) -> dict[str, str]:
            # Real DriveLM paths are relative to its `data/QA_dataset_nus/` dir, i.e.
            # `../nuscenes/samples/...`; the converter normalizes them (normalize_image_path).
            paths = {}
            for cam in CAMERAS:
                rel = f"samples/{cam}/{scene_name}__{cam}.jpg"
                cam_dir = samples_dir / cam
                cam_dir.mkdir(parents=True, exist_ok=True)
                if not (OUT_DRIVELM / rel).is_file():
                    _drivelm_image(scene_name, cam).save(OUT_DRIVELM / rel, quality=90)
                paths[cam] = f"../nuscenes/{rel}"
            return paths

        # Every scene gets one clean key_frame, so a scene that also carries a malformed frame
        # (below) still has at least one surviving QA record in its split — a frame-level drop
        # must not silently zero out an entire split (that would defeat the fixture's own
        # purpose: proving the drop accounting works *without* losing split coverage).
        is_train = scene_name in DRIVELM_TRAIN_SCENES
        clean_qa = _drivelm_qa(scene_name, rich=is_train)
        key_frames = {
            clean_frame_token: {
                "key_object_infos": _key_object_infos(scene_name),
                "QA": clean_qa,
                "image_paths": _full_image_paths(),
            }
        }

        if scene_name == "scene-0001":
            # DROP_REASONS["missing_qa_text"]: empty answer, alongside the clean QA above.
            clean_qa["perception"].append(_qa("What is behind the ego vehicle?", ""))
        if scene_name == "scene-0002":
            # DROP_REASONS["unknown_category"]: key outside DRIVELM_CATEGORIES.
            clean_qa["misc"] = [_qa("Off-taxonomy question?", "Off-taxonomy answer.")]
        # DROP_REASONS["val_not_in_official_eval"] needs no special case: every val clean frame's
        # perception[1] is selected by none of extract_data's rules.
        if scene_name == "scene-0003":
            # DROP_REASONS["missing_image_path"]: a second frame missing a camera key entirely.
            bad_token = _token(scene_name + "_frame_missing_view")
            bad_paths = _full_image_paths()
            del bad_paths["CAM_BACK_LEFT"]
            key_frames[bad_token] = {
                "key_object_infos": {},
                "QA": _drivelm_qa(scene_name, rich=False),
                "image_paths": bad_paths,
            }
        if scene_name == "scene-0012":
            # DROP_REASONS["image_file_missing"]: path declared, file never written.
            bad_token = _token(scene_name + "_frame_missing_file")
            bad_paths = _full_image_paths()
            bad_paths["CAM_BACK_RIGHT"] = (
                f"../nuscenes/samples/CAM_BACK_RIGHT/{scene_name}__does_not_exist.jpg"
            )
            key_frames[bad_token] = {
                "key_object_infos": {},
                "QA": _drivelm_qa(scene_name, rich=False),
                "image_paths": bad_paths,
            }

        scenes[scene_token] = {"key_frames": key_frames}

    (OUT_DRIVELM / "v1_1_mini_nus.json").write_text(json.dumps(scenes, indent=2) + "\n")
    (OUT_DRIVELM / "scene.json").write_text(json.dumps(scene_meta, indent=2) + "\n")
    (OUT_DRIVELM / "mini_splits.json").write_text(
        json.dumps(
            {
                "note": "Informational only. Fixture scenes use real official nuScenes scene "
                "names, so gdp.data.splits.resolve_split resolves them against the real vendored "
                "src/gdp/data/nuscenes_splits.json directly — this file is not read by any code "
                "path, it documents the expected resolution for test readers.",
                "train": list(DRIVELM_TRAIN_SCENES),
                "val": list(DRIVELM_VAL_SCENES),
            },
            indent=2,
        )
        + "\n"
    )
    n_images = sum(1 for _ in samples_dir.rglob("*.jpg"))
    print(f"wrote {len(all_scene_names)} scenes / {n_images} images to {OUT_DRIVELM}")


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

    _make_mini_drivelm()


if __name__ == "__main__":
    main()
