"""Seeded frame sampling + numbered-overlay rendering (design decision 5).

`frames.json`'s `sampled_at` timestamp is the evidence that frame sampling preceded phrase
authoring — the first of this spec's three bias mitigations (frames chosen before any phrase is
written, so a phrase can't be reverse-engineered from what the model already gets right). The
overlay PNGs exist purely to make authoring ergonomic: each GT box is drawn and labelled with its
annotation id, so a phrase's `target_ann_id` (design decision 3) is copied from the image, not
guessed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from gdp.data.core import Dataset
from gdp.ground.phrases import GtAnnotation, load_gt_index
from gdp.paths import git_sha, resolve

OVERLAY_COLOR = (255, 0, 0)


@dataclass(frozen=True)
class FrameSample:
    seed: int
    image_ids: tuple[int, ...]
    sampled_at: str
    git_sha: str

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "image_ids": list(self.image_ids),
            "sampled_at": self.sampled_at,
            "git_sha": self.git_sha,
        }


def sample_frames(dataset: Dataset, *, n: int, seed: int) -> FrameSample:
    """Deterministic: the same `(dataset, n, seed)` always samples the same image ids."""
    all_ids = sorted(s.image_id for s in dataset.samples)
    k = min(n, len(all_ids))
    chosen = tuple(sorted(random.Random(seed).sample(all_ids, k)))
    return FrameSample(
        seed=seed, image_ids=chosen, sampled_at=datetime.now(UTC).isoformat(), git_sha=git_sha()
    )


def write_frame_sample(frame_sample: FrameSample, *, out_path: str | Path) -> Path:
    path = resolve(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frame_sample.to_json(), indent=2) + "\n")
    return path


def render_overlay(image_path: Path, gt_boxes: list[GtAnnotation], *, out_path: str | Path) -> Path:
    """Draw every GT box in `gt_boxes`, labelled `<ann_id>:<class_name>`."""
    path = resolve(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for ann in gt_boxes:
            x0, y0, x1, y1 = ann.xyxy
            draw.rectangle([x0, y0, x1, y1], outline=OVERLAY_COLOR, width=2)
            label = f"{ann.ann_id}:{ann.class_name}"
            draw.text((x0, max(0, y0 - 12)), label, fill=OVERLAY_COLOR)
        img.save(path)
    return path


def sample_and_render(
    dataset: Dataset,
    annotations_path: str | Path,
    *,
    n: int,
    seed: int,
    frames_json_path: str | Path,
    overlays_dir: str | Path,
) -> tuple[FrameSample, list[Path]]:
    """The full task-4 pipeline: sample, write `frames.json`, render one overlay per frame."""
    frame_sample = sample_frames(dataset, n=n, seed=seed)
    write_frame_sample(frame_sample, out_path=frames_json_path)

    anns_by_id, ann_ids_by_image, _ = load_gt_index(annotations_path)
    samples_by_id = {s.image_id: s for s in dataset.samples}

    overlay_paths: list[Path] = []
    for image_id in frame_sample.image_ids:
        sample = samples_by_id[image_id]
        gt_boxes = [anns_by_id[aid] for aid in ann_ids_by_image.get(image_id, [])]
        out_path = resolve(overlays_dir) / f"{sample.image_path.stem}.png"
        overlay_paths.append(render_overlay(sample.image_path, gt_boxes, out_path=out_path))
    return frame_sample, overlay_paths
