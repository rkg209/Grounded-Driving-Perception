"""The open-vocabulary forgetting probe (spec 03 design decision 9) — a demo, labelled as one.

Out-of-taxonomy phrases like "construction cone" or "stroller" have no BDD100K ground truth, so
there is nothing to score them against. Under H9 that means no accuracy number, ever — this module
only ever produces a qualitative artifact (annotated crops + a JSON stamped `is_demo: true,
is_metric: false`), run once against the zero-shot checkpoint and once against the fine-tuned one so
the write-up can compare the two *qualitatively* (design decision 9). It reuses
`GroundingDinoDetector` completely unmodified, just with an arbitrary phrase list standing in for
`cfg.dataset.classes` — the model doesn't know the difference, which is exactly what makes this
probe meaningful.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from gdp.config import DetectorConfig
from gdp.data.core import Sample
from gdp.detect.detector import GroundingDinoDetector
from gdp.paths import git_sha

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_probe_phrases(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    phrases = data.get("phrases")
    if not phrases:
        raise ValueError(f"{path}: missing or empty 'phrases' list")
    return list(phrases)


def load_probe_images(images_root: str | Path, limit: int | None = None) -> list[Sample]:
    """Build `Sample`s straight from image files. No ground-truth annotations exist for
    out-of-taxonomy phrases, so there is nothing to load beyond the images themselves — each
    `Sample.boxes` is empty by construction, never a stand-in for ground truth."""
    root = Path(images_root)
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        paths = paths[:limit]

    samples = []
    for image_id, path in enumerate(paths):
        with Image.open(path) as img:
            width, height = img.size
        samples.append(
            Sample(image_id=image_id, image_path=path, width=width, height=height, boxes=())
        )
    return samples


def run_probe(
    detector_config: DetectorConfig,
    phrases: list[str],
    samples: list[Sample],
    *,
    device: str,
    box_threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    detector = GroundingDinoDetector(detector_config, phrases, device=device)
    detections = detector.detect_images(samples, box_threshold=box_threshold)

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    by_image = {s.image_id: s for s in samples}

    crop_paths = []
    for i, det in enumerate(detections):
        sample = by_image[det.image_id]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            x0, y0, x1, y1 = det.xyxy
            crop = img.crop((max(0, x0), max(0, y0), min(sample.width, x1), min(sample.height, y1)))
            phrase = phrases[det.class_id].replace(" ", "_")
            crop_path = crops_dir / f"{sample.image_path.stem}_{i}_{phrase}_{det.score:.2f}.png"
            crop.save(crop_path)
        crop_paths.append(str(crop_path.relative_to(out_dir)))

    return {
        "is_demo": True,
        "is_metric": False,
        "checkpoint": detector_config.model_id,
        "phrases": phrases,
        "box_threshold": box_threshold,
        "num_images": len(samples),
        "detections": [
            {
                "image_id": d.image_id,
                "phrase": phrases[d.class_id],
                "score": d.score,
                "bbox": list(d.xyxy),
            }
            for d in detections
        ],
        "crops": crop_paths,
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_probe_result(result: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "probe.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
