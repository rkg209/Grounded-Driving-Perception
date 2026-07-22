"""Zero-shot-miss -> fine-tuned-hit image pairs (spec 03 acceptance 5, task 10).

A qualitative gallery for the write-up, not a metric: every ground-truth box the zero-shot model
missed (no same-class prediction at IoU >= threshold) but the fine-tuned model caught, cropped for
a side-by-side "this got better" comparison. This complements `compare`'s per-class AP delta with
concrete examples — it never computes or claims a number of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from gdp.eval.operating_point import GtBox, PredBox, box_iou


@dataclass(frozen=True)
class MissToHit:
    image_id: int
    gt_box: GtBox
    finetuned_box: PredBox


def find_miss_to_hit_pairs(
    ground_truth: list[GtBox],
    zeroshot_preds: list[PredBox],
    finetuned_preds: list[PredBox],
    *,
    iou_threshold: float = 0.5,
    limit: int | None = None,
) -> list[MissToHit]:
    """Every GT box the zero-shot model missed but the fine-tuned model caught, in GT order."""
    pairs: list[MissToHit] = []
    for gt in ground_truth:
        zeroshot_hit = any(
            p.image_id == gt.image_id
            and p.class_id == gt.class_id
            and box_iou(p.xyxy, gt.xyxy) >= iou_threshold
            for p in zeroshot_preds
        )
        if zeroshot_hit:
            continue

        finetuned_hit = next(
            (
                p
                for p in finetuned_preds
                if p.image_id == gt.image_id
                and p.class_id == gt.class_id
                and box_iou(p.xyxy, gt.xyxy) >= iou_threshold
            ),
            None,
        )
        if finetuned_hit is not None:
            pairs.append(MissToHit(image_id=gt.image_id, gt_box=gt, finetuned_box=finetuned_hit))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def save_pair_crops(
    pairs: list[MissToHit],
    image_path_by_id: dict[int, Path],
    class_names: list[str],
    *,
    out_dir: Path,
) -> list[str]:
    """Crop the (padded) GT box region from each pair's image. Returns paths relative to
    `out_dir`'s parent, so the caller can stamp them into a JSON summary alongside `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, pair in enumerate(pairs):
        path = image_path_by_id[pair.image_id]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x0, y0, x1, y1 = pair.gt_box.xyxy
            pad_x, pad_y = (x1 - x0) * 0.3, (y1 - y0) * 0.3
            crop = img.crop(
                (
                    max(0, x0 - pad_x),
                    max(0, y0 - pad_y),
                    min(img.width, x1 + pad_x),
                    min(img.height, y1 + pad_y),
                )
            )
            class_name = class_names[pair.gt_box.class_id]
            crop_path = out_dir / f"{i}_{path.stem}_{class_name}.png"
            crop.save(crop_path)
        saved.append(str(crop_path.relative_to(out_dir.parent)))
    return saved
