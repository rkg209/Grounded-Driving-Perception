"""mAP via `pycocotools` — the reference implementation, never hand-rolled AP interpolation (H9).

This module only adapts pycocotools' I/O (paths/dicts in, a small typed result out); it does not
reimplement any of its math, so the number this produces is comparable to published work rather
than home-made (specs/02-zeroshot-baseline.md §3).
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class CocoMapResult:
    """`map`/`map50` are COCOeval's stats[0]/stats[1]: AP@[0.50:0.95] and AP@0.50, area=all,
    maxDets=100. `per_class_ap` is AP@[0.50:0.95] per category name, `nan` for a class with no
    ground-truth boxes in this split (pycocotools' own convention for "not evaluable")."""

    map: float
    map50: float
    per_class_ap: dict[str, float]


def _load_coco_gt_with_safe_ids(gt_path: str | Path) -> COCO:
    """Load a COCO-style ground-truth file, guarding against annotation id 0.

    `pycocotools.cocoeval.COCOeval.accumulate` encodes "this detection matched ground-truth
    annotation X" as the literal id X in its `dtMatches` array, and reads a match as boolean
    truthy (`np.logical_and(dtm, ...)`). An annotation whose id is `0` is therefore
    indistinguishable from "unmatched" and gets silently double-counted as a false positive
    instead of a true positive — corrupting mAP for exactly the one box that happens to own id
    0. `gdp.data.bdd100k.convert_bdd_to_coco` (spec 01) assigns annotation ids starting at 0, so
    every converted split has this landmine in its very first annotation. `pycocotools.loadRes`
    already reassigns detection ids as `1, 2, ...` internally, so this only needs fixing on the
    ground-truth side, and only here — not in the spec 01 converter, whose 0-based ids are a
    valid COCO-adjacent choice on their own and are unaffected by this pycocotools-specific
    sentinel quirk.
    """
    raw = json.loads(Path(gt_path).read_text())
    for ann in raw["annotations"]:
        ann["id"] += 1
    coco_gt = COCO()
    coco_gt.dataset = raw
    coco_gt.createIndex()
    return coco_gt


def evaluate_coco_map(
    gt_path: str | Path, predictions: list[dict[str, Any]] | str | Path
) -> CocoMapResult:
    """Score COCO-detection-result `predictions` against a COCO-style ground-truth file.

    `predictions` is either the in-memory list `gdp.detect.predictions.write_predictions` writes,
    or a path to that JSON on disk.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = _load_coco_gt_with_safe_ids(gt_path)
    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    if isinstance(predictions, str | Path):
        predictions = json.loads(Path(predictions).read_text())

    if not predictions:
        # pycocotools' loadRes() raises on an empty results list — report all-zero rather than
        # crash a `gdp evaluate` run just because the detector found nothing (H7: a real result,
        # not hidden).
        return CocoMapResult(
            map=0.0, map50=0.0, per_class_ap=dict.fromkeys(cat_names.values(), 0.0)
        )

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # precision: [T, R, K, A, M] = iou thresholds x recall thresholds x categories x areas x
    # maxDets. Per-class AP@[0.50:0.95] = mean over (T, R) at area="all" (index 0), maxDets=100
    # (last index) — the same slice COCOeval's own summarize() averages over K to get stats[0].
    precision = coco_eval.eval["precision"]
    per_class_ap: dict[str, float] = {}
    for k_idx, cat_id in enumerate(cat_ids):
        prec = precision[:, :, k_idx, 0, -1]
        prec = prec[prec > -1]
        per_class_ap[cat_names[cat_id]] = float(prec.mean()) if prec.size else float("nan")

    return CocoMapResult(
        map=float(coco_eval.stats[0]),
        map50=float(coco_eval.stats[1]),
        per_class_ap=per_class_ap,
    )
