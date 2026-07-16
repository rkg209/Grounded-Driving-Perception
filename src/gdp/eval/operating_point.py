"""Operating-point precision/recall: "at our box_threshold, we catch X% of pedestrians."

COCOeval's precision/recall curves are threshold-swept; this is the ADAS-readable number at *one*
score threshold and IoU >= 0.5, computed with a small, transparent greedy matcher — separate from
and much simpler than pycocotools, and easy to unit-test on hand-built boxes
(specs/02-zeroshot-baseline.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class GtBox:
    image_id: int
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class PredBox:
    image_id: int
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class PrecisionRecall:
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def match_operating_point(
    predictions: list[PredBox], ground_truth: list[GtBox], *, iou_threshold: float = 0.5
) -> tuple[dict[int, PrecisionRecall], PrecisionRecall]:
    """Greedy per-class matching at a fixed score cut. Returns `(per_class, overall)`.

    Per class: predictions are taken highest-score-first and matched to the highest-IoU unused
    ground-truth box in the same image at `IoU >= iou_threshold`; an unmatched prediction is a
    false positive, an unmatched ground-truth box is a false negative. `predictions` is expected
    to already be filtered to the operating threshold — this function does not itself filter by
    score, only by IoU/class/image.
    """
    classes = {p.class_id for p in predictions} | {g.class_id for g in ground_truth}
    per_class: dict[int, PrecisionRecall] = {}
    total_tp = total_fp = total_fn = 0

    for class_id in sorted(classes):
        class_preds = sorted(
            (p for p in predictions if p.class_id == class_id), key=lambda p: -p.score
        )
        class_gt = [g for g in ground_truth if g.class_id == class_id]
        gt_by_image: dict[int, list[int]] = {}
        for idx, g in enumerate(class_gt):
            gt_by_image.setdefault(g.image_id, []).append(idx)
        matched = [False] * len(class_gt)

        tp = fp = 0
        for p in class_preds:
            best_idx, best_iou = -1, 0.0
            for idx in gt_by_image.get(p.image_id, []):
                if matched[idx]:
                    continue
                iou = box_iou(p.xyxy, class_gt[idx].xyxy)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn = matched.count(False)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[class_id] = PrecisionRecall(
            precision=precision, recall=recall, tp=tp, fp=fp, fn=fn
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall = PrecisionRecall(
        precision=total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
        recall=total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
    )
    return per_class, overall


def sweep_threshold(
    predictions: list[PredBox],
    ground_truth: list[GtBox],
    *,
    candidates: list[float],
    iou_threshold: float = 0.5,
) -> tuple[float, PrecisionRecall]:
    """Pick the score threshold from `candidates` maximizing overall F1.

    Callers (spec 02 task 6) must only ever call this against a held-out slice of *train* —
    sweeping on val is leakage and would poison the H8 baseline this spec exists to produce.
    That guard lives in the caller (the CLI never wires `--split val` into a sweep), not here:
    this function has no notion of which split its inputs came from.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_threshold = candidates[0]
    best_pr: PrecisionRecall = match_operating_point([], ground_truth, iou_threshold=iou_threshold)[
        1
    ]
    best_f1 = -1.0
    for threshold in candidates:
        kept = [p for p in predictions if p.score >= threshold]
        _, overall = match_operating_point(kept, ground_truth, iou_threshold=iou_threshold)
        denom = overall.precision + overall.recall
        f1 = 2 * overall.precision * overall.recall / denom if denom else 0.0
        if f1 > best_f1:
            best_f1, best_threshold, best_pr = f1, threshold, overall
    return best_threshold, best_pr
