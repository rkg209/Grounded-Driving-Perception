"""Single-phrase grounding + scoring (design decision 1).

One phrase = one prompt = one forward pass, and the prediction is the top-1 query by aggregated
score. `ground_phrase` reuses `GroundingDinoDetector`'s already-loaded `model`/`processor`/`device`
and the same free functions `detect_images` calls internally (`assign_classes`,
`convert_boxes_cxcywh_norm_to_xyxy_abs`, `assert_span_round_trip`) — no new inference code, and no
second scoring path that could drift from spec 02/03's. It deliberately does **not** construct a
fresh `GroundingDinoDetector` per phrase (which would call `.from_pretrained(...)` up to 300 times
on the cluster run): CLAUDE.md §4's memory-hygiene rule — learned the hard way earlier in this
spec's own implementation — treats every real model load as a resource to budget, and one shared
detector instance scored against N single-phrase prompts is both cheaper and behaviourally
identical to N separately-constructed detectors (same weights, same processor, same tokenizer).

`score_phrase` is the pure half: given a phrase and its (possibly absent) top-1 prediction, decide
correctness. It takes no detector and loads no model, which is what makes it unit-testable on
hand-built boxes (task 5's actual risk surface — the grounding-accuracy *definition*, not the
inference call).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gdp.config import Config
from gdp.data.core import Dataset, Sample
from gdp.data.prompt import PromptSpans
from gdp.detect.detector import (
    GroundingDinoDetector,
    assert_span_round_trip,
    assign_classes,
    convert_boxes_cxcywh_norm_to_xyxy_abs,
)
from gdp.detect.predictions import Detection
from gdp.eval.operating_point import box_iou
from gdp.ground.phrases import QUALIFIER_TYPES, Phrase, PhraseSet, load_gt_index
from gdp.paths import git_sha


def ground_phrase(
    detector: GroundingDinoDetector, sample: Sample, phrase: str, *, box_threshold: float
) -> Detection | None:
    """One forward pass: prompt = `"<phrase>."`, prediction = the top-1 query by score.

    Returns `None` if nothing clears `box_threshold` — the correct outcome for a well-grounded
    negative phrase, and an incorrect one (scored by `score_phrase`) for a missed positive.
    """
    spans = PromptSpans.from_classes((phrase,), tokenizer=detector.processor.tokenizer)
    assert_span_round_trip(spans, (phrase,), detector.processor.tokenizer)
    positive_map = spans.positive_map()

    with Image.open(sample.image_path) as img:
        image = img.convert("RGB")
        inputs = detector.processor(images=[image], text=[spans.prompt], return_tensors="pt").to(
            detector.device
        )
        with torch.no_grad():
            outputs = detector.model(**inputs)

    class_ids, scores = assign_classes(outputs.logits[0], positive_map)
    xyxy = convert_boxes_cxcywh_norm_to_xyxy_abs(outputs.pred_boxes[0], sample.width, sample.height)

    best_idx = int(scores.argmax())
    if scores[best_idx] < box_threshold:
        return None
    return Detection(
        image_id=sample.image_id,
        class_id=int(class_ids[best_idx]),
        score=float(scores[best_idx]),
        xyxy=tuple(float(v) for v in xyxy[best_idx]),
    )


@dataclass(frozen=True)
class PhraseResult:
    phrase: str
    qualifier_type: str
    image_id: int
    correct: bool
    predicted_box: tuple[float, float, float, float] | None
    predicted_score: float | None
    top1_iou: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "qualifier_type": self.qualifier_type,
            "image_id": self.image_id,
            "correct": self.correct,
            "predicted_box": list(self.predicted_box) if self.predicted_box else None,
            "predicted_score": self.predicted_score,
            "top1_iou": self.top1_iou,
        }


def score_phrase(
    phrase: Phrase,
    prediction: Detection | None,
    *,
    target_box: tuple[float, float, float, float] | None,
    iou_threshold: float = 0.5,
) -> PhraseResult:
    """Grounding accuracy's actual definition (spec § Approach 4): a positive phrase is correct
    iff its top-1 prediction has IoU >= `iou_threshold` with `target_box`; a negative phrase is
    correct iff nothing was predicted above the operating threshold."""
    if phrase.qualifier_type == "negative":
        return PhraseResult(
            phrase=phrase.phrase,
            qualifier_type=phrase.qualifier_type,
            image_id=phrase.image_id,
            correct=prediction is None,
            predicted_box=prediction.xyxy if prediction else None,
            predicted_score=prediction.score if prediction else None,
            top1_iou=None,
        )

    if prediction is None or target_box is None:
        iou = None
        correct = False
    else:
        iou = box_iou(prediction.xyxy, target_box)
        correct = iou >= iou_threshold

    return PhraseResult(
        phrase=phrase.phrase,
        qualifier_type=phrase.qualifier_type,
        image_id=phrase.image_id,
        correct=correct,
        predicted_box=prediction.xyxy if prediction else None,
        predicted_score=prediction.score if prediction else None,
        top1_iou=iou,
    )


def score_phrase_set(
    detector: GroundingDinoDetector,
    dataset: Dataset,
    phrase_set: PhraseSet,
    annotations_path: str | Path,
    *,
    box_threshold: float,
    iou_threshold: float = 0.5,
) -> list[PhraseResult]:
    """The full per-phrase pipeline: one `ground_phrase` forward pass + one `score_phrase` call
    per phrase in `phrase_set`, in order."""
    anns_by_id, _, _ = load_gt_index(annotations_path)
    samples_by_id = {s.image_id: s for s in dataset.samples}

    results: list[PhraseResult] = []
    for phrase in phrase_set.phrases:
        sample = samples_by_id[phrase.image_id]
        target_box = (
            anns_by_id[phrase.target_ann_id].xyxy if phrase.target_ann_id is not None else None
        )
        prediction = ground_phrase(detector, sample, phrase.phrase, box_threshold=box_threshold)
        results.append(
            score_phrase(phrase, prediction, target_box=target_box, iou_threshold=iou_threshold)
        )
    return results


def _stats(results: list[PhraseResult]) -> dict[str, Any]:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else 0.0}


def aggregate_results(results: list[PhraseResult]) -> dict[str, Any]:
    """Never one aggregate number (design decision 8): per qualifier type, overall, and
    positive/negative subtotals — a model that boxes everything scores 100% on positives and 0%
    on negatives, and a single aggregate would hide that."""
    per_type = {t: _stats([r for r in results if r.qualifier_type == t]) for t in QUALIFIER_TYPES}
    positives = _stats([r for r in results if r.qualifier_type != "negative"])
    negatives = _stats([r for r in results if r.qualifier_type == "negative"])
    overall = _stats(results)
    return {
        "per_type": per_type,
        "positives": positives,
        "negatives": negatives,
        "overall": overall,
    }


CAVEAT = (
    "Small, self-built grounding evaluation set (H9's sanctioned exception, CLAUDE.md §2) — "
    "not an official benchmark. Legitimate only because construction, bias, and drop rate are "
    "documented in data/grounding_eval/construction.md."
)


def build_grounding_metrics(
    *,
    aggregates: dict[str, Any],
    lock: dict[str, Any],
    box_threshold: float,
    chosen_on: str,
    iou_threshold: float,
    cfg: Config,
    dataset: str,
    is_synthetic: bool,
) -> dict[str, Any]:
    """Assemble `metrics.json` in the shape of `gdp.eval.metrics_io.build_metrics` — provenance
    first, per H7/H9 the `is_self_built_benchmark`/`caveat` labels travel with every number."""

    return {
        "dataset": dataset,
        "is_synthetic": is_synthetic,
        "is_self_built_benchmark": True,
        "caveat": CAVEAT,
        "model_id": cfg.detector.model_id,
        "num_phrases": lock["num_phrases"],
        "per_type_counts": lock["per_type_counts"],
        "phrases_sha256": lock["sha256"],
        "frozen_at": lock["frozen_at"],
        "box_threshold": box_threshold,
        "chosen_on": chosen_on,
        "iou_threshold": iou_threshold,
        **aggregates,
        "config": asdict(cfg),
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_grounding_metrics(metrics: dict[str, Any], *, out_dir: Path) -> Path:

    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def write_per_phrase(results: list[PhraseResult], *, out_dir: Path) -> Path:

    path = out_dir / "per_phrase.json"
    path.write_text(json.dumps([r.to_json() for r in results], indent=2) + "\n")
    return path
