"""`evaluate_variant`'s per-variant resume (no real model needed — a fake Detector suffices).

A multi-thousand-image real split can run for hours; a walltime kill or crash between variants
must not force redoing an already-completed one on resubmission (progress_report.md [SEQ-0159]).
"""

from __future__ import annotations

import json

from gdp.data.core import Sample
from gdp.deploy.evaluate import evaluate_variant
from gdp.detect.predictions import Detection

GT_PATH = "tests/fixtures/mini_bdd/annotations.json"


class _CountingDetector:
    """Fails the test if `detect_images` is called after a resume should have skipped it."""

    def __init__(self) -> None:
        self.calls = 0

    def detect_images(self, samples, *, box_threshold, batch_size: int = 8):
        self.calls += 1
        return [
            Detection(image_id=s.image_id, class_id=0, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0))
            for s in samples
        ]


def _sample() -> Sample:
    return Sample(
        image_id=0,
        image_path="tests/fixtures/mini_bdd/images/0.jpg",
        width=100,
        height=100,
        boxes=(),
    )


def test_evaluate_variant_detects_when_no_predictions_file_exists(tmp_path):
    detector = _CountingDetector()

    evaluate_variant(
        detector,
        [_sample()],
        gt_path=GT_PATH,
        box_threshold=0.25,
        out_dir=tmp_path,
        variant="fp32-pt",
    )

    assert detector.calls == 1
    assert (tmp_path / "predictions_fp32-pt.json").is_file()


def test_evaluate_variant_skips_detection_when_predictions_file_already_exists(tmp_path):
    detector = _CountingDetector()
    predictions_path = tmp_path / "predictions_fp32-pt.json"
    predictions_path.write_text(json.dumps([]))

    evaluate_variant(
        detector,
        [_sample()],
        gt_path=GT_PATH,
        box_threshold=0.25,
        out_dir=tmp_path,
        variant="fp32-pt",
    )

    assert detector.calls == 0, "resume must not re-run detection when the predictions file exists"
