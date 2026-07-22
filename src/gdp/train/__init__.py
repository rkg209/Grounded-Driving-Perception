"""Spec 03 — fine-tuning Grounding-DINO on BDD100K."""

from gdp.train.dataset import DetectionCollator, SampleDataset
from gdp.train.labels import assert_label_map_alignment, build_coco_target
from gdp.train.trainer import DetectorTrainer

__all__ = [
    "DetectionCollator",
    "DetectorTrainer",
    "SampleDataset",
    "assert_label_map_alignment",
    "build_coco_target",
]
