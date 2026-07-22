from __future__ import annotations

import json

import pytest

from gdp.config import DetectorConfig
from gdp.probe.openvocab import (
    load_probe_images,
    load_probe_phrases,
    run_probe,
    write_probe_result,
)

FIXTURE_ROOT = "tests/fixtures/mini_bdd"


def test_load_probe_phrases_reads_the_yaml_list():
    phrases = load_probe_phrases("configs/openvocab_probe.yaml")
    assert "construction cone" in phrases
    assert len(phrases) >= 1


def test_load_probe_phrases_rejects_empty_list(tmp_path):
    bad = tmp_path / "empty.yaml"
    bad.write_text("phrases: []\n")
    with pytest.raises(ValueError, match="missing or empty"):
        load_probe_phrases(bad)


def test_load_probe_images_builds_samples_with_no_ground_truth():
    samples = load_probe_images(FIXTURE_ROOT, limit=2)
    assert len(samples) == 2
    for sample in samples:
        assert sample.boxes == ()
        assert sample.width > 0 and sample.height > 0


@pytest.mark.model_heavy
def test_run_probe_writes_demo_stamped_json_with_no_accuracy_field(tmp_path):
    """H9: no capability outside the evaluated set gets an accuracy number — ever."""
    detector_config = DetectorConfig(box_threshold=0.15)
    samples = load_probe_images(FIXTURE_ROOT, limit=2)
    phrases = ["construction cone", "stroller"]

    result = run_probe(
        detector_config,
        phrases,
        samples,
        device="cpu",
        box_threshold=0.15,
        out_dir=tmp_path,
    )

    assert result["is_demo"] is True
    assert result["is_metric"] is False
    forbidden_keys = {"accuracy", "map", "score_avg", "precision", "recall"}
    assert not forbidden_keys & result.keys()
    assert result["phrases"] == phrases
    assert result["num_images"] == 2

    probe_path = write_probe_result(result, out_dir=tmp_path)
    assert probe_path.is_file()
    reloaded = json.loads(probe_path.read_text())
    assert reloaded["is_demo"] is True


@pytest.mark.model_heavy
def test_run_probe_saves_a_crop_per_detection(tmp_path):
    detector_config = DetectorConfig(box_threshold=0.01)
    samples = load_probe_images(FIXTURE_ROOT, limit=1)
    phrases = ["car", "person"]

    result = run_probe(
        detector_config,
        phrases,
        samples,
        device="cpu",
        box_threshold=0.01,
        out_dir=tmp_path,
    )

    assert len(result["crops"]) == len(result["detections"])
    for crop_rel_path in result["crops"]:
        assert (tmp_path / crop_rel_path).is_file()
