"""`ModelRegistry`/`detect_scene`/`filter_detections`/`DetectorMode`/`LocalVLM`/`RemoteVLM`
(spec 09 tasks 6-7).

The pure functions (`filter_detections`, `DetectResult.latency_label`, `ModelRegistry`
construction, `RemoteVLM` against a stubbed HTTP call) load no model and run in the default suite.
`detect_scene` and `LocalVLM.answer` call real `.from_pretrained(...)` and are marked
`model_heavy` individually, per `tests/test_cli.py`'s idiom — the rest of this file must keep
running on a bare `uv run pytest`.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gdp.config import DetectorConfig
from gdp.demo.backends import (
    DetectorMode,
    DetectResult,
    LocalVLM,
    ModelRegistry,
    RemoteVLM,
    detect_scene,
    filter_detections,
)
from gdp.detect.predictions import Detection
from gdp.paths import resolve

FIXTURE_IMAGE = resolve("tests/fixtures/mini_bdd/scene_000.jpg")


def _result(**overrides) -> DetectResult:
    defaults = dict(
        detections=(
            Detection(image_id=0, class_id=0, score=0.9, xyxy=(0.0, 0.0, 10.0, 10.0)),
            Detection(image_id=0, class_id=1, score=0.2, xyxy=(0.0, 0.0, 10.0, 10.0)),
        ),
        classes=("pedestrian", "car"),
        seconds=0.412,
        checkpoint="IDEA-Research/grounding-dino-tiny",
        variant="pt",
        mode=DetectorMode.TAXONOMY,
        from_cache=False,
    )
    defaults.update(overrides)
    return DetectResult(**defaults)


def test_detector_mode_values():
    assert DetectorMode.TAXONOMY.value == "taxonomy"
    assert DetectorMode.FREE_TEXT.value == "free-text"


def test_filter_detections_applies_threshold_without_new_inference():
    result = _result()
    kept = filter_detections(result, box_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_filter_detections_threshold_zero_keeps_everything():
    result = _result()
    assert len(filter_detections(result, box_threshold=0.0)) == 2


def test_latency_label_fresh_forward_pass():
    result = _result(from_cache=False, seconds=0.412)
    label = result.latency_label()
    assert "measured, this request" in label
    assert "412" in label


def test_latency_label_cached_forward_pass():
    result = _result(from_cache=True)
    assert result.latency_label() == ("re-filtered from the cached forward pass — no new inference")


def test_model_registry_construction_loads_no_model():
    """Instantiating the registry must not touch `.from_pretrained` — the detector is lazy."""
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    assert registry._detector is None
    assert registry._detector_key is None


def test_get_detector_requires_onnx_path_for_int8_variant():
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    with pytest.raises(ValueError):
        registry.get_detector("int8-onnx", ["car"], onnx_path=None)


@pytest.mark.model_heavy
def test_detect_scene_caches_the_raw_forward_pass():
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    first = detect_scene(
        registry,
        FIXTURE_IMAGE,
        mode=DetectorMode.TAXONOMY,
        query=["car", "pedestrian"],
        variant="pt",
    )
    assert first.from_cache is False
    assert first.seconds > 0
    assert "measured, this request" in first.latency_label()

    second = detect_scene(
        registry,
        FIXTURE_IMAGE,
        mode=DetectorMode.TAXONOMY,
        query=["car", "pedestrian"],
        variant="pt",
    )
    assert second.from_cache is True
    assert second.latency_label() == ("re-filtered from the cached forward pass — no new inference")
    assert second.detections == first.detections


@pytest.mark.model_heavy
def test_detect_scene_free_text_forces_pt_variant():

    registry = ModelRegistry(DetectorConfig(), device="cpu")
    result = detect_scene(
        registry,
        FIXTURE_IMAGE,
        mode=DetectorMode.FREE_TEXT,
        query=["traffic cone"],
        variant="int8-onnx",  # deliberately wrong — FREE_TEXT must override it
    )
    assert result.variant == "pt"


def test_local_vlm_construction_loads_no_model():
    import torch

    vlm = LocalVLM(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_dir=None,
        device=torch.device("cpu"),
        scenes_root=resolve("tests/fixtures/mini_bdd"),
    )
    assert vlm._model is None
    assert vlm._processor is None


def test_local_vlm_badge_names_device():
    import torch

    vlm = LocalVLM(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_dir=None,
        device=torch.device("cpu"),
        scenes_root=resolve("tests/fixtures/mini_bdd"),
    )
    assert vlm.badge() == "Stage 2: local, bf16 on cpu"


def test_remote_vlm_answer_posts_and_returns_round_trip_seconds():
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"text": "A cyclist is ahead."}).encode("utf-8")

    with patch("gdp.demo.backends.urllib.request.urlopen", return_value=_FakeResponse()):
        vlm = RemoteVLM("https://example-space.hf.space/answer")
        text, seconds = vlm.answer(FIXTURE_IMAGE, "What is ahead?")
    assert text == "A cyclist is ahead."
    assert seconds >= 0.0


def test_remote_vlm_badge_names_host():
    vlm = RemoteVLM("https://my-space.hf.space/answer")
    assert vlm.badge() == (
        "Stage 2: remote (my-space.hf.space) — hosted because the 3B VLM is slow on this laptop"
    )


def test_model_registry_get_local_vlm_evicts_on_adapter_change():
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    scenes_root = resolve("tests/fixtures/mini_bdd")
    first = registry.get_local_vlm(
        "Qwen/Qwen2.5-VL-3B-Instruct", adapter_dir=None, scenes_root=scenes_root
    )
    same = registry.get_local_vlm(
        "Qwen/Qwen2.5-VL-3B-Instruct", adapter_dir=None, scenes_root=scenes_root
    )
    assert first is same

    switched = registry.get_local_vlm(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_dir="runs/07-finetune-vlm/x/adapter",
        scenes_root=scenes_root,
    )
    assert switched is not first


def test_model_registry_get_remote_vlm_evicts_local_slot():
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    scenes_root = resolve("tests/fixtures/mini_bdd")
    local = registry.get_local_vlm(
        "Qwen/Qwen2.5-VL-3B-Instruct", adapter_dir=None, scenes_root=scenes_root
    )
    remote = registry.get_remote_vlm("https://example-space.hf.space/answer")
    assert remote is not local
    assert isinstance(remote, RemoteVLM)


@pytest.mark.model_heavy
def test_local_vlm_answer_produces_text_on_fixture_scene():
    import torch

    vlm = LocalVLM(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_dir=None,
        device=torch.device("cpu"),
        scenes_root=resolve("tests/fixtures/mini_bdd"),
    )
    text, seconds = vlm.answer(FIXTURE_IMAGE, "What is in this scene?")
    assert isinstance(text, str) and text.strip()
    assert seconds > 0
