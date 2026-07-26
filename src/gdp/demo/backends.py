"""Model loading and inference for both stages, behind one lazy, single-slot registry.

`ModelRegistry` (plan design decision 11) caches by `(variant, tuple(classes))` for the detector
and `(mode, adapter)` for the VLM, evicting the previous slot on any key change — a 16 GB M4 cannot
hold two detector variants and a 3B VLM at once. The VLM (`LocalVLM`/`RemoteVLM`, decision 12) is
loaded lazily on the first question, never at app start.

`detect_scene` always runs the raw forward pass at `box_threshold=0.0` and caches it per
`(image, prompt, variant)` (design decision 6): the UI's threshold slider re-filters that cached
array instead of re-running inference, and the latency label is explicitly two-valued so a stale
number is never presented as fresh.
"""

from __future__ import annotations

import base64
import gc
import json
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import torch

from gdp.config import DetectorConfig
from gdp.demo.adapt import record_from_image, sample_from_image
from gdp.demo.honesty import BASE_VLM_WEIGHTS
from gdp.detect.detector import GroundingDinoDetector
from gdp.detect.predictions import Detection
from gdp.seed import select_device


class DetectorMode(StrEnum):
    TAXONOMY = "taxonomy"  # the 10 BDD100K classes — the evaluated prompt
    FREE_TEXT = "free-text"  # arbitrary phrase — H7 unmeasured demo


class Detector(Protocol):
    """Duck-typed the same way `gdp.deploy.evaluate.Detector` lets the PyTorch and ONNX
    detectors share one call site — `GroundingDinoDetector` and `OnnxDetector` both satisfy this
    with no shared base class."""

    config: DetectorConfig

    def detect_images(
        self, samples: Sequence, *, box_threshold: float, batch_size: int = 8
    ) -> list[Detection]: ...


@dataclass(frozen=True)
class DetectResult:
    """All queries' detections at `box_threshold=0.0`, plus provenance for the UI's badges."""

    detections: tuple[Detection, ...]
    classes: tuple[str, ...]
    seconds: float
    checkpoint: str
    variant: str
    mode: DetectorMode
    from_cache: bool

    def latency_label(self) -> str:
        if self.from_cache:
            return "re-filtered from the cached forward pass — no new inference"
        return f"detector: {self.seconds * 1000:.0f} ms (measured, this request)"


def filter_detections(result: DetectResult, *, box_threshold: float) -> list[Detection]:
    """Re-filter `result`'s already-computed detections — never a new forward pass."""
    return [d for d in result.detections if d.score >= box_threshold]


class VLMBackend(Protocol):
    """Duck-typed the same way `gdp.deploy.evaluate.Detector` lets the PyTorch/ONNX detectors
    share one call site (decision 12) — `LocalVLM` and `RemoteVLM` share no base class."""

    def answer(self, image_path: Path, question: str) -> tuple[str, float]: ...  # (text, seconds)

    def badge(self) -> str: ...

    def weights_badge(self) -> str: ...


class LocalVLM:
    """Base Qwen2.5-VL (or base + LoRA adapter), bf16, loaded lazily on the first `answer()` call
    — never at construction, so `ModelRegistry.get_local_vlm()` returning a fresh instance costs
    nothing until a question is actually asked (decision 11).

    No cache on the VLM path and no fixture-answer fallback, ever (spec §5's explicit
    prohibition, decision 12): every `answer()` call is a real `generate()`.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapter_dir: str | Path | None,
        device: torch.device,
        scenes_root: str | Path,
    ) -> None:
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.device = device
        self.scenes_root = scenes_root
        self._processor: object | None = None
        self._model: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        if self.adapter_dir is not None:
            from gdp.vqa.generate import load_adapter

            self._model = load_adapter(self.model_id, self.adapter_dir, device=self.device)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16
            )
            model.to(self.device)
            model.eval()
            self._model = model

    def answer(self, image_path: str | Path, question: str) -> tuple[str, float]:
        from gdp.vqa.generate import generate_with_stats

        self._ensure_loaded()
        record = record_from_image(image_path, question, scenes_root=self.scenes_root)
        start = time.perf_counter()
        result = generate_with_stats(
            self._processor,
            self._model,
            record,
            nuscenes_root=self.scenes_root,
            device=self.device,
            num_views=1,
        )
        elapsed = time.perf_counter() - start
        return result["text"], elapsed

    def badge(self) -> str:
        return f"Stage 2: local, bf16 on {self.device.type}"

    def weights_badge(self) -> str:
        """H8: names the adapter actually loaded, or admits none is (`BASE_VLM_WEIGHTS`)."""
        if self.adapter_dir is None:
            return BASE_VLM_WEIGHTS
        return f"adapter: {self.adapter_dir}"


class RemoteVLM:
    """POSTs `{image_b64, question}` to an HF-Space-style endpoint and expects back
    `{"text": ...}`. `seconds` is the full round trip, labelled as such in `badge()` — this is
    genuinely slower than a co-located call and the UI must not hide that.
    """

    def __init__(self, endpoint: str, *, timeout: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def answer(self, image_path: str | Path, question: str) -> tuple[str, float]:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        payload = json.dumps({"image_b64": image_b64, "question": question}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        return body["text"], elapsed

    def badge(self) -> str:
        host = urlparse(self.endpoint).netloc or self.endpoint
        return f"Stage 2: remote ({host}) — hosted because the 3B VLM is slow on this laptop"

    def weights_badge(self) -> str:
        """H8: a remote endpoint cannot honestly claim to know which weights it's serving —
        `BASE_VLM_WEIGHTS` would be as much a guess as "fine-tuned" would."""
        return "remote endpoint — weights not verifiable from this app"


class ModelRegistry:
    """One detector and one VLM resident at most (decision 11).

    `del` + `gc.collect()` + `torch.mps.empty_cache()` (when MPS is available) on every key
    change — the same memory-hygiene rule CLAUDE.md §4 applies to the test suite, applied here to
    a long-lived process instead: a Gradio app with a variant dropdown is the same hazard with a
    UI on it.
    """

    def __init__(self, detector_config: DetectorConfig, *, device: str = "auto") -> None:
        self.detector_config = detector_config
        self.device = device
        self._detector: Detector | None = None
        self._detector_key: tuple[str, tuple[str, ...]] | None = None
        self._forward_cache: dict[tuple[str, str, tuple[str, ...]], DetectResult] = {}
        self._vlm: VLMBackend | None = None
        self._vlm_key: tuple[str, str | None] | None = None

    def _evict_detector(self) -> None:
        if self._detector is not None:
            del self._detector
            self._detector = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _evict_vlm(self) -> None:
        if self._vlm is not None:
            del self._vlm
            self._vlm = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_detector(
        self, variant: str, classes: Sequence[str], *, onnx_path: str | Path | None = None
    ) -> Detector:
        key = (variant, tuple(classes))
        if key != self._detector_key:
            self._evict_detector()
            if variant == "int8-onnx":
                if onnx_path is None:
                    raise ValueError("onnx_path is required when variant='int8-onnx'")
                from gdp.deploy.onnx_detector import OnnxDetector

                self._detector = OnnxDetector(onnx_path, self.detector_config, classes)
            else:
                self._detector = GroundingDinoDetector(
                    self.detector_config, classes, device=self.device
                )
            self._detector_key = key
            self._forward_cache = {k: v for k, v in self._forward_cache.items() if k[1] == variant}
        return self._detector

    def get_local_vlm(
        self,
        model_id: str,
        *,
        adapter_dir: str | Path | None,
        scenes_root: str | Path,
    ) -> LocalVLM:
        key = ("local", str(adapter_dir) if adapter_dir else None)
        if key != self._vlm_key:
            self._evict_vlm()
            self._vlm = LocalVLM(
                model_id,
                adapter_dir=adapter_dir,
                device=select_device(self.device),
                scenes_root=scenes_root,
            )
            self._vlm_key = key
        return self._vlm

    def get_remote_vlm(self, endpoint: str) -> RemoteVLM:
        key = ("remote", endpoint)
        if key != self._vlm_key:
            self._evict_vlm()
            self._vlm = RemoteVLM(endpoint)
            self._vlm_key = key
        return self._vlm


def detect_scene(
    registry: ModelRegistry,
    path: str | Path,
    *,
    mode: DetectorMode,
    query: Sequence[str],
    variant: str,
    onnx_path: str | Path | None = None,
) -> DetectResult:
    """Run (or reuse a cached) raw forward pass over `query` on the image at `path`.

    `FREE_TEXT` always forces the PyTorch variant regardless of `variant` (decision 4) — the
    INT8-ONNX export is prompt-frozen (`gdp.deploy.export.ExportResult.fixed_prompt`) and cannot
    serve a free-text phrase; this is a visible fork, never a silent fallback.
    """
    effective_variant = "pt" if mode == DetectorMode.FREE_TEXT else variant
    classes = tuple(query)
    cache_key = (str(Path(path).resolve()), effective_variant, classes)

    cached = registry._forward_cache.get(cache_key)
    if cached is not None:
        return DetectResult(
            detections=cached.detections,
            classes=cached.classes,
            seconds=cached.seconds,
            checkpoint=cached.checkpoint,
            variant=cached.variant,
            mode=mode,
            from_cache=True,
        )

    detector = registry.get_detector(effective_variant, classes, onnx_path=onnx_path)
    sample = sample_from_image(path)

    start = time.perf_counter()
    detections = detector.detect_images([sample], box_threshold=0.0)
    elapsed = time.perf_counter() - start

    result = DetectResult(
        detections=tuple(detections),
        classes=classes,
        seconds=elapsed,
        checkpoint=detector.config.model_id,
        variant=effective_variant,
        mode=mode,
        from_cache=False,
    )
    registry._forward_cache[cache_key] = result
    return result
