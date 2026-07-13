"""YAML → dataclass configuration, validated on load.

No magic constants live in code (CLAUDE.md §7): every tunable is declared here with a default,
loaded from `configs/*.yaml`, and validated before use. Unknown keys are an error, not a shrug —
a silently-ignored typo in a config is how a "result" becomes a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from gdp.paths import resolve

# BDD100K's 10 detection classes. Order is load-bearing: it defines the class index used for
# Grounding-DINO's contrastive text targets, and the prompt is built by joining these with ". ".
BDD100K_CLASSES: tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


@dataclass
class PathsConfig:
    data_root: str = "data"
    runs_root: str = "runs"

    def data(self) -> Path:
        return resolve(self.data_root)

    def runs(self) -> Path:
        return resolve(self.runs_root)


@dataclass
class DatasetConfig:
    name: str = "mini_bdd"
    root: str = "tests/fixtures/mini_bdd"
    annotations: str = "tests/fixtures/mini_bdd/annotations.json"
    classes: list[str] = field(default_factory=lambda: list(BDD100K_CLASSES))

    def root_path(self) -> Path:
        return resolve(self.root)

    def annotations_path(self) -> Path:
        return resolve(self.annotations)


@dataclass
class DetectorConfig:
    """Stage 1 — open-vocabulary detector."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    # Grounding-DINO's custom MS-deformable-attention kernels are not ONNX-exportable.
    # Spec 05 must set this True before export; training leaves it False for speed.
    disable_custom_kernels: bool = False

    def prompt(self, classes: list[str]) -> str:
        """Grounding-DINO expects classes separated by periods: 'car. pedestrian. bus.'"""
        return ". ".join(classes) + "."


@dataclass
class VLMConfig:
    """Stage 2 — driving-scene VQA. Separate model from the detector (H6)."""

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 128
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    def validate(self) -> Config:
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")
        if not self.dataset.classes:
            raise ValueError("dataset.classes must not be empty")
        for name, value in (
            ("detector.box_threshold", self.detector.box_threshold),
            ("detector.text_threshold", self.detector.text_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.vlm.max_new_tokens <= 0:
            raise ValueError("vlm.max_new_tokens must be positive")
        return self


def _build(cls: type, data: dict[str, Any], path: str = "") -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys.

    Field types are read via get_type_hints rather than `Field.type`: this module uses
    `from __future__ import annotations`, so `Field.type` is the *string* "PathsConfig",
    which would never match is_dataclass().
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"unknown config key(s) in {where}: {sorted(unknown)}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        ftype = hints[key]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*paths: str | Path) -> Config:
    """Load and merge YAML configs left-to-right (later wins), then validate.

    With no arguments, returns the validated defaults.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        f = resolve(p)
        if not f.is_file():
            raise FileNotFoundError(f"config not found: {f}")
        loaded = yaml.safe_load(f.read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"config {f} must be a YAML mapping, got {type(loaded).__name__}")
        merged = _deep_merge(merged, loaded)

    return _build(Config, merged).validate()
