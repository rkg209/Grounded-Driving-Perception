"""YAML → dataclass configuration, validated on load.

No magic constants live in code (CLAUDE.md §8): every tunable is declared here with a default,
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
    # Raw BDD `det_20` label directory (holds det_train.json / det_val.json), input to spec 01's
    # converter. Not used by mini_bdd, which reads its raw fixtures straight from tests/fixtures/.
    raw_labels: str = "data/bdd100k/labels/det_20"
    # A class below this box count in a split is flagged in stats.json's rare_classes, not
    # silently dropped to flatter later mAP (H7).
    min_boxes_for_eval: int = 100

    def root_path(self) -> Path:
        return resolve(self.root)

    def annotations_path(self) -> Path:
        return resolve(self.annotations)

    def raw_labels_path(self) -> Path:
        return resolve(self.raw_labels)


@dataclass
class DetectorConfig:
    """Stage 1 — open-vocabulary detector."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    # Grounding-DINO's custom MS-deformable-attention kernels are not ONNX-exportable.
    # Spec 05 must set this True before export; training leaves it False for speed.
    disable_custom_kernels: bool = False
    # Candidate thresholds `gdp evaluate --sweep` picks from — swept on a held-out slice of
    # *train* only, never val (leakage would poison the H8 baseline). specs/02-zeroshot-baseline §5.
    sweep_candidates: list[float] = field(default_factory=lambda: [0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    def prompt(self, classes: list[str]) -> str:
        """Grounding-DINO expects classes separated by periods: 'car. pedestrian. bus.'"""
        return ". ".join(classes) + "."


@dataclass
class TrainingConfig:
    """Spec 03 — fine-tuning the detector. Tunables only; no magic constants in trainer code."""

    lr: float = 1e-4
    # Grounding-DINO's vision backbone is pretrained and near-converged; it moves slower than the
    # freshly-initialized fusion/decoder heads it feeds into.
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    epochs: int = 12
    batch_size: int = 4
    grad_accum: int = 1
    max_grad_norm: float = 0.1
    freeze_text_encoder: bool = True
    gradient_checkpointing: bool = False
    # The overfit-20 gate (spec 03 design decision 5): trains on a fixed subset and asserts
    # final loss < overfit_loss_target before the real run is allowed to start.
    overfit_images: int = 20
    overfit_max_steps: int = 200
    overfit_loss_target: float = 1.0
    save_every: int = 500
    log_every: int = 10

    def validate(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"training.lr must be positive, got {self.lr}")
        if self.backbone_lr_mult <= 0:
            raise ValueError(
                f"training.backbone_lr_mult must be positive, got {self.backbone_lr_mult}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be non-negative, got {self.weight_decay}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"training.warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.epochs <= 0:
            raise ValueError(f"training.epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive, got {self.batch_size}")
        if self.grad_accum <= 0:
            raise ValueError(f"training.grad_accum must be positive, got {self.grad_accum}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"training.max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.overfit_images <= 0:
            raise ValueError(f"training.overfit_images must be positive, got {self.overfit_images}")
        if self.overfit_max_steps <= 0:
            raise ValueError(
                f"training.overfit_max_steps must be positive, got {self.overfit_max_steps}"
            )
        if self.overfit_loss_target <= 0:
            raise ValueError(
                f"training.overfit_loss_target must be positive, got {self.overfit_loss_target}"
            )
        if self.save_every <= 0:
            raise ValueError(f"training.save_every must be positive, got {self.save_every}")
        if self.log_every <= 0:
            raise ValueError(f"training.log_every must be positive, got {self.log_every}")


@dataclass
class GroundingConfig:
    """Spec 04 — the self-built grounding evaluation set. A named H9 exception (CLAUDE.md §2):
    permitted only because its construction, bias, and drop rate are documented and it is
    labelled self-built everywhere it appears."""

    iou_threshold: float = 0.5
    num_frames: int = 60
    min_phrases: int = 150
    qualifier_types: list[str] = field(
        default_factory=lambda: ["spatial", "attribute", "relational", "negative"]
    )
    phrases_path: str = "data/grounding_eval/phrases.json"
    overlays_dir: str = "data/grounding_eval/overlays"

    def phrases_path_resolved(self) -> Path:
        return resolve(self.phrases_path)

    def overlays_dir_resolved(self) -> Path:
        return resolve(self.overlays_dir)

    def validate(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"grounding.iou_threshold must be in [0, 1], got {self.iou_threshold}")
        if self.num_frames <= 0:
            raise ValueError(f"grounding.num_frames must be positive, got {self.num_frames}")
        if self.min_phrases <= 0:
            raise ValueError(f"grounding.min_phrases must be positive, got {self.min_phrases}")
        if not self.qualifier_types:
            raise ValueError("grounding.qualifier_types must not be empty")


@dataclass
class VLMConfig:
    """Stage 2 — driving-scene VQA. Separate model from the detector (H6)."""

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 128
    lora_r: int = 16
    lora_alpha: int = 32
    # Number of camera views fed into the chat prompt (spec 06 design decision 5). Default 1
    # (CAM_FRONT) keeps Qwen2.5-VL's image-token budget sane; all six views are still stored
    # per-record so this is a training-time knob, not a data-loss decision.
    num_views: int = 1
    # Passed straight to the Qwen2.5-VL processor to cap per-image token count. None = processor
    # default. Spec 07 flags max_pixels/sequence-length blowup as an accuracy lever.
    max_pixels: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"vlm.max_new_tokens must be positive, got {self.max_new_tokens}")
        if not 1 <= self.num_views <= 6:
            raise ValueError(f"vlm.num_views must be in [1, 6], got {self.num_views}")
        if self.max_pixels is not None and self.max_pixels <= 0:
            raise ValueError(f"vlm.max_pixels must be positive, got {self.max_pixels}")


# DriveLM's own category spelling ("behavior", US) is kept verbatim as the canonical set of
# `category` values (spec 06 design decision 6) — the charter/spec prose says "behaviour" (UK),
# but rewriting the key would either break the join with DriveLM's raw JSON or silently merge two
# buckets in spec 08's per-category accuracy. An unknown category key is a counted drop, never a
# coerced one.
DRIVELM_CATEGORIES: tuple[str, ...] = ("perception", "prediction", "planning", "behavior")


@dataclass
class DriveLMConfig:
    """Spec 06 — DriveLM data pipeline. Stage 2's data source; preserves the official nuScenes
    train/val scene split verbatim (H2) — see specs/06-data-drivelm.md's status note."""

    annotations: str = "tests/fixtures/mini_drivelm/v1_1_mini_nus.json"
    nuscenes_root: str = "tests/fixtures/mini_drivelm"
    scene_meta: str = "tests/fixtures/mini_drivelm/scene.json"
    out_dir: str = "data/drivelm"
    # Fraction of *official train* scenes kept after scene-level stratified subsampling
    # (spec 06 design decision 7). Val is never subsampled — no config field controls it.
    train_scene_fraction: float = 1.0
    subsample_seed: int = 42
    categories: list[str] = field(default_factory=lambda: list(DRIVELM_CATEGORIES))

    def annotations_path(self) -> Path:
        return resolve(self.annotations)

    def nuscenes_root_path(self) -> Path:
        return resolve(self.nuscenes_root)

    def scene_meta_path(self) -> Path:
        return resolve(self.scene_meta)

    def out_dir_path(self) -> Path:
        return resolve(self.out_dir)

    def validate(self) -> None:
        if not 0.0 < self.train_scene_fraction <= 1.0:
            raise ValueError(
                f"drivelm.train_scene_fraction must be in (0, 1], got {self.train_scene_fraction}"
            )
        if not self.categories:
            raise ValueError("drivelm.categories must not be empty")


@dataclass
class VLMTrainingConfig:
    """Spec 07 — LoRA fine-tuning the VLM. Tunables only; no magic constants in trainer code."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: int = 2
    batch_size: int = 1
    # A 3B VLM at batch_size=1 needs accumulation to reach a usable effective batch size — unlike
    # spec 03's detector trainer, which refuses grad_accum, this one implements it (spec 07 task 6).
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    # The overfit-N gate (spec 07 design decision, mirroring spec 03): trains on a fixed subset and
    # asserts final loss < overfit_loss_target before the real run is allowed to start.
    overfit_qa_pairs: int = 16
    overfit_max_steps: int = 200
    overfit_loss_target: float = 0.05
    save_every: int = 200
    log_every: int = 10
    max_seq_len: int = 4096

    def validate(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"vlm_training.lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(
                f"vlm_training.weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(
                f"vlm_training.warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        if self.epochs <= 0:
            raise ValueError(f"vlm_training.epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"vlm_training.batch_size must be positive, got {self.batch_size}")
        if self.grad_accum <= 0:
            raise ValueError(f"vlm_training.grad_accum must be positive, got {self.grad_accum}")
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"vlm_training.max_grad_norm must be positive, got {self.max_grad_norm}"
            )
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                f"vlm_training.lora_dropout must be in [0, 1), got {self.lora_dropout}"
            )
        if self.overfit_qa_pairs <= 0:
            raise ValueError(
                f"vlm_training.overfit_qa_pairs must be positive, got {self.overfit_qa_pairs}"
            )
        if self.overfit_max_steps <= 0:
            raise ValueError(
                f"vlm_training.overfit_max_steps must be positive, got {self.overfit_max_steps}"
            )
        if self.overfit_loss_target <= 0:
            raise ValueError(
                f"vlm_training.overfit_loss_target must be positive, got {self.overfit_loss_target}"
            )
        if self.save_every <= 0:
            raise ValueError(f"vlm_training.save_every must be positive, got {self.save_every}")
        if self.log_every <= 0:
            raise ValueError(f"vlm_training.log_every must be positive, got {self.log_every}")
        if self.max_seq_len <= 0:
            raise ValueError(f"vlm_training.max_seq_len must be positive, got {self.max_seq_len}")


VALID_QUANTIZATION = ("dynamic", "static")


@dataclass
class DeployConfig:
    """Spec 05 — ONNX export, quantization, and latency benchmarking on the M4 (the edge target)."""

    warmup_iters: int = 50
    timed_iters: int = 200
    quantization: str = "dynamic"
    onnx_opset: int = 17

    def validate(self) -> None:
        if self.warmup_iters <= 0:
            raise ValueError(f"deploy.warmup_iters must be positive, got {self.warmup_iters}")
        if self.timed_iters <= 0:
            raise ValueError(f"deploy.timed_iters must be positive, got {self.timed_iters}")
        if self.quantization not in VALID_QUANTIZATION:
            raise ValueError(
                f"deploy.quantization must be one of {VALID_QUANTIZATION}, "
                f"got {self.quantization!r}"
            )
        if self.onnx_opset <= 0:
            raise ValueError(f"deploy.onnx_opset must be positive, got {self.onnx_opset}")


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    drivelm: DriveLMConfig = field(default_factory=DriveLMConfig)
    vlm_training: VLMTrainingConfig = field(default_factory=VLMTrainingConfig)

    def validate(self) -> Config:
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")
        if not self.dataset.classes:
            raise ValueError("dataset.classes must not be empty")
        if self.dataset.min_boxes_for_eval <= 0:
            raise ValueError("dataset.min_boxes_for_eval must be positive")
        for name, value in (
            ("detector.box_threshold", self.detector.box_threshold),
            ("detector.text_threshold", self.detector.text_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not self.detector.sweep_candidates:
            raise ValueError("detector.sweep_candidates must not be empty")
        for value in self.detector.sweep_candidates:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"detector.sweep_candidates entries must be in [0, 1], got {value}"
                )
        self.vlm.validate()
        self.training.validate()
        self.grounding.validate()
        self.deploy.validate()
        self.drivelm.validate()
        self.vlm_training.validate()
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
