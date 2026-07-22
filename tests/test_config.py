from __future__ import annotations

import pytest

from gdp.config import BDD100K_CLASSES, Config, load_config


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.device == "auto"
    assert cfg.dataset.classes == list(BDD100K_CLASSES)


def test_loads_default_yaml():
    cfg = load_config("configs/default.yaml")
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"
    assert cfg.vlm.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(cfg.dataset.classes) == 10


def test_merge_is_deep_and_right_wins(tmp_path):
    """A later config overrides only the keys it names — it must not wipe its siblings."""
    override = tmp_path / "o.yaml"
    override.write_text("detector:\n  box_threshold: 0.5\n")

    cfg = load_config("configs/default.yaml", override)
    assert cfg.detector.box_threshold == 0.5
    # sibling keys inside `detector` survive the merge
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"
    assert cfg.detector.text_threshold == 0.25


def test_bdd100k_overlay_only_changes_dataset():
    cfg = load_config("configs/default.yaml", "configs/bdd100k.yaml")
    assert cfg.dataset.name == "bdd100k"
    assert cfg.seed == 42
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"


def test_unknown_key_is_rejected(tmp_path):
    """A typo'd config key must fail loudly — silently ignoring it corrupts a result."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("detector:\n  box_treshold: 0.5\n")  # typo

    with pytest.raises(ValueError, match="unknown config key"):
        load_config("configs/default.yaml", bad)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml")


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: setattr(c, "device", "tpu"), "device must be one of"),
        (lambda c: setattr(c.detector, "box_threshold", 1.5), "must be in"),
        (lambda c: setattr(c.dataset, "classes", []), "must not be empty"),
        (lambda c: setattr(c.vlm, "max_new_tokens", 0), "must be positive"),
        (lambda c: setattr(c.detector, "sweep_candidates", []), "must not be empty"),
        (lambda c: setattr(c.detector, "sweep_candidates", [0.2, 1.5]), "must be in"),
        (lambda c: setattr(c.training, "lr", 0.0), "training.lr must be positive"),
        (
            lambda c: setattr(c.training, "warmup_ratio", 1.5),
            "training.warmup_ratio must be in",
        ),
        (
            lambda c: setattr(c.training, "overfit_loss_target", -1.0),
            "training.overfit_loss_target must be positive",
        ),
    ],
)
def test_validation_rejects_bad_values(mutate, match):
    cfg = Config()
    mutate(cfg)
    with pytest.raises((ValueError, TypeError), match=match):
        cfg.validate()


def test_train_detector_overlay_only_changes_training():
    cfg = load_config("configs/default.yaml", "configs/train_detector.yaml")
    assert cfg.training.lr == 1e-4
    assert cfg.training.overfit_images == 20
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"


def test_detector_prompt_format():
    """Grounding-DINO wants period-separated classes: 'car. pedestrian.'"""
    cfg = Config()
    assert cfg.detector.prompt(["car", "pedestrian"]) == "car. pedestrian."
