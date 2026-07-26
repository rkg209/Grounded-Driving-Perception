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
        (
            lambda c: setattr(c.grounding, "iou_threshold", 1.5),
            "grounding.iou_threshold must be in",
        ),
        (lambda c: setattr(c.grounding, "num_frames", 0), "grounding.num_frames must be positive"),
        (
            lambda c: setattr(c.grounding, "min_phrases", 0),
            "grounding.min_phrases must be positive",
        ),
        (
            lambda c: setattr(c.grounding, "qualifier_types", []),
            "grounding.qualifier_types must not be empty",
        ),
        (
            lambda c: setattr(c.deploy, "warmup_iters", 0),
            "deploy.warmup_iters must be positive",
        ),
        (
            lambda c: setattr(c.deploy, "timed_iters", 0),
            "deploy.timed_iters must be positive",
        ),
        (
            lambda c: setattr(c.deploy, "quantization", "int4"),
            "deploy.quantization must be one of",
        ),
        (
            lambda c: setattr(c.deploy, "onnx_opset", 0),
            "deploy.onnx_opset must be positive",
        ),
        (
            lambda c: setattr(c.vlm, "num_views", 7),
            "vlm.num_views must be in",
        ),
        (
            lambda c: setattr(c.vlm, "max_pixels", -1),
            "vlm.max_pixels must be positive",
        ),
        (
            lambda c: setattr(c.drivelm, "train_scene_fraction", 0.0),
            "drivelm.train_scene_fraction must be in",
        ),
        (
            lambda c: setattr(c.drivelm, "train_scene_fraction", 1.5),
            "drivelm.train_scene_fraction must be in",
        ),
        (
            lambda c: setattr(c.drivelm, "categories", []),
            "drivelm.categories must not be empty",
        ),
        (lambda c: setattr(c.vlm_training, "lr", 0.0), "vlm_training.lr must be positive"),
        (
            lambda c: setattr(c.vlm_training, "weight_decay", -1.0),
            "vlm_training.weight_decay must be non-negative",
        ),
        (
            lambda c: setattr(c.vlm_training, "warmup_ratio", 1.5),
            "vlm_training.warmup_ratio must be in",
        ),
        (lambda c: setattr(c.vlm_training, "epochs", 0), "vlm_training.epochs must be positive"),
        (
            lambda c: setattr(c.vlm_training, "batch_size", 0),
            "vlm_training.batch_size must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "grad_accum", 0),
            "vlm_training.grad_accum must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "max_grad_norm", 0.0),
            "vlm_training.max_grad_norm must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "lora_dropout", 1.0),
            "vlm_training.lora_dropout must be in",
        ),
        (
            lambda c: setattr(c.vlm_training, "overfit_qa_pairs", 0),
            "vlm_training.overfit_qa_pairs must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "overfit_max_steps", 0),
            "vlm_training.overfit_max_steps must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "overfit_loss_target", -1.0),
            "vlm_training.overfit_loss_target must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "save_every", 0),
            "vlm_training.save_every must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "log_every", 0),
            "vlm_training.log_every must be positive",
        ),
        (
            lambda c: setattr(c.vlm_training, "max_seq_len", 0),
            "vlm_training.max_seq_len must be positive",
        ),
        (
            lambda c: setattr(c.vqa_eval, "do_sample", True),
            "vqa_eval.do_sample must be False",
        ),
        (
            lambda c: setattr(c.vqa_eval, "max_new_tokens", 0),
            "vqa_eval.max_new_tokens must be positive",
        ),
        (
            lambda c: setattr(c.vqa_eval, "num_beams", 0),
            "vqa_eval.num_beams must be positive",
        ),
        (
            lambda c: setattr(c.vqa_eval, "batch_size", 0),
            "vqa_eval.batch_size must be positive",
        ),
        (
            lambda c: setattr(c.vqa_eval, "failure_sample_size", 29),
            "vqa_eval.failure_sample_size must be >= 30",
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


def test_grounding_eval_overlay_loads():
    cfg = load_config("configs/default.yaml", "configs/grounding_eval.yaml")
    assert cfg.grounding.iou_threshold == 0.5
    assert cfg.grounding.min_phrases == 150
    assert cfg.grounding.qualifier_types == ["spatial", "attribute", "relational", "negative"]


def test_deploy_overlay_loads():
    cfg = load_config("configs/default.yaml", "configs/deploy.yaml")
    assert cfg.deploy.warmup_iters == 50
    assert cfg.deploy.timed_iters == 200
    assert cfg.deploy.quantization == "dynamic"
    assert cfg.deploy.onnx_opset == 17


def test_vlm_training_defaults_are_valid():
    cfg = load_config()
    assert cfg.vlm_training.lr == 1e-4
    assert cfg.vlm_training.overfit_qa_pairs == 16
    assert cfg.vlm_training.grad_accum == 8


def test_train_vlm_overlay_only_changes_vlm_training():
    cfg = load_config("configs/default.yaml", "configs/train_vlm.yaml")
    assert cfg.vlm_training.lr == 1e-4
    assert cfg.vlm_training.overfit_qa_pairs == 16
    assert cfg.vlm_training.max_seq_len == 4096
    assert cfg.vlm.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"


def test_drivelm_defaults_point_at_fixture():
    cfg = load_config("configs/default.yaml")
    assert cfg.drivelm.annotations == "tests/fixtures/mini_drivelm/v1_1_mini_nus.json"
    assert cfg.drivelm.categories == ["perception", "prediction", "planning", "behavior"]
    assert cfg.vlm.num_views == 1
    assert cfg.vlm.max_pixels is None


def test_drivelm_overlay_only_changes_drivelm():
    cfg = load_config("configs/default.yaml", "configs/drivelm.yaml")
    assert cfg.drivelm.annotations == "data/drivelm/v1_1_train_nus.json"
    assert cfg.drivelm.nuscenes_root == "data/nuscenes"
    assert cfg.detector.model_id == "IDEA-Research/grounding-dino-tiny"
    assert cfg.seed == 42


def test_vqa_eval_defaults_pin_greedy_decoding():
    """H8: base and fine-tuned runs must decode identically — round-trip proves the default
    config never accidentally turns sampling on."""
    cfg = load_config()
    assert cfg.vqa_eval.do_sample is False
    assert cfg.vqa_eval.num_beams == 1
    assert cfg.vqa_eval.max_new_tokens == 128
    assert cfg.vqa_eval.failure_sample_size >= 30


def test_vqa_eval_overlay_only_changes_vqa_eval():
    cfg = load_config("configs/default.yaml", "configs/vqa_eval.yaml")
    assert cfg.vqa_eval.do_sample is False
    assert cfg.vqa_eval.failure_sample_size == 40
    assert cfg.vlm.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert cfg.drivelm.categories == ["perception", "prediction", "planning", "behavior"]


def test_detector_prompt_format():
    """Grounding-DINO wants period-separated classes: 'car. pedestrian.'"""
    cfg = Config()
    assert cfg.detector.prompt(["car", "pedestrian"]) == "car. pedestrian."
