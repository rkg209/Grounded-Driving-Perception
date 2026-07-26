"""`build_app` and the pure helper functions behind each panel (spec 09 task 8).

`gradio` is imported lazily inside `build_app` (decision 3) — verified here by AST inspection so
the test is meaningful even in an environment where `gradio` isn't installed. `build_app` itself is
exercised with stub `ModelRegistry`/`VLMBackend` objects that never touch a real model, so it stays
in the default (non-`model_heavy`) suite. One `model_heavy` test does a real zero-shot detect via
`run_detect` on `tests/fixtures/mini_bdd/scene_000.jpg`.
"""

from __future__ import annotations

import ast

import pytest

from gdp.config import DetectorConfig, load_config
from gdp.demo.backends import DetectorMode, ModelRegistry
from gdp.demo.honesty import BASE_VLM_WEIGHTS, ZEROSHOT_WEIGHTS
from gdp.demo.scenes import Scene, load_scenes
from gdp.paths import repo_root, resolve

APP_SRC = repo_root() / "src" / "gdp" / "demo" / "app.py"
FIXTURE_SCENE = resolve("tests/fixtures/mini_bdd/scene_000.jpg")


class _StubVLM:
    def __init__(self, weights_badge_value: str = BASE_VLM_WEIGHTS) -> None:
        self._weights_badge_value = weights_badge_value

    def answer(self, image_path, question):
        return f"stub answer to: {question}", 0.01

    def badge(self):
        return "Stage 2: local, bf16 on cpu"

    def weights_badge(self):
        return self._weights_badge_value


def test_gradio_is_never_imported_at_module_level():
    tree = ast.parse(APP_SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "gradio" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "gradio" and (node.module or "").split(".")[0] != "gradio"


def test_build_app_imports_gradio_somewhere_inside_it():
    tree = ast.parse(APP_SRC.read_text())
    build_app_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_app"
    )
    imports_gradio = any(
        isinstance(n, ast.Import) and any(a.name == "gradio" for a in n.names)
        for n in ast.walk(build_app_fn)
    )
    assert imports_gradio


def test_checkpoint_badge_zeroshot_default():
    from gdp.demo.app import checkpoint_badge

    default_id = DetectorConfig().model_id
    assert checkpoint_badge(default_id, default_model_id=default_id) == ZEROSHOT_WEIGHTS


def test_checkpoint_badge_names_a_different_checkpoint_verbatim():
    from gdp.demo.app import checkpoint_badge

    default_id = DetectorConfig().model_id
    badge = checkpoint_badge("runs/03-finetune/x/checkpoint-500", default_model_id=default_id)
    assert badge == "checkpoint: runs/03-finetune/x/checkpoint-500"


def test_vlm_weights_badge_base_vs_adapter():
    from gdp.demo.app import vlm_weights_badge

    assert vlm_weights_badge(None) == BASE_VLM_WEIGHTS
    assert vlm_weights_badge("runs/07-finetune-vlm/x/adapter") == (
        "adapter: runs/07-finetune-vlm/x/adapter"
    )


def test_resolve_query_taxonomy_returns_configured_classes():
    from gdp.demo.app import resolve_query

    classes = ("pedestrian", "car")
    assert resolve_query(DetectorMode.TAXONOMY, classes, "") == classes


def test_resolve_query_free_text_requires_a_phrase():
    from gdp.demo.app import resolve_query

    with pytest.raises(ValueError):
        resolve_query(DetectorMode.FREE_TEXT, ("car",), "   ")


def test_resolve_query_free_text_returns_the_phrase():
    from gdp.demo.app import resolve_query

    assert resolve_query(DetectorMode.FREE_TEXT, ("car",), "traffic cone") == ("traffic cone",)


def test_resolve_scene_path_prefers_upload_over_dropdown():
    from gdp.demo.app import resolve_scene_path

    scenes = [Scene(name="scene_000", path=FIXTURE_SCENE, is_synthetic=True)]
    path = resolve_scene_path(scenes, "scene_000", "/tmp/upload.jpg")
    assert str(path) == "/tmp/upload.jpg"


def test_resolve_scene_path_falls_back_to_dropdown():
    from gdp.demo.app import resolve_scene_path

    scenes = [Scene(name="scene_000", path=FIXTURE_SCENE, is_synthetic=True)]
    path = resolve_scene_path(scenes, "scene_000", None)
    assert path == FIXTURE_SCENE


def test_resolve_scene_path_raises_when_neither_given():
    from gdp.demo.app import resolve_scene_path

    with pytest.raises(ValueError):
        resolve_scene_path([], None, None)


def test_mode_note_flags_free_text_as_unmeasured():
    from gdp.demo.app import mode_note

    text = mode_note(DetectorMode.FREE_TEXT)
    assert "H7" in text
    assert "frozen prompt" in text


def test_build_app_with_stub_backends_returns_blocks():
    pytest.importorskip("gradio")  # optional `demo` extra — a bare `uv sync` does not have it
    cfg = load_config("configs/default.yaml", "configs/demo.yaml")
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    from gdp.demo.app import build_app

    demo = build_app(cfg, registry=registry, vlm=_StubVLM())
    assert demo is not None
    assert type(demo).__name__ == "Blocks"


def test_build_app_weights_textbox_reflects_the_actual_vlm_passed_in():
    """H8 regression: the "Weights" field must come from `vlm.weights_badge()`, not a hardcoded
    `None` — a fine-tuned adapter passed via `--adapter` must not be silently badged as base."""
    pytest.importorskip("gradio")  # optional `demo` extra — a bare `uv sync` does not have it
    cfg = load_config("configs/default.yaml", "configs/demo.yaml")
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    from gdp.demo.app import build_app

    adapter_badge = "adapter: runs/07-finetune-vlm/x/adapter"
    demo = build_app(cfg, registry=registry, vlm=_StubVLM(weights_badge_value=adapter_badge))
    weights_values = [
        block.value for block in demo.blocks.values() if getattr(block, "label", None) == "Weights"
    ]
    assert weights_values == [adapter_badge]


@pytest.mark.model_heavy
def test_run_detect_real_zero_shot_forward_pass():
    from gdp.demo.app import run_detect

    cfg = load_config("configs/default.yaml", "configs/demo.yaml")
    registry = ModelRegistry(DetectorConfig(), device="cpu")
    scenes = load_scenes(cfg.demo.scenes_dir_path())

    image, latency, ckpt_badge, scn_badge, note, result = run_detect(
        cfg,
        registry,
        scenes,
        scene_name="scene_000",
        upload_path=None,
        mode=DetectorMode.TAXONOMY,
        free_text="",
        variant="pt",
        box_threshold=cfg.demo.threshold_default,
    )
    assert image.size[0] > 0
    assert "measured, this request" in latency
    assert ckpt_badge == ZEROSHOT_WEIGHTS
    assert result.from_cache is False
