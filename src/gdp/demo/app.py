"""The three-panel Gradio layout: query→boxes (Stage 1), question→answer (Stage 2), and the
attributed scene report (presentation layer).

`gradio` is imported **lazily inside** `build_app` (plan design decision 3) so `gdp --help` and
the rest of the CLI import instantly and without the optional `demo` dependency group installed.
Every callback body below is a one-line delegation to a pure, plain-typed helper function defined
at module level — this is what makes the panel logic testable without Gradio at all (task 8's
`tests/test_demo_app.py` exercises the helpers directly; only one test builds the real `gr.Blocks`).

Panels are physically separate components with no shared input box (H6): there is no single
"ask anything" field that could route to both models. `SEPARATE_MODELS` is rendered above both
detection and VQA panels, unconditionally.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from gdp.config import Config, DetectorConfig
from gdp.demo.backends import (
    DetectorMode,
    DetectResult,
    ModelRegistry,
    VLMBackend,
    filter_detections,
)
from gdp.demo.honesty import (
    BASE_VLM_WEIGHTS,
    HEURISTIC_BADGE,
    SEPARATE_MODELS,
    UNMEASURED_QUERY,
    ZEROSHOT_WEIGHTS,
)
from gdp.demo.overlay import draw_boxes
from gdp.demo.report import Attribution, ReportLine, compose_report, summarize_detections
from gdp.demo.risk import heuristic_highlights
from gdp.demo.scenes import Scene, load_scenes, scene_badge
from gdp.deploy.bench import collect_hardware_info

FREE_TEXT_ONNX_NOTE = (
    "INT8 ONNX is exported with a frozen prompt (spec 05) — free-text queries run the PyTorch "
    "detector."
)


# ---------------------------------------------------------------------------
# Pure helpers — plain types in, plain types out. No Gradio import here.
# ---------------------------------------------------------------------------


def checkpoint_badge(model_id: str, *, default_model_id: str) -> str:
    """H8: the UI names the checkpoint actually loaded. A model id equal to the repo's stock
    zero-shot default is badged as such; anything else is named verbatim, never assumed
    fine-tuned just because it differs."""
    if model_id == default_model_id:
        return ZEROSHOT_WEIGHTS
    return f"checkpoint: {model_id}"


def vlm_weights_badge(adapter_dir: str | Path | None) -> str:
    if adapter_dir is None:
        return BASE_VLM_WEIGHTS
    return f"adapter: {adapter_dir}"


def resolve_query(
    mode: DetectorMode, taxonomy_classes: Sequence[str], free_text: str
) -> tuple[str, ...]:
    if mode == DetectorMode.FREE_TEXT:
        phrase = (free_text or "").strip()
        if not phrase:
            raise ValueError("Free-text mode needs a phrase to search for.")
        return (phrase,)
    return tuple(taxonomy_classes)


def resolve_scene_path(
    scenes: Sequence[Scene], scene_name: str | None, upload_path: str | Path | None
) -> Path:
    """An upload always wins over the dropdown — it is the more specific, more recent choice."""
    if upload_path:
        return Path(upload_path)
    if scene_name:
        for scene in scenes:
            if scene.name == scene_name:
                return scene.path
    raise ValueError("Select a scene from the dropdown or upload an image.")


def mode_note(mode: DetectorMode) -> str:
    """What the UI shows explaining the visible free-text/taxonomy fork (decision 4) — never a
    silent fallback."""
    if mode == DetectorMode.FREE_TEXT:
        return f"{FREE_TEXT_ONNX_NOTE} {UNMEASURED_QUERY}"
    return "taxonomy — the evaluated 10-class BDD100K prompt"


def run_detect(
    cfg: Config,
    registry: ModelRegistry,
    scenes: Sequence[Scene],
    *,
    scene_name: str | None,
    upload_path: str | Path | None,
    mode: DetectorMode,
    free_text: str,
    variant: str,
    box_threshold: float,
) -> tuple[Image.Image, str, str, str, str, DetectResult]:
    """The `Detect` button's full body: one real (or cached) forward pass.

    Returns `(overlay_image, latency_label, checkpoint_badge, scene_badge_text, mode_note_text,
    result)` — `result` is stashed in a `gr.State` so the threshold slider can re-filter it
    without calling this function again (decision 6).
    """
    path = resolve_scene_path(scenes, scene_name, upload_path)
    query = resolve_query(mode, cfg.dataset.classes, free_text)

    onnx_path = cfg.demo.onnx_path
    result = _detect_scene_import()(
        registry, path, mode=mode, query=query, variant=variant, onnx_path=onnx_path
    )
    filtered = filter_detections(result, box_threshold=box_threshold)

    with Image.open(path) as img:
        image = draw_boxes(img, filtered, list(result.classes))

    matching_scene = next((s for s in scenes if s.path == path), None)
    scene_badge_text = (
        scene_badge(matching_scene)
        if matching_scene
        else "uploaded image — provenance not verified"
    )

    return (
        image,
        result.latency_label(),
        checkpoint_badge(result.checkpoint, default_model_id=DetectorConfig().model_id),
        scene_badge_text,
        mode_note(mode),
        result,
    )


def _detect_scene_import():
    # Imported lazily via a function (not at module top) purely so `run_detect`'s signature stays
    # readable above the import block — `gdp.demo.backends.detect_scene` has no heavy import cost
    # itself, this is just ordering, not a lazy-load requirement like Gradio's.
    from gdp.demo.backends import detect_scene

    return detect_scene


def rethreshold(
    result: DetectResult | None, box_threshold: float, image_path: str | Path | None
) -> tuple[Image.Image | None, str]:
    """The threshold slider's full body: re-filter the cached `DetectResult` — no new inference."""
    if result is None or image_path is None:
        return None, "move the threshold after running Detect at least once"
    filtered = filter_detections(result, box_threshold=box_threshold)
    with Image.open(image_path) as img:
        image = draw_boxes(img, filtered, list(result.classes))
    return image, result.latency_label()


def _safe_report_line(text: str, source: Attribution) -> ReportLine:
    """A VLM answer that happens to read as a driving command must not crash the whole report
    (H5) — it is replaced with a line that says so, still attributed and still guard-clean."""
    try:
        return ReportLine(text, source)
    except ValueError:
        return ReportLine("answer withheld — first word read as a driving command (H5)", source)


def run_report(
    cfg: Config,
    registry: ModelRegistry,
    vlm: VLMBackend,
    scenes: Sequence[Scene],
    *,
    scene_name: str | None,
    upload_path: str | Path | None,
    box_threshold: float,
) -> tuple[str, Image.Image]:
    """The `Generate report` button's full body (decision 9): Stage-1 counts + Stage-2 answers +
    the badged heuristic highlight, composed by `compose_report` — the only function allowed to
    build the final display text.
    """
    path = resolve_scene_path(scenes, scene_name, upload_path)
    detect_scene = _detect_scene_import()
    result = detect_scene(
        registry,
        path,
        mode=DetectorMode.TAXONOMY,
        query=cfg.dataset.classes,
        variant=cfg.demo.detector_variant,
        onnx_path=cfg.demo.onnx_path,
    )
    filtered = filter_detections(result, box_threshold=box_threshold)
    det_line = summarize_detections(filtered, cfg.dataset.classes)

    with Image.open(path) as img:
        width, height = img.size
    highlighted = heuristic_highlights(filtered, width=width, height=height)
    highlight_indices = {filtered.index(d) for d in highlighted}
    if highlighted:
        heuristic_text = f"{len(highlighted)} detection(s) roughly ahead of the ego vehicle."
    else:
        heuristic_text = "Nothing highlighted as roughly ahead of the ego vehicle."
    heuristic_line = ReportLine(heuristic_text, Attribution.HEURISTIC)

    vlm_lines = []
    for question in cfg.demo.report_questions[: cfg.demo.max_report_lines]:
        text, _seconds = vlm.answer(path, question)
        vlm_lines.append(_safe_report_line(text, Attribution.VLM))

    report_text = compose_report([det_line], vlm_lines, [heuristic_line])

    with Image.open(path) as img:
        overlay = draw_boxes(img, filtered, list(result.classes), highlight=highlight_indices)

    return report_text, overlay


def footer_text(cfg: Config) -> str:
    """H7/decision 10: file paths, not numbers. Never parses a metrics.json for a headline
    figure — links to the artifact the number actually lives in."""
    hardware = collect_hardware_info(batch_size=1, image_size=(640, 360))
    lines = [
        f"hardware: {hardware['processor']}, {hardware['thread_count']} threads, "
        f"torch {hardware['torch_version']}, onnxruntime {hardware['onnxruntime_version']}",
        "accuracy-vs-latency curve (spec 05): runs/05-deploy/<timestamp>/curve.png",
        "VQA base-vs-fine-tuned metrics (spec 08): runs/08-vqa/<timestamp>/metrics.json",
        "grounding accuracy (spec 04, self-built set — H9): "
        "runs/04-grounding/<timestamp>/metrics.json",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio wiring
# ---------------------------------------------------------------------------


def build_app(cfg: Config, *, registry: ModelRegistry, vlm: VLMBackend) -> Any:
    """Wire the three panels. `gradio` is imported here, not at module scope (decision 3)."""
    import gradio as gr

    scenes = load_scenes(cfg.demo.scenes_dir_path())
    scene_names = [s.name for s in scenes]
    default_scene = scene_names[0] if scene_names else None

    with gr.Blocks(title="Grounded Driving Perception — demo") as demo:
        gr.Markdown(f"**{SEPARATE_MODELS}**")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Stage 1 — Grounding-DINO (boxes)")
                scene_dd_1 = gr.Dropdown(choices=scene_names, value=default_scene, label="Scene")
                upload_1 = gr.Image(type="filepath", label="or upload an image")
                mode_radio = gr.Radio(
                    choices=[DetectorMode.TAXONOMY.value, DetectorMode.FREE_TEXT.value],
                    value=DetectorMode.TAXONOMY.value,
                    label="Query mode",
                )
                free_text_box = gr.Textbox(label="Free-text phrase (free-text mode only)")
                variant_dd = gr.Dropdown(
                    choices=["pt", "int8-onnx"], value=cfg.demo.detector_variant, label="Variant"
                )
                threshold_slider = gr.Slider(
                    minimum=cfg.demo.threshold_min,
                    maximum=cfg.demo.threshold_max,
                    value=cfg.demo.threshold_default,
                    label="Box threshold",
                )
                detect_btn = gr.Button("Detect")
                detect_image = gr.Image(label="Detections")
                latency_1 = gr.Textbox(label="Latency", interactive=False)
                checkpoint_badge_1 = gr.Textbox(label="Checkpoint", interactive=False)
                scene_badge_1 = gr.Textbox(label="Scene provenance", interactive=False)
                mode_note_1 = gr.Textbox(label="Mode note", interactive=False)
                detect_state = gr.State(None)
                detect_path_state = gr.State(None)

            with gr.Column():
                gr.Markdown("## Stage 2 — Qwen2.5-VL (text)")
                scene_dd_2 = gr.Dropdown(choices=scene_names, value=default_scene, label="Scene")
                upload_2 = gr.Image(type="filepath", label="or upload an image")
                question_box = gr.Textbox(label="Question")
                ask_btn = gr.Button("Ask")
                answer_box = gr.Textbox(label="Answer", interactive=False)
                latency_2 = gr.Textbox(label="Latency", interactive=False)
                gr.Textbox(label="Backend", value=vlm.badge(), interactive=False)
                gr.Textbox(label="Weights", value=vlm.weights_badge(), interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Scene report (qualitative demo)")
                scene_dd_3 = gr.Dropdown(choices=scene_names, value=default_scene, label="Scene")
                upload_3 = gr.Image(type="filepath", label="or upload an image")
                report_btn = gr.Button("Generate report")
                report_box = gr.Textbox(label="Report", lines=12, interactive=False)
                report_image = gr.Image(label=f"Attention overlay ({HEURISTIC_BADGE})")

        gr.Markdown(footer_text(cfg))

        def _on_detect(scene_name, upload, mode_value, free_text, variant, threshold):
            image, latency, ckpt_badge, scn_badge, note, result = run_detect(
                cfg,
                registry,
                scenes,
                scene_name=scene_name,
                upload_path=upload,
                mode=DetectorMode(mode_value),
                free_text=free_text,
                variant=variant,
                box_threshold=threshold,
            )
            path = resolve_scene_path(scenes, scene_name, upload)
            return image, latency, ckpt_badge, scn_badge, note, result, str(path)

        detect_btn.click(
            _on_detect,
            inputs=[scene_dd_1, upload_1, mode_radio, free_text_box, variant_dd, threshold_slider],
            outputs=[
                detect_image,
                latency_1,
                checkpoint_badge_1,
                scene_badge_1,
                mode_note_1,
                detect_state,
                detect_path_state,
            ],
        )

        def _on_threshold_change(threshold, result, path):
            return rethreshold(result, threshold, path)

        threshold_slider.release(
            _on_threshold_change,
            inputs=[threshold_slider, detect_state, detect_path_state],
            outputs=[detect_image, latency_1],
        )

        def _on_ask(scene_name, upload, question):
            path = resolve_scene_path(scenes, scene_name, upload)
            text, seconds = vlm.answer(path, question)
            return text, f"VLM: {seconds * 1000:.0f} ms (measured, this request)"

        ask_btn.click(
            _on_ask, inputs=[scene_dd_2, upload_2, question_box], outputs=[answer_box, latency_2]
        )

        def _on_report(scene_name, upload, threshold):
            return run_report(
                cfg,
                registry,
                vlm,
                scenes,
                scene_name=scene_name,
                upload_path=upload,
                box_threshold=threshold,
            )

        report_btn.click(
            _on_report,
            inputs=[scene_dd_3, upload_3, threshold_slider],
            outputs=[report_box, report_image],
        )

    return demo


def launch(cfg: Config, *, registry: ModelRegistry, vlm: VLMBackend, **launch_kwargs: Any) -> None:
    """Build and serve the app. `**launch_kwargs` forwards straight to `gr.Blocks.launch`
    (`server_port`, `share`, ...) — no re-implementation of Gradio's own launch surface."""
    demo = build_app(cfg, registry=registry, vlm=vlm)
    demo.launch(**launch_kwargs)
