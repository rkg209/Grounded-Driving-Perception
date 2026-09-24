"""The `gdp` command line.

The full command surface is declared here from day one, but only what a *completed spec* has
implemented actually runs. Every other command exits with a pointer to the spec that will fill it
in. The skeleton is therefore an honest roadmap: `gdp --help` tells you exactly how far the
project has got, and nothing pretends to work before it does.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    Qwen2_5_VLForConditionalGeneration,
)

from gdp import __version__
from gdp.config import DRIVELM_CATEGORIES, Config, load_config
from gdp.data.bdd100k import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, convert_bdd_to_coco
from gdp.data.core import load_dataset
from gdp.data.drivelm import (
    convert_drivelm,
    select_holdout_scenes,
    subsample_scenes,
    write_jsonl,
    write_subsample_json,
)
from gdp.data.drivelm_stats import build_drivelm_stats, split_labels
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.data.stats import build_stats, write_stats
from gdp.deploy.bench import collect_hardware_info, run_interleaved
from gdp.deploy.curve import VariantPoint, plot_accuracy_vs_latency
from gdp.deploy.evaluate import evaluate_variant
from gdp.deploy.export import INPUT_NAMES, OUTPUT_NAMES, ExportResult, export_to_onnx
from gdp.deploy.metrics import build_deploy_metrics, write_deploy_metrics, write_latency
from gdp.deploy.onnx_detector import OnnxDetector
from gdp.deploy.quantize import quantize_dynamic_int8
from gdp.detect.detector import GroundingDinoDetector
from gdp.detect.predictions import (
    read_evaluated_image_ids,
    read_score_floor,
    write_predictions,
    write_predictions_meta,
)
from gdp.eval.coco_map import evaluate_coco_map
from gdp.eval.compare import build_comparison, load_metrics, write_comparison
from gdp.eval.metrics_io import build_metrics, write_metrics
from gdp.eval.operating_point import GtBox, PredBox, match_operating_point, sweep_threshold
from gdp.eval.qualitative import find_miss_to_hit_pairs, save_pair_crops
from gdp.ground.compare import (
    build_grounding_comparison,
    load_grounding_metrics,
    write_grounding_comparison,
)
from gdp.ground.evaluate import (
    aggregate_results,
    build_grounding_metrics,
    score_phrase_set,
    write_grounding_metrics,
    write_per_phrase,
)
from gdp.ground.phrases import assert_frozen, ensure_valid, load_phrases, validate_phrases
from gdp.ground.phrases import freeze as freeze_phrases
from gdp.ground.sample import sample_and_render
from gdp.logging import get_logger
from gdp.paths import git_sha, repo_root, resolve, run_dir
from gdp.probe.openvocab import load_probe_images, load_probe_phrases, run_probe, write_probe_result
from gdp.report import (
    SNAPSHOT_PATH,
    build_snapshot,
    check_blocks,
    inject_blocks,
    load_snapshot,
    render_all,
    write_snapshot,
)
from gdp.seed import select_device, set_seed
from gdp.train.dataset import DetectionCollator, SampleDataset
from gdp.train.trainer import DetectorTrainer
from gdp.vqa.chat import load_jsonl
from gdp.vqa.compare import build_vqa_comparison, write_vqa_comparison
from gdp.vqa.dataset import DriveLMVQADataset, VQACollator
from gdp.vqa.failures import sample_failures, write_failures_md
from gdp.vqa.generate import answer as vqa_answer
from gdp.vqa.generate import load_adapter
from gdp.vqa.hallucination import hallucination_stats
from gdp.vqa.labels import processor_pixel_budget
from gdp.vqa.lora import (
    LM_TARGET_RE,
    attach_lora,
    load_lora_for_resume,
    trainable_parameter_summary,
)
from gdp.vqa.official import OfficialScorerUnavailable
from gdp.vqa.predict import load_predictions, predict_split
from gdp.vqa.score import annotate_per_item, build_vqa_metrics, write_per_item, write_vqa_metrics
from gdp.vqa.trainer import VLMTrainer

app = typer.Typer(
    name="gdp",
    help="Grounded Driving Perception — open-vocabulary detection + driving-scene VQA.",
    no_args_is_help=True,
    add_completion=False,
)

data_app = typer.Typer(
    name="data",
    help="Prepare BDD100K / mini_bdd into the project's canonical COCO-style schema.",
    no_args_is_help=True,
    add_completion=False,
)

train_app = typer.Typer(
    name="train",
    help="Fine-tune a pretrained backbone (spec 03) — adaptation, never training from scratch "
    "(H1).",
    no_args_is_help=True,
    add_completion=False,
)

ground_app = typer.Typer(
    name="ground",
    help="Spec 04 — curate and score the self-built grounding phrase set (H9's sanctioned "
    "exception).",
    no_args_is_help=True,
    add_completion=False,
)

deploy_app = typer.Typer(
    name="deploy",
    help="Spec 05 — export to ONNX, quantize to INT8, and benchmark accuracy-vs-latency on the "
    "M4 (the edge target itself, H3).",
    no_args_is_help=True,
    add_completion=False,
)

vqa_app = typer.Typer(
    name="vqa",
    help="Stage 2 — driving-scene VQA with Qwen2.5-VL (a separate model from `detect`, H6).",
    no_args_is_help=True,
    add_completion=False,
)

report_app = typer.Typer(
    name="report",
    help="Spec 10 — snapshot `runs/*/metrics.json` and render the README's tables from it. "
    "No metric in the README is ever typed by hand.",
    no_args_is_help=True,
    add_completion=False,
)

log = get_logger("gdp.cli")

ConfigOpt = Annotated[
    list[str] | None,
    typer.Option("--config", "-c", help="YAML config(s), merged left-to-right."),
]


def _load(configs: list[str] | None) -> Config:
    return load_config(*(configs or ["configs/default.yaml"]))


def _pending(spec: str, what: str) -> None:
    """Exit honestly: this command's spec is not implemented yet."""
    typer.secho(f"Not implemented: {what}", fg=typer.colors.YELLOW, err=True)
    typer.secho(
        f"  Implemented by spec: specs/{spec}.md\n"
        f"  Run the SDD loop:    /plan {spec} → /tasks → /implement → /verify",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def info(config: ConfigOpt = None) -> None:
    """Show resolved config, device and environment. (Implemented: spec 00.)"""
    cfg = _load(config)
    device = select_device(cfg.device)
    set_seed(cfg.seed)
    typer.echo(
        json.dumps(
            {
                "version": __version__,
                "repo_root": str(repo_root()),
                "device": str(device),
                "seed": cfg.seed,
                "dataset": cfg.dataset.name,
                "num_classes": len(cfg.dataset.classes),
                "detector": cfg.detector.model_id,
                "vlm": cfg.vlm.model_id,
            },
            indent=2,
        )
    )


_VALID_DATASETS = ("bdd100k", "mini_bdd")
_VALID_SPLITS = ("train", "val", "both")


def _die(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _load_pred_boxes(predictions_path: Path) -> list[PredBox]:
    """COCO-detection-result JSON (absolute xywh) -> `PredBox`es (absolute xyxy)."""
    raw_predictions = json.loads(predictions_path.read_text())
    return [
        PredBox(
            image_id=p["image_id"],
            class_id=p["category_id"],
            score=p["score"],
            xyxy=(
                p["bbox"][0],
                p["bbox"][1],
                p["bbox"][0] + p["bbox"][2],
                p["bbox"][1] + p["bbox"][3],
            ),
        )
        for p in raw_predictions
    ]


def _raw_frames_path(cfg: Config, dataset: str, split: str) -> Path:
    if dataset == "mini_bdd":
        return resolve(f"tests/fixtures/mini_bdd_raw/det_{split}.json")
    return cfg.dataset.raw_labels_path() / f"det_{split}.json"


def _images_root(cfg: Config, dataset: str, split: str) -> Path:
    if dataset == "mini_bdd":
        return resolve("tests/fixtures/mini_bdd")
    return cfg.dataset.root_path() / split


@data_app.command("prepare")
def data_prepare(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    split: Annotated[str, typer.Option("--split", help="train, val, or both")] = "val",
) -> None:
    """Convert raw BDD `det_20` labels into the project's COCO-style schema (spec 01).

    Writes `det_<split>_coco.json` plus a `stats.json` sanity report (never a metric — H9) to a
    fresh `runs/01-data/<timestamp>/` directory. `--dataset mini_bdd` runs the *same* code path on
    the synthetic fixture — the offline smoke path that needs no download.
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in _VALID_SPLITS:
        _die(f"unknown --split {split!r}; expected one of {_VALID_SPLITS}")
    if dataset == "mini_bdd" and split != "val":
        _die("the mini_bdd fixture only provides a 'val' split")

    cfg = _load(config)
    splits = ["train", "val"] if split == "both" else [split]
    image_width = 640 if dataset == "mini_bdd" else DEFAULT_IMAGE_WIDTH
    image_height = 360 if dataset == "mini_bdd" else DEFAULT_IMAGE_HEIGHT

    out_dir = run_dir("01-data")
    for s in splits:
        raw_path = _raw_frames_path(cfg, dataset, s)
        if not raw_path.is_file():
            _die(f"raw labels not found: {raw_path}")

        frames = json.loads(raw_path.read_text())
        coco, conv_stats = convert_bdd_to_coco(
            frames,
            classes=tuple(cfg.dataset.classes),
            image_width=image_width,
            image_height=image_height,
        )

        coco_path = out_dir / f"det_{s}_coco.json"
        coco_path.write_text(json.dumps(coco, indent=2) + "\n")

        stats = build_stats(
            coco,
            conv_stats,
            split=s,
            source_file=raw_path,
            min_boxes_for_eval=cfg.dataset.min_boxes_for_eval,
        )
        stats_filename = "stats.json" if len(splits) == 1 else f"stats_{s}.json"
        stats_path = write_stats(stats, out_dir=out_dir, filename=stats_filename)

        typer.echo(f"{s}: wrote {coco_path} and {stats_path}")

    images_root = _images_root(cfg, dataset, splits[0])
    typer.echo(f"images root: {images_root}")


_VALID_DRIVELM_DATASETS = ("drivelm", "mini_drivelm")


@data_app.command("prepare-drivelm")
def data_prepare_drivelm(
    config: ConfigOpt = None,
    dataset: Annotated[
        str, typer.Option("--dataset", help="drivelm or mini_drivelm")
    ] = "mini_drivelm",
    split: Annotated[str, typer.Option("--split", help="train, val, or both")] = "both",
    train_fraction: Annotated[
        float, typer.Option("--train-fraction", help="scene-level train subsample fraction")
    ] = 1.0,
) -> None:
    """Convert DriveLM's raw nested JSON into `{train,val}.jsonl` on the official nuScenes
    train/val scene split (spec 06). Writes `stats_{train,val}.json` and, if `--train-fraction`
    subsamples, `subsample.json` to a fresh `runs/06-data/<timestamp>/` directory.

    `--dataset mini_drivelm` runs the same code path on the synthetic fixture — the offline smoke
    path that needs no registration or download. Val is **never** subsampled: `--train-fraction`
    with `--split val` is rejected outright, not silently ignored.
    """
    if dataset not in _VALID_DRIVELM_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DRIVELM_DATASETS}")
    if split not in _VALID_SPLITS:
        _die(f"unknown --split {split!r}; expected one of {_VALID_SPLITS}")
    if split == "val" and train_fraction != 1.0:
        _die("--train-fraction only applies to train; val is never subsampled (H7)")

    cfg = _load(config)
    drivelm_cfg = cfg.drivelm
    is_synthetic = dataset == "mini_drivelm"

    annotations_path = drivelm_cfg.annotations_path()
    if not annotations_path.is_file():
        _die(f"DriveLM annotations not found: {annotations_path}")

    raw = json.loads(annotations_path.read_text())
    scene_meta = load_scene_meta(drivelm_cfg.scene_meta_path())
    official_splits = load_official_splits()
    labels = split_labels(
        drivelm_cfg.val_source,
        holdout_fraction=drivelm_cfg.holdout_fraction,
        holdout_seed=drivelm_cfg.holdout_seed,
    )

    try:
        holdout: list[str] = []
        if drivelm_cfg.val_source == "train_scene_holdout":
            holdout = select_holdout_scenes(
                raw,
                scene_meta=scene_meta,
                splits=official_splits,
                fraction=drivelm_cfg.holdout_fraction,
                seed=drivelm_cfg.holdout_seed,
            )
        train_records, val_records, conv_stats = convert_drivelm(
            raw,
            scene_meta=scene_meta,
            splits=official_splits,
            images_root=drivelm_cfg.nuscenes_root_path(),
            categories=tuple(drivelm_cfg.categories),
            is_synthetic=is_synthetic,
            holdout_scene_tokens=frozenset(holdout),
        )
    except (KeyError, ValueError) as exc:
        _die(str(exc))
    if split in ("val", "both") and not val_records:
        # [SEQ-0147]: an empty val once passed with exit 0. Never again silently.
        _die(
            f"val split is empty (val_source={drivelm_cfg.val_source!r}). Real DriveLM's answered "
            "file is all nuScenes-train; use drivelm.val_source: train_scene_holdout."
        )

    out_dir = run_dir("06-data")
    if holdout:
        name_by_token = scene_meta
        holdout_path = out_dir / "holdout.json"
        holdout_path.write_text(
            json.dumps(
                {
                    "val_source": drivelm_cfg.val_source,
                    "fraction": drivelm_cfg.holdout_fraction,
                    "seed": drivelm_cfg.holdout_seed,
                    "n_scenes": len(holdout),
                    "scene_tokens": holdout,
                    "scene_names": [name_by_token[t] for t in holdout],
                    "caveat": labels["caveat"],
                },
                indent=2,
            )
            + "\n"
        )
        typer.echo(f"holdout: {len(holdout)} scene(s) -> {holdout_path}")
    wanted_splits = ["train", "val"] if split == "both" else [split]

    def _stats_filename(s: str) -> str:
        return "stats.json" if len(wanted_splits) == 1 else f"stats_{s}.json"

    if "train" in wanted_splits:
        if train_fraction != 1.0:
            try:
                subsample_result = subsample_scenes(
                    train_records, fraction=train_fraction, seed=drivelm_cfg.subsample_seed
                )
            except ValueError as exc:
                _die(str(exc))
            selected = set(subsample_result.selected_scene_tokens)
            train_records = [r for r in train_records if r.scene_token in selected]
            subsample_path = write_subsample_json(subsample_result, out_dir / "subsample.json")
            typer.echo(f"train: wrote {subsample_path}")

        train_path = write_jsonl(train_records, out_dir / "train.jsonl")
        train_stats = build_drivelm_stats(
            train_records,
            conv_stats,
            split="train",
            source_file=annotations_path,
            is_synthetic=is_synthetic,
            labels=labels,
        )
        train_stats_path = write_stats(
            train_stats, out_dir=out_dir, filename=_stats_filename("train")
        )
        typer.echo(f"train: wrote {train_path} and {train_stats_path}")

    if "val" in wanted_splits:
        val_path = write_jsonl(val_records, out_dir / "val.jsonl")
        val_stats = build_drivelm_stats(
            val_records,
            conv_stats,
            split="val",
            source_file=annotations_path,
            is_synthetic=is_synthetic,
            labels=labels,
        )
        val_stats_path = write_stats(val_stats, out_dir=out_dir, filename=_stats_filename("val"))
        typer.echo(f"val: wrote {val_path} and {val_stats_path}")


def _detect_dataset_paths(cfg: Config, dataset: str, split: str) -> tuple[Path, Path]:
    """The COCO-style annotations + images-root pair `gdp detect` reads for one split.

    mini_bdd only has a 'val' split (the committed fixture). For real bdd100k the configured
    `dataset.annotations` path names the 'val' file (`det_val_coco.json`, per
    `configs/bdd100k.yaml`) — swap the split name in for train, matching `data prepare`'s own
    `det_<split>_coco.json` output convention.
    """
    if dataset == "mini_bdd":
        return cfg.dataset.annotations_path(), cfg.dataset.root_path()
    annotations = cfg.dataset.annotations_path()
    if split != "val":
        annotations = annotations.with_name(annotations.name.replace("val", split))
    return annotations, cfg.dataset.root_path() / split


@app.command()
def detect(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    split: Annotated[str, typer.Option("--split", help="train or val")] = "val",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Cap the number of images processed.")
    ] = None,
    box_threshold: Annotated[
        float | None,
        typer.Option("--box-threshold", help="Override the config's detector.box_threshold."),
    ] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option(
            "--checkpoint",
            help="Local checkpoint dir (e.g. runs/03-finetune/<ts>/checkpoint-N) overriding "
            "detector.model_id — lets the fine-tuned model reuse this exact code path (spec 03).",
        ),
    ] = None,
    run_spec: Annotated[
        str,
        typer.Option(
            "--run-spec", help="Which runs/<spec>/ directory to write under (spec 03's rescoring)."
        ),
    ] = "02-zeroshot",
) -> None:
    """Run the detector on images with text queries (Stage 1, spec 02/03, H1).

    Writes `runs/<run-spec>/<timestamp>/predictions.json` — COCO-detection-result format
    (absolute xywh boxes + score + image_id + category_id), the input `gdp evaluate` scores
    against ground truth. `--dataset mini_bdd` runs the identical code path on the synthetic
    fixture: an offline smoke path whose output is never a reported metric (H7). `--checkpoint`
    swaps in a fine-tuned local checkpoint while keeping every other step of the pipeline
    byte-identical to the zero-shot run — that identity is what makes the spec 03 mAP delta (H8)
    honest.
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in ("train", "val"):
        _die(f"unknown --split {split!r}; expected 'train' or 'val'")
    if dataset == "mini_bdd" and split != "val":
        _die("the mini_bdd fixture only provides a 'val' split")

    cfg = _load(config)
    if checkpoint is not None:
        cfg.detector.model_id = checkpoint
    set_seed(cfg.seed)

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    ds = load_dataset(annotations, images_root)
    samples = ds.samples[:limit] if limit else ds.samples
    threshold = cfg.detector.box_threshold if box_threshold is None else box_threshold

    detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    detections = detector.detect_images(samples, box_threshold=threshold)

    out_dir = run_dir(run_spec)
    predictions_path = out_dir / "predictions.json"
    write_predictions(detections, predictions_path)
    # Which images this run actually opened — `gdp evaluate` scores against exactly this set, so a
    # `--limit`ed run is never silently scored against the whole split. See write_predictions_meta.
    write_predictions_meta([s.image_id for s in samples], predictions_path, score_floor=threshold)
    typer.echo(
        f"{dataset}/{split}: {len(samples)} images, {len(detections)} detections "
        f"(box_threshold={threshold}) -> {predictions_path}"
    )


def _latest_predictions_path(run_spec: str = "02-zeroshot") -> Path | None:
    runs_dir = resolve(f"runs/{run_spec}")
    if not runs_dir.is_dir():
        return None
    candidates = [
        d / "predictions.json" for d in runs_dir.iterdir() if (d / "predictions.json").is_file()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


@app.command()
def evaluate(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    split: Annotated[str, typer.Option("--split", help="train or val")] = "val",
    predictions: Annotated[
        str | None,
        typer.Option(
            "--predictions",
            help="Path to a predictions.json; defaults to the most recent "
            "runs/02-zeroshot/<ts>/predictions.json.",
        ),
    ] = None,
    box_threshold: Annotated[
        float | None,
        typer.Option(
            "--box-threshold",
            help="Operating threshold for P/R (defaults to detector.box_threshold).",
        ),
    ] = None,
    sweep: Annotated[
        bool,
        typer.Option(
            "--sweep",
            help="Pick the threshold via detector.sweep_candidates instead of a fixed value. "
            "Only valid with --split train — sweeping on val is leakage (H8).",
        ),
    ] = False,
    chosen_on_override: Annotated[
        str | None,
        typer.Option(
            "--chosen-on",
            help="Attest that --box-threshold's value was originally chosen on a train-split "
            "sweep and is being replayed here (e.g. for the val run after task 6's sweep). "
            "Only accepts 'train'; requires --box-threshold.",
        ),
    ] = None,
    run_spec: Annotated[
        str,
        typer.Option(
            "--run-spec",
            help="Which runs/<spec>/ directory to read predictions from and write metrics under "
            "(spec 03 rescores the fine-tuned model under 03-finetune).",
        ),
    ] = "02-zeroshot",
) -> None:
    """Score detections against ground truth: mAP, per-class AP, operating P/R (spec 02/03, H8).

    Writes `runs/<run-spec>/<timestamp>/metrics.json` with full provenance (config, checkpoint,
    split, image count, git SHA, the threshold and how it was chosen) — a metric without that is
    a rumour (CLAUDE.md §8). `--dataset mini_bdd` scores the synthetic fixture: the plumbing
    smoke path, stamped `is_synthetic: true` and never a reported result (H7).
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in ("train", "val"):
        _die(f"unknown --split {split!r}; expected 'train' or 'val'")
    if dataset == "mini_bdd" and split != "val":
        _die("the mini_bdd fixture only provides a 'val' split")
    if sweep and split != "train":
        _die(
            "--sweep must run on --split train, never val — sweeping the threshold on val is "
            "leakage and would poison the H8 baseline"
        )
    if sweep and box_threshold is not None:
        _die("--sweep and --box-threshold are mutually exclusive")
    if chosen_on_override is not None and chosen_on_override != "train":
        _die("--chosen-on only accepts 'train' (attesting the threshold came from a train sweep)")
    if chosen_on_override is not None and box_threshold is None:
        _die("--chosen-on requires --box-threshold — it labels where that value came from")
    if chosen_on_override is not None and sweep:
        _die("--chosen-on is redundant with --sweep, which already stamps chosen_on=train")

    cfg = _load(config)

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    predictions_path = resolve(predictions) if predictions else _latest_predictions_path(run_spec)
    if predictions_path is None or not predictions_path.is_file():
        _die(f"predictions not found: {predictions_path or '(none) — run `gdp detect` first'}")

    ds = load_dataset(annotations, images_root)

    # Score against exactly the images `gdp detect` ran on, not everything in the annotations
    # file. A `--limit`ed detect run scored against the full split turns every unprocessed image's
    # ground truth into a phantom false negative: mAP collapses, recall collapses, and `--sweep`'s
    # F1 argmax is pulled toward a threshold that is too low — all of it silent. `None` means the
    # predictions have no sidecar (hand-written, or from before this was recorded), in which case
    # the only honest assumption is that they cover the whole split, said out loud.
    evaluated_image_ids = read_evaluated_image_ids(predictions_path)
    if evaluated_image_ids is None:
        typer.secho(
            f"no {predictions_path.stem}_meta.json beside {predictions_path.name}; assuming these "
            "predictions cover the whole split. If they came from a `--limit`ed run, this metric "
            "is wrong.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        samples = ds.samples
    else:
        keep = set(evaluated_image_ids)
        samples = [s for s in ds.samples if s.image_id in keep]
        missing = keep - {s.image_id for s in samples}
        if missing:
            _die(
                f"predictions cover {len(missing)} image id(s) absent from {annotations} "
                f"(e.g. {sorted(missing)[:5]}) — predictions and ground truth disagree about "
                "which split this is"
            )

    coco_result = evaluate_coco_map(annotations, predictions_path, image_ids=evaluated_image_ids)

    pred_boxes = _load_pred_boxes(predictions_path)
    gt_boxes = [
        GtBox(
            image_id=sample.image_id, class_id=box.class_id, xyxy=(box.x0, box.y0, box.x1, box.y1)
        )
        for sample in samples
        for box in sample.boxes
    ]

    # mAP above was computed over every saved box (the floor); the threshold only picks the P/R
    # operating point. A threshold (or sweep candidate) under the floor would count boxes that were
    # never saved as misses and understate recall — refuse rather than report it.
    score_floor = read_score_floor(predictions_path)
    if sweep:
        lowest = min(cfg.detector.sweep_candidates)
    else:
        lowest = cfg.detector.box_threshold if box_threshold is None else box_threshold
    if score_floor is not None and lowest < score_floor:
        _die(
            f"operating threshold {lowest} is below the score floor {score_floor} these "
            "predictions were saved at — re-run `gdp detect` with a lower --box-threshold"
        )

    if sweep:
        threshold, _ = sweep_threshold(
            pred_boxes, gt_boxes, candidates=cfg.detector.sweep_candidates
        )
        chosen_on = "train"
    else:
        threshold = cfg.detector.box_threshold if box_threshold is None else box_threshold
        if box_threshold is None:
            chosen_on = "config_default"
        else:
            chosen_on = chosen_on_override or "cli_override"

    kept = [p for p in pred_boxes if p.score >= threshold]
    per_class_pr_by_id, overall_pr = match_operating_point(kept, gt_boxes)
    per_class_pr = {
        cfg.dataset.classes[class_id]: pr for class_id, pr in per_class_pr_by_id.items()
    }

    metrics = build_metrics(
        coco_result=coco_result,
        per_class_pr=per_class_pr,
        overall_pr=overall_pr,
        box_threshold=threshold,
        chosen_on=chosen_on,
        map_score_floor=score_floor,
        cfg=cfg,
        dataset=dataset,
        split=split,
        num_images=len(samples),
        is_synthetic=(dataset == "mini_bdd"),
    )
    out_dir = run_dir(run_spec)
    metrics_path = write_metrics(metrics, out_dir=out_dir)
    typer.echo(
        f"{dataset}/{split}: mAP={coco_result.map:.4f} mAP50={coco_result.map50:.4f} "
        f"P={overall_pr.precision:.3f} R={overall_pr.recall:.3f} (threshold={threshold}) "
        f"-> {metrics_path}"
    )


@train_app.command("detector")
def train_detector(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    split: Annotated[str, typer.Option("--split", help="train or val")] = "train",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Cap the number of training images.")
    ] = None,
    overfit: Annotated[
        int | None,
        typer.Option(
            "--overfit",
            help="Train on a fixed N-image subset and enforce training.overfit_loss_target as an "
            "exit-code gate (design decision 5) — overrides --limit and training.overfit_images.",
        ),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Total optimizer steps. Defaults to training.overfit_max_steps in --overfit "
            "mode, else training.epochs * steps-per-epoch.",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="A checkpoint-<step> dir to restore weights, step, optimizer and schedule from.",
        ),
    ] = None,
) -> None:
    """Fine-tune the detector on BDD100K train (spec 03, H1: adapting a pretrained backbone).

    Writes `runs/03-finetune/<timestamp>/checkpoint-<step>/` (model + processor + trainer_state.pt)
    and `train_log.jsonl`. `--overfit N` is the overfit gate (acceptance 1): trains on N fixed
    images and exits non-zero if the final loss doesn't clear `training.overfit_loss_target` — the
    SLURM job refuses to start the full run if this fails.
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in ("train", "val"):
        _die(f"unknown --split {split!r}; expected 'train' or 'val'")

    cfg = _load(config)
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    if device.type == "mps":
        # CLAUDE.md §4: this M4 cannot fine-tune Grounding-DINO, full stop — verified empirically,
        # not just asserted: MPS's SDPA backend rejects dropout (raised mid-forward the moment
        # model.train() enables it), and forcing attn_implementation="eager" to route around that
        # OOMs at ~20GB on Grounding-DINO's decoder cross-attention. CPU is slow but correct, which
        # is exactly what design decision 5 asks the laptop path to be — never a training device
        # this command silently pretends works.
        typer.secho(
            "device=mps cannot fine-tune Grounding-DINO (CLAUDE.md §4); falling back to cpu.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        device = select_device("cpu")

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    ds = load_dataset(annotations, images_root)
    n_images = overfit if overfit is not None else limit
    samples = ds.samples[:n_images] if n_images else ds.samples

    processor = AutoProcessor.from_pretrained(cfg.detector.model_id)
    # A resume must continue from the checkpoint's *weights*, not just its step and optimizer:
    # `trainer.resume` restores only the latter, so loading the pretrained model here would put
    # step N's optimizer onto untrained weights and silently discard every step before it.
    model = GroundingDinoForObjectDetection.from_pretrained(
        resume or cfg.detector.model_id,
        disable_custom_kernels=cfg.detector.disable_custom_kernels,
    )

    out_dir = run_dir("03-finetune")
    trainer = DetectorTrainer(model, processor, cfg.training, out_dir, device=device)

    collator = DetectionCollator(processor, ds.prompt())
    loader = DataLoader(
        SampleDataset(samples),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.training.num_workers,
    )
    steps_per_epoch = max(len(loader), 1)
    total_steps = max_steps or (
        cfg.training.overfit_max_steps
        if overfit is not None
        else cfg.training.epochs * steps_per_epoch
    )
    trainer.build_scheduler(total_steps)
    # After build_scheduler, never before: the checkpoint's LR-schedule position needs a scheduler
    # to be restored into. `finetune_detector.slurm` resumes on every SLURM preemption, so the
    # wrong order here restarts the cosine warmup each time — see DetectorTrainer.resume.
    if resume:
        trainer.resume(resume)

    final_loss: float | None = None
    while trainer.step < total_steps:
        for batch in loader:
            if trainer.step >= total_steps:
                break
            record = trainer.train_step(batch)
            final_loss = record["loss"]
            if trainer.step % cfg.training.save_every == 0:
                trainer.save_checkpoint()
    ckpt_dir = trainer.save_checkpoint()

    if overfit is not None:
        target = cfg.training.overfit_loss_target
        if final_loss is None or final_loss >= target:
            typer.secho(
                f"overfit gate FAILED: final loss {final_loss} >= target {target}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"overfit gate passed: final loss {final_loss:.4f} < {target} -> {ckpt_dir}")
    else:
        typer.echo(f"trained {trainer.step} steps, final loss {final_loss:.4f} -> {ckpt_dir}")


@train_app.command("vlm")
def train_vlm(
    config: ConfigOpt = None,
    train_jsonl: Annotated[
        str | None,
        typer.Option(
            "--train-jsonl",
            help="DriveLM train.jsonl (defaults to drivelm.out_dir/train.jsonl, spec 06).",
        ),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Cap the number of QA pairs.")
    ] = None,
    overfit: Annotated[
        int | None,
        typer.Option(
            "--overfit",
            help="Train on a fixed N-QA-pair subset and enforce vlm_training.overfit_loss_target "
            "as an exit-code gate (mirrors spec 03 design decision 5) — overrides --limit and "
            "vlm_training.overfit_qa_pairs.",
        ),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Total optimizer steps. Defaults to vlm_training.overfit_max_steps in --overfit "
            "mode, else vlm_training.epochs * steps-per-epoch.",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="A checkpoint-<step> dir to restore weights, step, optimizer and schedule from.",
        ),
    ] = None,
) -> None:
    """LoRA fine-tune Qwen2.5-VL-3B on DriveLM (spec 07, H1: adapting a pretrained VLM).

    Writes `runs/07-finetune-vlm/<timestamp>/checkpoint-<step>/` (LoRA adapter only + processor +
    trainer_state.pt), `train_log.jsonl`, and `train_config.json` (subset size, drop stats, seed,
    the pixel budget read back off the processor, the LoRA target regex, trainable-parameter
    summary, git sha — acceptance criterion 5). `--overfit N` is the overfit gate (acceptance 1).
    **No evaluation happens here** (H2) — spec 08 owns every accuracy number.
    """
    cfg = _load(config)
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    jsonl_path = resolve(train_jsonl) if train_jsonl else cfg.drivelm.out_dir_path() / "train.jsonl"
    if not jsonl_path.is_file():
        _die(f"train jsonl not found: {jsonl_path}")

    records = load_jsonl(jsonl_path)
    n_records = overfit if overfit is not None else limit
    if n_records:
        records = records[:n_records]

    processor_kwargs = {}
    if cfg.vlm.max_pixels is not None:
        processor_kwargs["max_pixels"] = cfg.vlm.max_pixels
    processor = AutoProcessor.from_pretrained(cfg.vlm.model_id, **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.vlm.model_id)

    out_dir = run_dir("07-finetune-vlm")
    # A resume continues from the checkpoint's adapter; `trainer.resume` below restores only
    # step/optimizer/schedule, so a fresh adapter here would silently discard prior training.
    model = load_lora_for_resume(model, resume) if resume else attach_lora(model, cfg)

    dataset = DriveLMVQADataset(
        records,
        processor,
        cfg.drivelm.nuscenes_root_path(),
        num_views=cfg.vlm.num_views,
        max_seq_len=cfg.vlm_training.max_seq_len,
    )
    if len(dataset) == 0:
        _die(f"no usable QA pairs after encoding and filtering (drop_stats={dataset.drop_stats})")

    collator = VQACollator(processor)
    loader = DataLoader(
        dataset, batch_size=cfg.vlm_training.batch_size, shuffle=True, collate_fn=collator
    )

    trainer = VLMTrainer(model, processor, cfg.vlm_training, out_dir, device=device)

    steps_per_epoch = max(len(loader) // cfg.vlm_training.grad_accum, 1)
    total_steps = max_steps or (
        cfg.vlm_training.overfit_max_steps
        if overfit is not None
        else cfg.vlm_training.epochs * steps_per_epoch
    )
    trainer.build_scheduler(total_steps)
    # After build_scheduler, never before — see DetectorTrainer.resume / VLMTrainer.resume.
    if resume:
        trainer.resume(resume)

    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "subset_size": len(dataset),
                "drop_stats": dataset.drop_stats,
                "seed": cfg.seed,
                "max_pixels": processor_pixel_budget(processor),
                "lora_target_regex": LM_TARGET_RE,
                "trainable_parameters": trainable_parameter_summary(trainer.model),
                "total_steps": total_steps,
                "overfit_qa_pairs": overfit,
                "git_sha": git_sha(),
            },
            indent=2,
        )
    )

    final_loss: float | None = None
    last_saved_step = trainer.step
    while trainer.step < total_steps:
        for batch in loader:
            if trainer.step >= total_steps:
                break
            record = trainer.train_step(batch)
            final_loss = record["loss"]
            if trainer.step != last_saved_step and trainer.step % cfg.vlm_training.save_every == 0:
                trainer.save_checkpoint()
                last_saved_step = trainer.step
    ckpt_dir = trainer.save_checkpoint()

    if overfit is not None:
        target = cfg.vlm_training.overfit_loss_target
        if final_loss is None or final_loss >= target:
            typer.secho(
                f"overfit gate FAILED: final loss {final_loss} >= target {target}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"overfit gate passed: final loss {final_loss:.4f} < {target} -> {ckpt_dir}")
    else:
        typer.echo(f"trained {trainer.step} steps, final loss {final_loss:.4f} -> {ckpt_dir}")


@vqa_app.command("generate")
def vqa_generate(
    adapter: Annotated[
        str, typer.Option("--adapter", help="LoRA adapter checkpoint dir from `gdp train vlm`.")
    ],
    config: ConfigOpt = None,
    train_jsonl: Annotated[
        str | None,
        typer.Option(
            "--train-jsonl", help="DriveLM jsonl to pull a fixture record's question from."
        ),
    ] = None,
    index: Annotated[
        int, typer.Option("--index", help="Which record in the jsonl to generate an answer for.")
    ] = 0,
) -> None:
    """Load an adapter and generate one answer — a **demo, not a result** (H2).

    Never scored, never compared to `record.answer` — spec 08 owns every VQA accuracy number. This
    command only proves the adapter loads back and produces coherent text (acceptance criterion 4).
    """
    cfg = _load(config)
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    jsonl_path = resolve(train_jsonl) if train_jsonl else cfg.drivelm.out_dir_path() / "train.jsonl"
    if not jsonl_path.is_file():
        _die(f"jsonl not found: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    if not 0 <= index < len(records):
        _die(f"--index {index} out of range for {len(records)} record(s) in {jsonl_path}")
    record = records[index]

    processor_kwargs = {}
    if cfg.vlm.max_pixels is not None:
        processor_kwargs["max_pixels"] = cfg.vlm.max_pixels
    processor = AutoProcessor.from_pretrained(cfg.vlm.model_id, **processor_kwargs)
    model = load_adapter(cfg.vlm.model_id, adapter, device=device)

    generated = vqa_answer(
        processor,
        model,
        record,
        nuscenes_root=cfg.drivelm.nuscenes_root_path(),
        device=device,
        num_views=cfg.vlm.num_views,
        max_new_tokens=cfg.vlm.max_new_tokens,
    )

    typer.echo("demo, not a result (H2) — never scored against record.answer")
    typer.echo(f"question: {record.question}")
    typer.echo(f"generated: {generated} -> {adapter}")


_VALID_MODEL_ROLES = ("base", "finetuned")


@vqa_app.command("predict")
def vqa_predict(
    val_jsonl: Annotated[
        str, typer.Option("--val-jsonl", help="DriveLM val.jsonl from `gdp data prepare-drivelm`.")
    ],
    model_role: Annotated[
        str, typer.Option("--model-role", help="'base' or 'finetuned' — stamped onto every row.")
    ],
    out: Annotated[
        str, typer.Option("--out", help="predictions.jsonl to append to (resumable, task 3).")
    ],
    config: ConfigOpt = None,
    adapter: Annotated[
        str | None,
        typer.Option("--adapter", help="LoRA adapter dir — required when --model-role finetuned."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Smoke run: only generate the first N not-yet-done items."),
    ] = None,
) -> None:
    """Generate one model's predictions over one split (spec 08 task 3) — GPU/cluster-expensive,
    resumable at item granularity. Never scores anything — `gdp vqa score` owns every number."""
    if model_role not in _VALID_MODEL_ROLES:
        _die(f"--model-role must be one of {_VALID_MODEL_ROLES}, got {model_role!r}")
    if model_role == "finetuned" and not adapter:
        _die("--model-role finetuned requires --adapter")

    cfg = _load(config)
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    records = load_jsonl(resolve(val_jsonl))

    processor_kwargs = {}
    if cfg.vlm.max_pixels is not None:
        processor_kwargs["max_pixels"] = cfg.vlm.max_pixels
    processor = AutoProcessor.from_pretrained(cfg.vlm.model_id, **processor_kwargs)
    if model_role == "finetuned":
        model = load_adapter(cfg.vlm.model_id, adapter, device=device)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.vlm.model_id)
        model.to(device)
        model.eval()

    out_path = resolve(out)
    result_path = predict_split(
        records,
        processor,
        model,
        cfg=cfg.vqa_eval,
        nuscenes_root=cfg.drivelm.nuscenes_root_path(),
        device=device,
        model_role=model_role,
        out_path=out_path,
        limit=limit,
    )
    typer.echo(f"{model_role}: {len(records)} val item(s) in split -> {result_path}")


def _val_split_labels(val_jsonl_path: Path, cfg: Config) -> dict[str, str]:
    """The val split's own provenance/caveat, read from the stats file `prepare-drivelm` wrote
    beside it, so metrics.json says which val it was scored on. Falls back to the config only
    when there is no sidecar (e.g. a hand-built test val.jsonl)."""
    for name in ("stats_val.json", "stats.json"):
        sidecar = val_jsonl_path.parent / name
        if sidecar.is_file():
            stats = json.loads(sidecar.read_text())
            if stats.get("split") == "val":
                return {k: stats[k] for k in ("split_provenance", "official_split", "caveat")}
    d = cfg.drivelm
    return split_labels(
        d.val_source, holdout_fraction=d.holdout_fraction, holdout_seed=d.holdout_seed
    )


def _score_one_model(
    role: str, predictions_path: Path, records: list, cfg: Config, val_jsonl_path: Path
) -> tuple[dict, list[dict]]:
    predictions = load_predictions(predictions_path)
    roles_present = {row["model_role"] for row in predictions}
    if roles_present != {role}:
        _die(
            f"{predictions_path}: expected all rows to have model_role={role!r}, "
            f"found {roles_present}"
        )

    is_synthetic = any(r.is_synthetic for r in records)
    val_labels = _val_split_labels(val_jsonl_path, cfg)
    metrics = build_vqa_metrics(
        model_role=role,
        predictions=predictions,
        records=records,
        cfg=cfg.vqa_eval,
        gdp_cfg=cfg,
        val_jsonl_path=val_jsonl_path,
        caveat=val_labels["caveat"],
        split_provenance=val_labels["split_provenance"],
        is_synthetic=is_synthetic,
    )
    metrics["hallucination"] = hallucination_stats(predictions)
    per_item_rows = annotate_per_item(predictions)
    return metrics, per_item_rows


@vqa_app.command("score")
def vqa_score(
    val_jsonl: Annotated[str, typer.Option("--val-jsonl", help="DriveLM val.jsonl (task 3/5).")],
    config: ConfigOpt = None,
    predictions: Annotated[
        str | None, typer.Option("--predictions", help="Score a single model's predictions.jsonl.")
    ] = None,
    base_predictions: Annotated[
        str | None,
        typer.Option("--base-predictions", help="Base model's predictions.jsonl."),
    ] = None,
    finetuned_predictions: Annotated[
        str | None,
        typer.Option("--finetuned-predictions", help="Fine-tuned model's predictions.jsonl."),
    ] = None,
) -> None:
    """Score `predictions.jsonl` against DriveLM's official split with the vendored scorer
    (spec 08 tasks 4-7) — no metric arithmetic is ours (H2/H9). One `--predictions` writes a
    single-model `metrics.json`; both `--base-predictions`/`--finetuned-predictions` together also
    write `comparison.json` (design decision 8) — the fine-tuned number never ships without the
    base one beside it (H8)."""
    if predictions and (base_predictions or finetuned_predictions):
        _die("--predictions is exclusive with --base-predictions/--finetuned-predictions")
    if bool(base_predictions) != bool(finetuned_predictions):
        _die("--base-predictions and --finetuned-predictions must be given together")
    if not predictions and not base_predictions:
        _die("pass --predictions, or both --base-predictions and --finetuned-predictions")

    cfg = _load(config)
    val_path = resolve(val_jsonl)
    records = load_jsonl(val_path)
    out_dir = run_dir("08-vqa")

    try:
        if predictions:
            preds_path = resolve(predictions)
            role = {row["model_role"] for row in load_predictions(preds_path)}
            if len(role) != 1:
                _die(f"{preds_path}: predictions.jsonl must have a single model_role, found {role}")
            metrics, per_item_rows = _score_one_model(
                role.pop(), preds_path, records, cfg, val_path
            )
            metrics_path = write_vqa_metrics(metrics, out_dir=out_dir)
            per_item_path = write_per_item(per_item_rows, out_dir=out_dir)
            typer.echo(f"per-item: {per_item_path}")
            typer.echo(f"{metrics['model_role']}: {metrics['num_items']} item(s) -> {metrics_path}")
        else:
            base_metrics, base_rows = _score_one_model(
                "base", resolve(base_predictions), records, cfg, val_path
            )
            ft_metrics, ft_rows = _score_one_model(
                "finetuned", resolve(finetuned_predictions), records, cfg, val_path
            )
            combined = {"models": {"base": base_metrics, "finetuned": ft_metrics}}
            metrics_path = write_vqa_metrics(combined, out_dir=out_dir)
            write_per_item(base_rows, out_dir=out_dir, name="per_item_base.jsonl")
            ft_per_item_path = write_per_item(
                ft_rows, out_dir=out_dir, name="per_item_finetuned.jsonl"
            )
            comparison = build_vqa_comparison(base_metrics, ft_metrics)
            comparison_path = write_vqa_comparison(comparison, out_dir=out_dir)
            typer.echo(f"per-item (finetuned): {ft_per_item_path}")
            typer.echo(
                f"accuracy: {comparison['overall_accuracy']['before']} -> "
                f"{comparison['overall_accuracy']['after']}; regressions: "
                f"{comparison['regressions']} -> {comparison_path}"
            )
    except (ValueError, OfficialScorerUnavailable) as exc:
        _die(str(exc))


@vqa_app.command("failures")
def vqa_failures(
    per_item: Annotated[
        str, typer.Option("--per-item", help="Fine-tuned run's per_item.jsonl (from `vqa score`).")
    ],
    config: ConfigOpt = None,
    base_per_item: Annotated[
        str | None,
        typer.Option("--base-per-item", help="Base run's per_item.jsonl, for side-by-side."),
    ] = None,
    val_jsonl: Annotated[
        str | None, typer.Option("--val-jsonl", help="Defaults to drivelm.out_dir/val.jsonl.")
    ] = None,
    error_on: Annotated[
        str | None,
        typer.Option(
            "--error-on", help="Per-item field naming a failure; default finetuned_exact_match."
        ),
    ] = None,
) -> None:
    """Sample `vqa_eval.failure_sample_size` failures, stratified across categories, and scaffold
    `failures.md` for human annotation (spec 08 task 8) — the commentary is never auto-generated
    (H7)."""
    cfg = _load(config)
    val_path = resolve(val_jsonl) if val_jsonl else cfg.drivelm.out_dir_path() / "val.jsonl"
    if not val_path.is_file():
        _die(f"val jsonl not found: {val_path}")
    records = {r.qa_id: r for r in load_jsonl(val_path)}

    ft_rows = {row["qa_id"]: row for row in load_predictions(resolve(per_item))}
    base_rows = (
        {row["qa_id"]: row for row in load_predictions(resolve(base_per_item))}
        if base_per_item
        else {}
    )

    merged = []
    for qa_id, ft_row in ft_rows.items():
        record = records.get(qa_id)
        if record is None:
            continue
        base_row = base_rows.get(qa_id, {})
        merged.append(
            {
                "qa_id": qa_id,
                "category": ft_row["category"],
                "question": ft_row["question"],
                "gt_answer": ft_row["gt_answer"],
                "base_prediction": base_row.get("prediction", ft_row["prediction"]),
                "finetuned_prediction": ft_row["prediction"],
                "finetuned_exact_match": ft_row.get("exact_match"),
                "image_path": record.image_paths["CAM_FRONT"],
            }
        )

    error_on_field = error_on or "finetuned_exact_match"
    has_scores = any(row.get(error_on_field) is not None for row in merged)
    error_rule = (
        f"per-item field {error_on_field!r} where available, else prediction != GT"
        if has_scores
        else "prediction != GT (no per-item score available for this sub-metric)"
    )

    sample = sample_failures(
        merged,
        seed=cfg.seed,
        n=cfg.vqa_eval.failure_sample_size,
        categories=DRIVELM_CATEGORIES,
        error_on=error_on_field,
    )
    out_dir = run_dir("08-vqa")
    path = write_failures_md(
        sample, out_dir=out_dir, images_root=cfg.drivelm.nuscenes_root_path(), error_rule=error_rule
    )
    typer.echo(f"{len(sample)} failure(s) sampled ({error_rule}) -> {path}")


@app.command()
def compare(
    zeroshot: Annotated[
        str, typer.Option("--zeroshot", help="Path to spec 02's zero-shot metrics.json.")
    ],
    finetuned: Annotated[
        str, typer.Option("--finetuned", help="Path to spec 03's fine-tuned metrics.json.")
    ],
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    split: Annotated[str, typer.Option("--split", help="train or val")] = "val",
    zeroshot_predictions: Annotated[
        str | None,
        typer.Option(
            "--zeroshot-predictions", help="Zero-shot predictions.json (required for --dump-pairs)."
        ),
    ] = None,
    finetuned_predictions: Annotated[
        str | None,
        typer.Option(
            "--finetuned-predictions",
            help="Fine-tuned predictions.json (required for --dump-pairs).",
        ),
    ] = None,
    dump_pairs: Annotated[
        int,
        typer.Option(
            "--dump-pairs",
            help="Save up to N zero-shot-miss -> fine-tuned-hit crops (acceptance 5). 0 disables.",
        ),
    ] = 0,
) -> None:
    """The zero-shot -> fine-tuned mAP delta (spec 03 design decision 8, H8).

    Writes `runs/03-finetune/<timestamp>/comparison.json`: `map_zeroshot`, `map_finetuned`,
    `map_delta`, the same for mAP50, a per-class AP before/after/delta table, and a `regressions`
    field listing every class whose AP went down — never only a paragraph someone could omit (H7).
    Refuses to emit unless both inputs agree on dataset/split/num_images/box_threshold.
    `--dump-pairs N` additionally saves up to N ground-truth boxes the zero-shot model missed but
    the fine-tuned model caught, cropped to `runs/03-finetune/<timestamp>/pairs/` — a qualitative
    gallery for the write-up, never a metric of its own.
    """
    zeroshot_metrics = load_metrics(zeroshot)
    finetuned_metrics = load_metrics(finetuned)
    try:
        result = build_comparison(zeroshot_metrics, finetuned_metrics)
    except ValueError as exc:
        _die(str(exc))

    out_dir = run_dir("03-finetune")
    comparison_path = write_comparison(result, out_dir=out_dir)
    typer.echo(
        f"mAP: {result['map_zeroshot']:.4f} -> {result['map_finetuned']:.4f} "
        f"(delta {result['map_delta']:+.4f}); regressions: {result['regressions']} "
        f"-> {comparison_path}"
    )

    if dump_pairs > 0:
        if not zeroshot_predictions or not finetuned_predictions:
            _die("--dump-pairs requires both --zeroshot-predictions and --finetuned-predictions")

        cfg = _load(config)
        annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
        ds = load_dataset(annotations, images_root)
        gt_boxes = [
            GtBox(
                image_id=sample.image_id,
                class_id=box.class_id,
                xyxy=(box.x0, box.y0, box.x1, box.y1),
            )
            for sample in ds.samples
            for box in sample.boxes
        ]
        zeroshot_pred_boxes = _load_pred_boxes(resolve(zeroshot_predictions))
        finetuned_pred_boxes = _load_pred_boxes(resolve(finetuned_predictions))

        pairs = find_miss_to_hit_pairs(
            gt_boxes, zeroshot_pred_boxes, finetuned_pred_boxes, limit=dump_pairs
        )
        image_path_by_id = {sample.image_id: sample.image_path for sample in ds.samples}
        pairs_dir = out_dir / "pairs"
        saved = save_pair_crops(
            pairs, image_path_by_id, list(cfg.dataset.classes), out_dir=pairs_dir
        )
        typer.echo(f"dumped {len(saved)} zero-shot-miss -> fine-tuned-hit crops -> {pairs_dir}")


@app.command("probe-openvocab")
def probe_openvocab(
    config: ConfigOpt = None,
    probe_config: Annotated[
        str,
        typer.Option("--probe-config", help="YAML with the out-of-taxonomy 'phrases' list."),
    ] = "configs/openvocab_probe.yaml",
    checkpoint: Annotated[
        str | None,
        typer.Option("--checkpoint", help="Local checkpoint dir or HF id (defaults to config)."),
    ] = None,
    images_root: Annotated[
        str | None,
        typer.Option("--images-root", help="Image directory to probe (defaults to dataset.root)."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap the number of images probed.")] = 4,
    box_threshold: Annotated[
        float | None,
        typer.Option("--box-threshold", help="Override detector.box_threshold for the probe."),
    ] = None,
) -> None:
    """Open-vocabulary forgetting probe (spec 03 design decision 9). Never a metric (H9).

    Runs a handful of out-of-taxonomy phrases (no BDD100K ground truth exists for them) through
    the *unmodified* detector — same `GroundingDinoDetector`, an arbitrary phrase list standing in
    for `cfg.dataset.classes`. Writes `runs/03-finetune/<timestamp>/probe.json`, stamped
    `is_demo: true, is_metric: false`, plus annotated crops for visual inspection. Run once against
    the zero-shot checkpoint and once against the fine-tuned one; the write-up compares them by eye.
    """
    cfg = _load(config)
    set_seed(cfg.seed)
    if checkpoint is not None:
        cfg.detector.model_id = checkpoint
    threshold = cfg.detector.box_threshold if box_threshold is None else box_threshold

    phrases = load_probe_phrases(probe_config)

    root = resolve(images_root) if images_root else cfg.dataset.root_path()
    if not root.is_dir():
        _die(f"images root not found: {root}")

    samples = load_probe_images(root, limit=limit)
    if not samples:
        _die(f"no images found under {root}")

    out_dir = run_dir("03-finetune")
    result = run_probe(
        cfg.detector, phrases, samples, device=cfg.device, box_threshold=threshold, out_dir=out_dir
    )
    probe_path = write_probe_result(result, out_dir=out_dir)
    typer.echo(
        f"probed {len(samples)} images x {len(phrases)} phrases: {len(result['detections'])} "
        f"detections (demo only, no metric — H9) -> {probe_path}"
    )


def _grounding_annotations(cfg: Config, dataset: str) -> Path:
    """Grounding curation always reads val — the split every phrase's `target_ann_id` refers to."""
    annotations, _ = _detect_dataset_paths(cfg, dataset, "val")
    return annotations


@ground_app.command("sample-frames")
def ground_sample_frames(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    n: Annotated[
        int | None,
        typer.Option("--n", help="Frames to sample; defaults to grounding.num_frames."),
    ] = None,
) -> None:
    """Seeded frame sample + numbered GT overlays (design decision 5) -> `data/grounding_eval/`.

    Frames are sampled *before* any phrase is written — `frames.json`'s `sampled_at` timestamp is
    the evidence that held-out discipline (acceptance 5) was followed, not merely claimed.
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")

    cfg = _load(config)
    annotations, images_root = _detect_dataset_paths(cfg, dataset, "val")
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    ds = load_dataset(annotations, images_root)
    num_frames = n if n is not None else cfg.grounding.num_frames
    overlays_dir = cfg.grounding.overlays_dir_resolved()
    frames_json_path = overlays_dir.parent / "frames.json"

    frame_sample, overlay_paths = sample_and_render(
        ds,
        annotations,
        n=num_frames,
        seed=cfg.seed,
        frames_json_path=frames_json_path,
        overlays_dir=overlays_dir,
    )
    typer.echo(
        f"{dataset}: sampled {len(frame_sample.image_ids)} frames (seed={frame_sample.seed}) "
        f"-> {frames_json_path}; {len(overlay_paths)} overlays -> {overlays_dir}"
    )


@ground_app.command("validate")
def ground_validate(
    phrases: Annotated[str, typer.Option("--phrases", help="Path to a phrases.json to validate.")],
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
) -> None:
    """Every mechanical check this spec's honesty depends on (task 2): orphan `target_ann_id`,
    false negatives, missing qualifier-type coverage, the token-span round-trip. Ambiguous
    near-duplicate targets (design decision 4) are printed as warnings, not errors."""
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")

    cfg = _load(config)
    annotations = _grounding_annotations(cfg, dataset)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    phrase_set = load_phrases(phrases)
    result = validate_phrases(
        phrase_set,
        annotations,
        min_phrases=cfg.grounding.min_phrases,
        qualifier_types=tuple(cfg.grounding.qualifier_types),
    )
    for warning in result.ambiguous:
        typer.secho(f"AMBIGUOUS: {warning}", fg=typer.colors.YELLOW, err=True)
    if not result.ok:
        for error in result.errors:
            typer.secho(f"ERROR: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{phrases}: {len(phrase_set)} phrases valid {result.type_counts}")


@ground_app.command("freeze")
def ground_freeze(
    phrases: Annotated[str, typer.Option("--phrases", help="Path to the phrases.json to freeze.")],
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    author: Annotated[
        str, typer.Option("--author", help="Who authored this phrase set.")
    ] = "rahul",
) -> None:
    """Validate, then hash-freeze `phrases.json` -> `<stem>.lock.json` (design decision 6).

    `gdp ground evaluate` refuses to run against a `phrases.json` that no longer hashes to this
    lock — the mechanical half of "no phrase was edited after seeing a prediction" (acceptance 5).
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")

    cfg = _load(config)
    annotations = _grounding_annotations(cfg, dataset)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    phrase_set = load_phrases(phrases)
    result = validate_phrases(
        phrase_set,
        annotations,
        min_phrases=cfg.grounding.min_phrases,
        qualifier_types=tuple(cfg.grounding.qualifier_types),
    )
    try:
        ensure_valid(result)
    except ValueError as exc:
        _die(str(exc))

    phrases_path = resolve(phrases)
    lock_path = phrases_path.with_name(f"{phrases_path.stem}.lock.json")
    lock = freeze_phrases(phrases_path, lock_path=lock_path, author=author)
    typer.echo(
        f"froze {lock['num_phrases']} phrases {lock['per_type_counts']} "
        f"(sha256={lock['sha256'][:12]}...) -> {lock_path}"
    )


@ground_app.command("evaluate")
def ground_evaluate(
    config: ConfigOpt = None,
    dataset: Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")] = "mini_bdd",
    phrases: Annotated[
        str | None,
        typer.Option("--phrases", help="Path to a frozen phrases.json; defaults to config."),
    ] = None,
    box_threshold: Annotated[
        float | None,
        typer.Option(
            "--box-threshold",
            help="Operating threshold, replayed from spec 02's train sweep (design decision 7) "
            "— never tuned on this set. Defaults to detector.box_threshold.",
        ),
    ] = None,
    chosen_on: Annotated[
        str | None,
        typer.Option(
            "--chosen-on",
            help="Attest --box-threshold came from a train-split sweep. Only accepts 'train'; "
            "requires --box-threshold.",
        ),
    ] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option("--checkpoint", help="Local checkpoint dir overriding detector.model_id."),
    ] = None,
) -> None:
    """Score a frozen phrase set: grounding accuracy per qualifier type (spec § Approach, H8/H9).

    Refuses to run unless `phrases.json` still hashes to its `<stem>.lock.json` (design decision
    6). Writes `runs/04-grounding/<timestamp>/metrics.json` (stamped `is_self_built_benchmark`,
    `is_synthetic`, `phrases_sha256`, `chosen_on`) and `per_phrase.json` (every phrase's outcome,
    for H7 failure inspection). Loads one `GroundingDinoDetector` for the whole run — never one per
    phrase (CLAUDE.md §4's memory-hygiene rule).
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if chosen_on is not None and chosen_on != "train":
        _die("--chosen-on only accepts 'train' (attesting the threshold came from a train sweep)")
    if chosen_on is not None and box_threshold is None:
        _die("--chosen-on requires --box-threshold — it labels where that value came from")

    cfg = _load(config)
    phrases_path = resolve(phrases or cfg.grounding.phrases_path)
    if not phrases_path.is_file():
        _die(f"phrases not found: {phrases_path}")
    lock_path = phrases_path.with_name(f"{phrases_path.stem}.lock.json")
    try:
        lock = assert_frozen(phrases_path, lock_path)
    except RuntimeError as exc:
        _die(str(exc))

    annotations = _grounding_annotations(cfg, dataset)
    _, images_root = _detect_dataset_paths(cfg, dataset, "val")
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    if checkpoint is not None:
        cfg.detector.model_id = checkpoint
    set_seed(cfg.seed)

    ds = load_dataset(annotations, images_root)
    phrase_set = load_phrases(phrases_path)

    threshold = cfg.detector.box_threshold if box_threshold is None else box_threshold
    if box_threshold is None:
        resolved_chosen_on = "config_default"
    else:
        resolved_chosen_on = chosen_on or "cli_override"

    detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    results = score_phrase_set(
        detector,
        ds,
        phrase_set,
        annotations,
        box_threshold=threshold,
        iou_threshold=cfg.grounding.iou_threshold,
    )
    aggregates = aggregate_results(results)
    metrics = build_grounding_metrics(
        aggregates=aggregates,
        lock=lock,
        box_threshold=threshold,
        chosen_on=resolved_chosen_on,
        iou_threshold=cfg.grounding.iou_threshold,
        cfg=cfg,
        dataset=dataset,
        is_synthetic=(dataset == "mini_bdd"),
    )

    out_dir = run_dir("04-grounding")
    metrics_path = write_grounding_metrics(metrics, out_dir=out_dir)
    per_phrase_path = write_per_phrase(results, out_dir=out_dir)
    overall = aggregates["overall"]
    # `scripts/slurm/grounding_eval.slurm`'s run_step extracts this command's output path with
    # `grep -oE -- '-> \S+' | tail -1`, so the LAST `-> ` on stdout must be the metrics path and
    # nothing may follow it on that line. The earlier form ended `-> {metrics}, {per_phrase}` and
    # handed the job a path with a trailing comma, which `ground compare` then failed to open.
    typer.echo(f"per-phrase outcomes (H7 failure inspection): {per_phrase_path}")
    typer.echo(
        f"{dataset}: {overall['n']} phrases, accuracy={overall['accuracy']:.3f} "
        f"(threshold={threshold}, self-built set — H9) -> {metrics_path}"
    )


@ground_app.command("compare")
def ground_compare(
    zeroshot: Annotated[
        str, typer.Option("--zeroshot", help="Path to the zero-shot grounding metrics.json.")
    ],
    finetuned: Annotated[
        str, typer.Option("--finetuned", help="Path to the fine-tuned grounding metrics.json.")
    ],
) -> None:
    """The zero-shot -> fine-tuned grounding-accuracy delta (design decision 9, H8).

    Writes `runs/04-grounding/<timestamp>/grounding_comparison.json`: overall and per-type
    accuracy before/after/delta, a `regressions` list, and the H9 `is_self_built_benchmark`/
    `caveat` labels. Refuses to emit unless both inputs share `phrases_sha256`, `num_phrases`, and
    `box_threshold` — the fine-tuned grounding number never exists in a file without the zero-shot
    one at its side.
    """
    zeroshot_metrics = load_grounding_metrics(zeroshot)
    finetuned_metrics = load_grounding_metrics(finetuned)
    try:
        result = build_grounding_comparison(zeroshot_metrics, finetuned_metrics)
    except ValueError as exc:
        _die(str(exc))

    out_dir = run_dir("04-grounding")
    comparison_path = write_grounding_comparison(result, out_dir=out_dir)
    typer.echo(
        f"grounding accuracy: {result['accuracy_zeroshot']:.3f} -> "
        f"{result['accuracy_finetuned']:.3f} (delta {result['accuracy_delta']:+.3f}); "
        f"regressions: {result['regressions']} -> {comparison_path}"
    )


def _latest_deploy_run_dir(run_spec: str = "05-deploy") -> Path | None:
    runs_dir = resolve(f"runs/{run_spec}")
    if not runs_dir.is_dir():
        return None
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "model.onnx").is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _validate_deploy_dataset_split(dataset: str, split: str) -> None:
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in ("train", "val"):
        _die(f"unknown --split {split!r}; expected 'train' or 'val'")
    if dataset == "mini_bdd" and split != "val":
        _die("the mini_bdd fixture only provides a 'val' split")


DeployDatasetOpt = Annotated[str, typer.Option("--dataset", help="bdd100k or mini_bdd")]
DeploySplitOpt = Annotated[str, typer.Option("--split", help="train or val")]
DeployRunSpecOpt = Annotated[
    str, typer.Option("--run-spec", help="Which runs/<spec>/ directory this pipeline writes under.")
]


@deploy_app.command("export")
def deploy_export(
    config: ConfigOpt = None,
    dataset: DeployDatasetOpt = "mini_bdd",
    split: DeploySplitOpt = "val",
    checkpoint: Annotated[
        str | None,
        typer.Option(
            "--checkpoint",
            help="Local fine-tuned checkpoint dir overriding detector.model_id (spec 03).",
        ),
    ] = None,
    run_spec: DeployRunSpecOpt = "05-deploy",
) -> None:
    """Export the detector to ONNX, frozen to the 10-class prompt (H3: not open-vocabulary).

    Writes `runs/<run-spec>/<timestamp>/model.onnx` and an `export_result.json` sidecar
    (`fixed_prompt`, `export_path`, `prompt`) that `quantize`/`bench`/`evaluate-variants` read
    back — the fixed-prompt limitation travels with the artifact, not just in this command's
    stdout.
    """
    _validate_deploy_dataset_split(dataset, split)

    cfg = _load(config)
    if checkpoint is not None:
        cfg.detector.model_id = checkpoint
    set_seed(cfg.seed)

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")
    ds = load_dataset(annotations, images_root)

    detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    out_dir = run_dir(run_spec)
    export_result = export_to_onnx(
        detector, ds.samples[0], out_dir / "model.onnx", opset=cfg.deploy.onnx_opset
    )

    (out_dir / "export_result.json").write_text(
        json.dumps(
            {
                "onnx_path": str(export_result.onnx_path),
                "export_path": export_result.export_path,
                "fixed_prompt": export_result.fixed_prompt,
                "prompt": export_result.prompt,
                "opset": export_result.opset,
                "dataset": dataset,
                "split": split,
                "num_images": len(ds.samples),
            },
            indent=2,
        )
        + "\n"
    )
    typer.echo(
        f"exported {export_result.onnx_path} (export_path={export_result.export_path}, "
        f"fixed_prompt={export_result.fixed_prompt}) -> {out_dir}"
    )


@deploy_app.command("quantize")
def deploy_quantize(config: ConfigOpt = None, run_spec: DeployRunSpecOpt = "05-deploy") -> None:
    """Dynamic INT8 quantization of the most recent `gdp deploy export` output."""
    cfg = _load(config)
    if cfg.deploy.quantization != "dynamic":
        _die(
            f"deploy.quantization={cfg.deploy.quantization!r} is not implemented — only "
            "'dynamic' is (static INT8 is a documented fallback, not yet built; plan decision 5)"
        )

    out_dir = _latest_deploy_run_dir(run_spec)
    if out_dir is None:
        _die("no exported model.onnx found — run `gdp deploy export` first")

    result = quantize_dynamic_int8(out_dir / "model.onnx", out_dir / "model.int8.onnx")
    typer.echo(
        f"quantized {result.onnx_path} "
        f"({result.fp32_size_bytes} -> {result.int8_size_bytes} bytes) -> {out_dir}"
    )


@deploy_app.command("bench")
def deploy_bench(
    config: ConfigOpt = None,
    dataset: DeployDatasetOpt = "mini_bdd",
    split: DeploySplitOpt = "val",
    run_spec: DeployRunSpecOpt = "05-deploy",
) -> None:
    """Interleaved p50/p95/p99/FPS + peak RSS for all three variants (design decision 6).

    Requires `gdp deploy export` and `gdp deploy quantize` to have already run into the same
    (most recent) `runs/<run-spec>/<timestamp>/` directory.
    """
    _validate_deploy_dataset_split(dataset, split)

    cfg = _load(config)
    out_dir = _latest_deploy_run_dir(run_spec)
    if out_dir is None:
        _die("no exported model.onnx found — run `gdp deploy export` first")
    int8_path = out_dir / "model.int8.onnx"
    if not int8_path.is_file():
        _die("no quantized model.int8.onnx found — run `gdp deploy quantize` first")

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    ds = load_dataset(annotations, images_root)
    sample = ds.samples[0]
    box_threshold = cfg.detector.box_threshold

    pt_detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    fp32_onnx_detector = OnnxDetector(out_dir / "model.onnx", cfg.detector, cfg.dataset.classes)
    int8_onnx_detector = OnnxDetector(int8_path, cfg.detector, cfg.dataset.classes)

    callables = {
        "fp32-pt": lambda: pt_detector.detect_images([sample], box_threshold=box_threshold),
        "fp32-onnx": lambda: fp32_onnx_detector.detect_images(
            [sample], box_threshold=box_threshold
        ),
        "int8-onnx": lambda: int8_onnx_detector.detect_images(
            [sample], box_threshold=box_threshold
        ),
    }
    results = run_interleaved(
        callables, warmup=cfg.deploy.warmup_iters, iters=cfg.deploy.timed_iters
    )
    hardware = collect_hardware_info(batch_size=1, image_size=(sample.height, sample.width))
    latency_path = write_latency(results, hardware=hardware, out_dir=out_dir)

    summary = ", ".join(
        f"{name} p50={r.p50 * 1000:.1f}ms fps={r.fps:.1f}" for name, r in results.items()
    )
    typer.echo(f"latency: {summary} -> {latency_path}")


@deploy_app.command("evaluate-variants")
def deploy_evaluate_variants(
    config: ConfigOpt = None,
    dataset: DeployDatasetOpt = "mini_bdd",
    split: DeploySplitOpt = "val",
    run_spec: DeployRunSpecOpt = "05-deploy",
) -> None:
    """mAP for all three variants + `curve.png` (H8: never a latency number without its accuracy).

    Requires `gdp deploy export` and `gdp deploy quantize` to have already run. If `gdp deploy
    bench` has also run, `curve.png` is produced from its `latency.json`; otherwise only
    `metrics.json` is written and a warning is printed.
    """
    _validate_deploy_dataset_split(dataset, split)

    cfg = _load(config)
    out_dir = _latest_deploy_run_dir(run_spec)
    if out_dir is None:
        _die("no exported model.onnx found — run `gdp deploy export` first")
    int8_path = out_dir / "model.int8.onnx"
    if not int8_path.is_file():
        _die("no quantized model.int8.onnx found — run `gdp deploy quantize` first")
    export_meta_path = out_dir / "export_result.json"
    if not export_meta_path.is_file():
        _die("no export_result.json found — run `gdp deploy export` first")
    export_meta = json.loads(export_meta_path.read_text())

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    ds = load_dataset(annotations, images_root)
    box_threshold = cfg.detector.box_threshold

    pt_detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    fp32_onnx_detector = OnnxDetector(out_dir / "model.onnx", cfg.detector, cfg.dataset.classes)
    int8_onnx_detector = OnnxDetector(int8_path, cfg.detector, cfg.dataset.classes)
    variant_detectors = {
        "fp32-pt": pt_detector,
        "fp32-onnx": fp32_onnx_detector,
        "int8-onnx": int8_onnx_detector,
    }
    variant_results = {
        name: evaluate_variant(
            det,
            ds.samples,
            gt_path=annotations,
            box_threshold=box_threshold,
            out_dir=out_dir,
            variant=name,
        )
        for name, det in variant_detectors.items()
    }

    export_result = ExportResult(
        onnx_path=Path(export_meta["onnx_path"]),
        export_path=export_meta["export_path"],
        fixed_prompt=export_meta["fixed_prompt"],
        prompt=export_meta["prompt"],
        opset=export_meta["opset"],
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )
    metrics = build_deploy_metrics(
        cfg=cfg,
        export_result=export_result,
        quantization=cfg.deploy.quantization,
        variant_results=variant_results,
        dataset=dataset,
        num_images=len(ds.samples),
        is_synthetic=(dataset == "mini_bdd"),
    )
    metrics_path = write_deploy_metrics(metrics, out_dir=out_dir)

    latency_path = out_dir / "latency.json"
    if latency_path.is_file():
        latency = json.loads(latency_path.read_text())
        points = {
            name: VariantPoint(map=variant_results[name].map, p50=latency["variants"][name]["p50"])
            for name in variant_detectors
        }
        curve_path = plot_accuracy_vs_latency(points, out_dir / "curve.png")
        typer.echo(f"curve -> {curve_path}")
    else:
        typer.secho(
            "no latency.json found — run `gdp deploy bench` first for curve.png",
            fg=typer.colors.YELLOW,
            err=True,
        )

    summary = ", ".join(f"{name}={r.map:.3f}" for name, r in variant_results.items())
    typer.echo(f"mAP: {summary} -> {metrics_path}")


@app.command()
def demo(
    config: ConfigOpt = None,
    checkpoint: Annotated[
        str | None,
        typer.Option(
            "--checkpoint",
            help="Fine-tuned detector checkpoint dir (spec 03). Defaults to cfg.detector.model_id "
            "(zero-shot) — the UI badges whichever is actually loaded (H8).",
        ),
    ] = None,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help="LoRA adapter dir from `gdp train vlm` (spec 07). Defaults to base Qwen2.5-VL — "
            "the UI badges whichever is actually loaded (H8).",
        ),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option("--variant", help="Detector variant: 'pt' or 'int8-onnx'. Overrides config."),
    ] = None,
    scenes: Annotated[
        str | None,
        typer.Option("--scenes", help="Directory of demo scene images. Overrides config."),
    ] = None,
    vlm_endpoint: Annotated[
        str | None,
        typer.Option(
            "--vlm-endpoint",
            help="Remote HF-Space-style endpoint for Stage 2 (decision 12). "
            "Default: run Qwen2.5-VL locally, lazy-loaded on the first question.",
        ),
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", help="Server port. Overrides config.")
    ] = None,
    share: Annotated[
        bool, typer.Option("--share", help="Create a public Gradio share link.")
    ] = False,
) -> None:
    """Launch the integrated Gradio demo (query→boxes, question→answer, attributed scene report).

    Two separate models, evaluated separately (H6) — see `SEPARATE_MODELS`, rendered in the UI.
    Runs entirely on the laptop against synthetic fixture scenes by default (CLAUDE.md §4); no
    cluster, no download, unless `--checkpoint`/`--adapter` point at real fine-tuned artifacts.
    """
    from gdp.demo.app import launch
    from gdp.demo.backends import ModelRegistry

    cfg = _load(config)
    set_seed(cfg.seed)

    detector_config = cfg.detector
    if checkpoint is not None:
        detector_config = replace(detector_config, model_id=checkpoint)
    demo_config = cfg.demo
    if variant is not None:
        demo_config = replace(demo_config, detector_variant=variant)
    if scenes is not None:
        demo_config = replace(demo_config, scenes_dir=scenes)
    if vlm_endpoint is not None:
        demo_config = replace(demo_config, vlm_endpoint=vlm_endpoint)
    if port is not None:
        demo_config = replace(demo_config, server_port=port)
    if share:
        demo_config = replace(demo_config, share=True)
    cfg = replace(cfg, detector=detector_config, demo=demo_config)

    registry = ModelRegistry(cfg.detector, device=cfg.device)
    if cfg.demo.vlm_endpoint is not None:
        vlm = registry.get_remote_vlm(cfg.demo.vlm_endpoint)
    else:
        vlm = registry.get_local_vlm(
            cfg.vlm.model_id, adapter_dir=adapter, scenes_root=cfg.demo.scenes_dir_path()
        )

    typer.echo(f"scenes: {cfg.demo.scenes_dir_path()}")
    typer.echo(f"detector: {cfg.detector.model_id} ({cfg.demo.detector_variant})")
    typer.echo(f"VLM backend: {vlm.badge()}")
    launch(cfg, registry=registry, vlm=vlm, server_port=cfg.demo.server_port, share=cfg.demo.share)


@report_app.command("snapshot")
def report_snapshot(
    out: Annotated[
        str | None,
        typer.Option("--out", help=f"Snapshot path (default {SNAPSHOT_PATH})."),
    ] = None,
) -> None:
    """Copy the fields the README's tables need out of `runs/` into a committed snapshot.

    `runs/` is gitignored, so this is the only way a clean clone can check that a published number
    came from a file the code wrote. A stage whose run was computed on `tests/fixtures/` is stamped
    `synthetic` and renders as pending — a fixture number is never a result (H7).
    """
    snapshot = build_snapshot()
    path = write_snapshot(snapshot, path=resolve(out) if out else None)
    for name, entry in snapshot["stages"].items():
        suffix = f" — {entry['reason']}" if entry.get("reason") else f" ({entry['source']})"
        typer.echo(f"  {name}: {entry['status']}{suffix}")
    typer.echo(f"snapshot -> {path}")


@report_app.command("render")
def report_render(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Do not write; exit 1 if the README's generated blocks differ from the snapshot.",
        ),
    ] = False,
    snapshot: Annotated[
        str | None, typer.Option("--snapshot", help=f"Snapshot path (default {SNAPSHOT_PATH}).")
    ] = None,
    readme: Annotated[
        str | None, typer.Option("--readme", help="README path (default README.md).")
    ] = None,
) -> None:
    """Render the snapshot into README.md's `<!-- BEGIN GENERATED: … -->` blocks.

    Reads the snapshot only, never `runs/` — so the README a clean clone renders is the README a
    clean clone has. `--check` is the "generated, not typed" gate: it fails on a hand-edited number.
    """
    readme_path = resolve(readme) if readme else resolve("README.md")
    if not readme_path.is_file():
        _die(f"no README at {readme_path}")

    blocks = render_all(load_snapshot(resolve(snapshot) if snapshot else None))
    text = readme_path.read_text()

    if check:
        drifted = check_blocks(text, blocks)
        if drifted:
            _die(
                "README generated blocks are out of date with docs/metrics_snapshot.json: "
                + ", ".join(drifted)
                + " — run `uv run gdp report render` (never edit a generated block by hand)"
            )
        typer.echo(f"README generated blocks match the snapshot ({len(blocks)} blocks)")
        return

    updated = inject_blocks(text, blocks)
    if updated == text:
        typer.echo(f"README already up to date ({len(blocks)} blocks) -> {readme_path}")
        return
    readme_path.write_text(updated)
    typer.echo(f"rendered {len(blocks)} blocks -> {readme_path}")


app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(ground_app, name="ground")
app.add_typer(deploy_app, name="deploy")
app.add_typer(vqa_app, name="vqa")
app.add_typer(report_app, name="report")


if __name__ == "__main__":
    app()
