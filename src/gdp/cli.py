"""The `gdp` command line.

The full command surface is declared here from day one, but only what a *completed spec* has
implemented actually runs. Every other command exits with a pointer to the spec that will fill it
in. The skeleton is therefore an honest roadmap: `gdp --help` tells you exactly how far the
project has got, and nothing pretends to work before it does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gdp import __version__
from gdp.config import Config, load_config
from gdp.data.bdd100k import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, convert_bdd_to_coco
from gdp.data.core import load_dataset
from gdp.data.stats import build_stats, write_stats
from gdp.detect.detector import GroundingDinoDetector
from gdp.detect.predictions import write_predictions
from gdp.eval.coco_map import evaluate_coco_map
from gdp.eval.metrics_io import build_metrics, write_metrics
from gdp.eval.operating_point import GtBox, PredBox, match_operating_point, sweep_threshold
from gdp.logging import get_logger
from gdp.paths import repo_root, resolve, run_dir
from gdp.seed import select_device, set_seed

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
) -> None:
    """Run the pretrained, unmodified detector on images with text queries (Stage 1, spec 02, H1).

    Writes `runs/02-zeroshot/<timestamp>/predictions.json` — COCO-detection-result format
    (absolute xywh boxes + score + image_id + category_id), the input `gdp evaluate` scores
    against ground truth. `--dataset mini_bdd` runs the identical code path on the synthetic
    fixture: an offline smoke path whose output is never a reported metric (H7).
    """
    if dataset not in _VALID_DATASETS:
        _die(f"unknown --dataset {dataset!r}; expected one of {_VALID_DATASETS}")
    if split not in ("train", "val"):
        _die(f"unknown --split {split!r}; expected 'train' or 'val'")
    if dataset == "mini_bdd" and split != "val":
        _die("the mini_bdd fixture only provides a 'val' split")

    cfg = _load(config)
    set_seed(cfg.seed)

    annotations, images_root = _detect_dataset_paths(cfg, dataset, split)
    if not annotations.is_file():
        _die(f"annotations not found: {annotations}")

    ds = load_dataset(annotations, images_root)
    samples = ds.samples[:limit] if limit else ds.samples
    threshold = cfg.detector.box_threshold if box_threshold is None else box_threshold

    detector = GroundingDinoDetector(cfg.detector, cfg.dataset.classes, device=cfg.device)
    detections = detector.detect_images(samples, box_threshold=threshold)

    out_dir = run_dir("02-zeroshot")
    predictions_path = out_dir / "predictions.json"
    write_predictions(detections, predictions_path)
    typer.echo(
        f"{dataset}/{split}: {len(samples)} images, {len(detections)} detections "
        f"(box_threshold={threshold}) -> {predictions_path}"
    )


def _latest_predictions_path() -> Path | None:
    runs_dir = resolve("runs/02-zeroshot")
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
) -> None:
    """Score detections against ground truth: mAP, per-class AP, operating P/R (spec 02, H8).

    Writes `runs/02-zeroshot/<timestamp>/metrics.json` with full provenance (config, checkpoint,
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

    predictions_path = resolve(predictions) if predictions else _latest_predictions_path()
    if predictions_path is None or not predictions_path.is_file():
        _die(f"predictions not found: {predictions_path or '(none) — run `gdp detect` first'}")

    ds = load_dataset(annotations, images_root)
    coco_result = evaluate_coco_map(annotations, predictions_path)

    raw_predictions = json.loads(predictions_path.read_text())
    pred_boxes = [
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
    gt_boxes = [
        GtBox(
            image_id=sample.image_id, class_id=box.class_id, xyxy=(box.x0, box.y0, box.x1, box.y1)
        )
        for sample in ds.samples
        for box in sample.boxes
    ]

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
        cfg=cfg,
        dataset=dataset,
        split=split,
        num_images=len(ds.samples),
        is_synthetic=(dataset == "mini_bdd"),
    )
    out_dir = run_dir("02-zeroshot")
    metrics_path = write_metrics(metrics, out_dir=out_dir)
    typer.echo(
        f"{dataset}/{split}: mAP={coco_result.map:.4f} mAP50={coco_result.map50:.4f} "
        f"P={overall_pr.precision:.3f} R={overall_pr.recall:.3f} (threshold={threshold}) "
        f"-> {metrics_path}"
    )


@app.command()
def deploy(config: ConfigOpt = None) -> None:
    """Quantize, export to ONNX, and benchmark latency/FPS (Stage 1 deployment)."""
    _pending("05-deploy-onnx-edge", "ONNX export + INT8 + latency benchmark")


@app.command()
def vqa(config: ConfigOpt = None) -> None:
    """Ask a question about a driving scene (Stage 2 — a separate model from `detect`, H6)."""
    _pending("07-finetune-vlm", "Qwen2.5-VL driving-scene VQA")


@app.command()
def demo(config: ConfigOpt = None) -> None:
    """Launch the integrated Gradio demo (query→boxes, question→answer)."""
    _pending("09-integrated-demo", "integrated Gradio demo")


app.add_typer(data_app, name="data")


if __name__ == "__main__":
    app()
