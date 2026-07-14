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
from gdp.data.stats import build_stats, write_stats
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


@app.command()
def detect(config: ConfigOpt = None) -> None:
    """Run the open-vocabulary detector on images with text queries (Stage 1)."""
    _pending("02-zeroshot-baseline", "Grounding-DINO inference")


@app.command()
def evaluate(config: ConfigOpt = None) -> None:
    """Evaluate detection mAP / grounding accuracy against a baseline (H8)."""
    _pending("02-zeroshot-baseline", "detection evaluation (mAP, grounding accuracy)")


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
