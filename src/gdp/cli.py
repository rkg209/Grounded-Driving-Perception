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
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from gdp import __version__
from gdp.config import Config, load_config
from gdp.data.bdd100k import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, convert_bdd_to_coco
from gdp.data.core import load_dataset
from gdp.data.stats import build_stats, write_stats
from gdp.detect.detector import GroundingDinoDetector
from gdp.detect.predictions import write_predictions
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
from gdp.paths import repo_root, resolve, run_dir
from gdp.probe.openvocab import load_probe_images, load_probe_phrases, run_probe, write_probe_result
from gdp.seed import select_device, set_seed
from gdp.train.dataset import DetectionCollator, SampleDataset
from gdp.train.trainer import DetectorTrainer

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
    coco_result = evaluate_coco_map(annotations, predictions_path)

    pred_boxes = _load_pred_boxes(predictions_path)
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
        typer.Option("--resume", help="A checkpoint-<step> dir to restore step/optimizer from."),
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
    model = GroundingDinoForObjectDetection.from_pretrained(
        cfg.detector.model_id, disable_custom_kernels=cfg.detector.disable_custom_kernels
    )

    out_dir = run_dir("03-finetune")
    trainer = DetectorTrainer(model, processor, cfg.training, out_dir, device=device)
    if resume:
        trainer.resume(resume)

    collator = DetectionCollator(processor, ds.prompt())
    loader = DataLoader(
        SampleDataset(samples),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    steps_per_epoch = max(len(loader), 1)
    total_steps = max_steps or (
        cfg.training.overfit_max_steps
        if overfit is not None
        else cfg.training.epochs * steps_per_epoch
    )
    trainer.build_scheduler(total_steps)

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
    typer.echo(
        f"{dataset}: {overall['n']} phrases, accuracy={overall['accuracy']:.3f} "
        f"(threshold={threshold}, self-built set — H9) -> {metrics_path}, {per_phrase_path}"
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
app.add_typer(train_app, name="train")
app.add_typer(ground_app, name="ground")


if __name__ == "__main__":
    app()
