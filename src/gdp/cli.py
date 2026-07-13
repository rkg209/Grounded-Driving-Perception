"""The `gdp` command line.

The full command surface is declared here from day one, but only what a *completed spec* has
implemented actually runs. Every other command exits with a pointer to the spec that will fill it
in. The skeleton is therefore an honest roadmap: `gdp --help` tells you exactly how far the
project has got, and nothing pretends to work before it does.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from gdp import __version__
from gdp.config import Config, load_config
from gdp.logging import get_logger
from gdp.paths import repo_root
from gdp.seed import select_device, set_seed

app = typer.Typer(
    name="gdp",
    help="Grounded Driving Perception — open-vocabulary detection + driving-scene VQA.",
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


@app.command()
def data(config: ConfigOpt = None) -> None:
    """Prepare BDD100K / DriveLM into the project's canonical format."""
    _pending("01-data-bdd100k", "dataset preparation")


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


if __name__ == "__main__":
    app()
