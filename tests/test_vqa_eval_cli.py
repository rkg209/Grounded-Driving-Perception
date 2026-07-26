"""Spec 08 task 9: `gdp vqa predict | score | failures` — CLI plumbing + the fixture end-to-end
chain from the plan's Verification section.

The `score`/`failures` round-trip needs `language_evaluation`/`openai` installed (see
`tests/test_vqa_official.py`); it skips itself with a clear reason when they aren't. `predict` is
not exercised here for real generation — that's `tests/test_vqa_predict.py`'s `model_heavy` test —
only its `--help`/argument-validation surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gdp.cli import app
from gdp.config import load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta

runner = CliRunner()


def test_vqa_help_lists_predict_score_failures():
    result = runner.invoke(app, ["vqa", "--help"])
    assert result.exit_code == 0
    for subcommand in ["generate", "predict", "score", "failures"]:
        assert subcommand in result.stdout


def test_vqa_predict_help():
    result = runner.invoke(app, ["vqa", "predict", "--help"])
    assert result.exit_code == 0
    assert "--val-jsonl" in result.stdout
    assert "--model-role" in result.stdout
    assert "--out" in result.stdout


def test_vqa_predict_rejects_bad_model_role(tmp_path):
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("")
    result = runner.invoke(
        app,
        [
            "vqa",
            "predict",
            "--val-jsonl",
            str(val_jsonl),
            "--model-role",
            "nonsense",
            "--out",
            str(tmp_path / "predictions.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "--model-role" in (result.stdout + str(result.stderr or ""))


def test_vqa_predict_finetuned_requires_adapter(tmp_path):
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("")
    result = runner.invoke(
        app,
        [
            "vqa",
            "predict",
            "--val-jsonl",
            str(val_jsonl),
            "--model-role",
            "finetuned",
            "--out",
            str(tmp_path / "predictions.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "--adapter" in (result.stdout + str(result.stderr or ""))


def test_vqa_score_help():
    result = runner.invoke(app, ["vqa", "score", "--help"])
    assert result.exit_code == 0
    assert "--predictions" in result.stdout
    assert "--base-predictions" in result.stdout
    assert "--finetuned-predictions" in result.stdout


def test_vqa_score_rejects_predictions_and_base_together(tmp_path):
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("")
    result = runner.invoke(
        app,
        [
            "vqa",
            "score",
            "--val-jsonl",
            str(val_jsonl),
            "--predictions",
            "x.jsonl",
            "--base-predictions",
            "y.jsonl",
        ],
    )
    assert result.exit_code != 0
    assert "exclusive" in (result.stdout + str(result.stderr or ""))


def test_vqa_score_rejects_base_without_finetuned(tmp_path):
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("")
    result = runner.invoke(
        app,
        ["vqa", "score", "--val-jsonl", str(val_jsonl), "--base-predictions", "y.jsonl"],
    )
    assert result.exit_code != 0
    assert "together" in (result.stdout + str(result.stderr or ""))


def test_vqa_failures_help():
    result = runner.invoke(app, ["vqa", "failures", "--help"])
    assert result.exit_code == 0
    assert "--per-item" in result.stdout
    assert "--base-per-item" in result.stdout


@pytest.fixture(scope="module")
def val_jsonl_path(tmp_path_factory):
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    _, val_records, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    path = tmp_path_factory.mktemp("cli_e2e") / "val.jsonl"
    with path.open("w") as f:
        for record in val_records:
            f.write(json.dumps(record.to_json()) + "\n")
    return path


def test_vqa_score_and_failures_end_to_end_fixture_chain(val_jsonl_path):
    """The plan's own end-to-end verification: `gdp vqa score --base-predictions
    --finetuned-predictions` -> `gdp vqa failures --per-item` on the hand-built fixtures."""
    result = runner.invoke(
        app,
        [
            "vqa",
            "score",
            "-c",
            "configs/default.yaml",
            "-c",
            "configs/vqa_eval.yaml",
            "--val-jsonl",
            str(val_jsonl_path),
            "--base-predictions",
            "tests/fixtures/vqa_predictions/base_mini.jsonl",
            "--finetuned-predictions",
            "tests/fixtures/vqa_predictions/finetuned_mini.jsonl",
        ],
    )
    output = result.stdout + str(result.stderr or "")
    if result.exit_code != 0 and "vendored scorer deps not installed" in output:
        pytest.skip("vendored scorer deps not installed")
    assert result.exit_code == 0, output
    assert "-> " in output.strip().splitlines()[-1]

    # the combined base+finetuned branch's last echoed line is comparison.json, not metrics.json
    # (see `vqa_score`'s `else` branch) — both live in the same `run_dir("08-vqa")` output dir.
    comparison_path = Path(output.strip().splitlines()[-1].rsplit("-> ", 1)[-1])
    metrics_path = comparison_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert "models" in metrics
    assert set(metrics["models"]) == {"base", "finetuned"}
    for role_metrics in metrics["models"].values():
        assert set(role_metrics["per_category"]) == {
            "perception",
            "prediction",
            "planning",
            "behavior",
        }
        assert role_metrics["overall"]["final_score"] is None
        assert role_metrics["comparable_to_leaderboard"] is False
        assert "hallucination" in role_metrics
        assert role_metrics["hallucination"]["caveat"]

    per_item_path = metrics_path.parent / "per_item_finetuned.jsonl"
    assert per_item_path.is_file()

    failures_result = runner.invoke(
        app,
        [
            "vqa",
            "failures",
            "-c",
            "configs/default.yaml",
            "--val-jsonl",
            str(val_jsonl_path),
            "--per-item",
            str(per_item_path),
            "--base-per-item",
            str(metrics_path.parent / "per_item_base.jsonl"),
        ],
    )
    failures_output = failures_result.stdout + str(failures_result.stderr or "")
    assert failures_result.exit_code == 0, failures_output
    failures_md = Path(failures_output.strip().splitlines()[-1].rsplit("-> ", 1)[-1])
    assert failures_md.is_file()
    text = failures_md.read_text()
    assert "Generated fields" in text
