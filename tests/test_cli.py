"""The CLI is an honest roadmap: implemented commands work, unimplemented ones name their spec."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from gdp.cli import app
from gdp.paths import resolve

runner = CliRunner()

# command → the spec that will implement it
PENDING = {
    "detect": "02-zeroshot-baseline",
    "evaluate": "02-zeroshot-baseline",
    "deploy": "05-deploy-onnx-edge",
    "vqa": "07-finetune-vlm",
    "demo": "09-integrated-demo",
}


def test_help_lists_the_whole_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["info", "data", *PENDING]:
        assert command in result.stdout


def test_info_is_implemented_and_reports_the_environment():
    result = runner.invoke(app, ["info", "-c", "configs/default.yaml"])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout)
    assert payload["seed"] == 42
    assert payload["num_classes"] == 10
    assert payload["device"] in ("cuda", "mps", "cpu")
    assert payload["detector"] == "IDEA-Research/grounding-dino-tiny"


@pytest.mark.parametrize(("command", "spec"), sorted(PENDING.items()))
def test_unimplemented_commands_fail_loudly_and_name_their_spec(command, spec):
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    output = result.stdout + str(result.stderr or "")
    assert "Not implemented" in output
    assert spec in output


def test_bad_config_path_is_an_error():
    result = runner.invoke(app, ["info", "-c", "configs/nope.yaml"])
    assert result.exit_code != 0


@pytest.fixture
def clean_01_data_runs():
    """`gdp data prepare` writes a fresh timestamped dir per invocation; clean up after."""
    runs_dir = resolve("runs/01-data")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


def test_data_prepare_mini_bdd_writes_coco_json_and_stats(clean_01_data_runs):
    """Acceptance 1 + 4: `gdp data prepare --dataset mini_bdd` exits 0 and writes both the
    converted COCO json and the stats report, loadable by spec 00's `load_dataset`."""
    result = runner.invoke(
        app, ["data", "prepare", "-c", "configs/default.yaml", "--dataset", "mini_bdd"]
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/01-data")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    assert (latest / "det_val_coco.json").is_file()
    assert (latest / "stats.json").is_file()

    stats = json.loads((latest / "stats.json").read_text())
    for name in json.loads((latest / "det_val_coco.json").read_text())["categories"]:
        assert stats["per_class"][name["name"]] > 0


def test_data_prepare_mini_bdd_rejects_non_val_split(clean_01_data_runs):
    result = runner.invoke(app, ["data", "prepare", "--dataset", "mini_bdd", "--split", "train"])
    assert result.exit_code == 1
    assert "'val' split" in (result.stdout + str(result.stderr or ""))


def test_data_prepare_rejects_unknown_dataset(clean_01_data_runs):
    result = runner.invoke(app, ["data", "prepare", "--dataset", "nope"])
    assert result.exit_code == 1
