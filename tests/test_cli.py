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
    "deploy": "05-deploy-onnx-edge",
    "vqa": "07-finetune-vlm",
    "demo": "09-integrated-demo",
}


def test_help_lists_the_whole_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["info", "data", "detect", "evaluate", *PENDING]:
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


@pytest.fixture
def clean_02_zeroshot_runs():
    """`gdp detect` writes a fresh timestamped dir per invocation; clean up after."""
    runs_dir = resolve("runs/02-zeroshot")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


def test_detect_mini_bdd_writes_well_formed_predictions(clean_02_zeroshot_runs):
    """Fixture smoke path (H7: no metric is ever reported on mini_bdd) — proves the plumbing
    (config → detector → span→class mapping → predictions.json) end-to-end."""
    result = runner.invoke(app, ["detect", "-c", "configs/default.yaml", "--dataset", "mini_bdd"])
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/02-zeroshot")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    predictions_path = latest / "predictions.json"
    assert predictions_path.is_file()

    predictions = json.loads(predictions_path.read_text())
    assert isinstance(predictions, list)
    for pred in predictions:
        assert {"image_id", "category_id", "bbox", "score"} <= pred.keys()
        assert len(pred["bbox"]) == 4
        assert 0 <= pred["category_id"] < 10
        assert 0.0 <= pred["score"] <= 1.0


def test_detect_mini_bdd_rejects_non_val_split(clean_02_zeroshot_runs):
    result = runner.invoke(
        app, ["detect", "-c", "configs/default.yaml", "--dataset", "mini_bdd", "--split", "train"]
    )
    assert result.exit_code == 1
    assert "'val' split" in (result.stdout + str(result.stderr or ""))


def test_evaluate_mini_bdd_writes_metrics_with_full_provenance(clean_02_zeroshot_runs):
    """Fixture smoke path (H7): `metrics.json` must exist, be stamped `is_synthetic: true`, and
    carry every provenance field CLAUDE.md §8 requires — never trust a metric without them."""
    detect_result = runner.invoke(
        app, ["detect", "-c", "configs/default.yaml", "--dataset", "mini_bdd"]
    )
    assert detect_result.exit_code == 0, detect_result.stdout

    eval_result = runner.invoke(
        app, ["evaluate", "-c", "configs/default.yaml", "--dataset", "mini_bdd"]
    )
    assert eval_result.exit_code == 0, eval_result.stdout

    runs_dir = resolve("runs/02-zeroshot")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    metrics_path = latest / "metrics.json"
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["is_synthetic"] is True
    assert metrics["dataset"] == "mini_bdd"
    assert metrics["split"] == "val"
    assert metrics["num_images"] == 4
    assert metrics["model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert set(metrics["per_class_ap"]) == {
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    }
    assert metrics["chosen_on"] == "config_default"
    assert 0.0 <= metrics["box_threshold"] <= 1.0
    assert "git_sha" in metrics and "created" in metrics
    assert metrics["config"]["detector"]["model_id"] == "IDEA-Research/grounding-dino-tiny"


def test_evaluate_fails_loudly_with_no_predictions(monkeypatch, clean_02_zeroshot_runs):
    monkeypatch.setattr("gdp.cli._latest_predictions_path", lambda: None)
    result = runner.invoke(app, ["evaluate", "-c", "configs/default.yaml", "--dataset", "mini_bdd"])
    assert result.exit_code == 1
    assert "predictions not found" in (result.stdout + str(result.stderr or ""))


def test_evaluate_sweep_on_val_is_rejected_as_leakage(clean_02_zeroshot_runs):
    """The leakage guard (H8, spec 02 task 6): sweeping the threshold on val would let the
    baseline tune toward the very split it's supposed to be measured against. mini_bdd forces
    split=val, so `--sweep` on it must always be rejected before any inference runs."""
    result = runner.invoke(
        app, ["evaluate", "-c", "configs/default.yaml", "--dataset", "mini_bdd", "--sweep"]
    )
    assert result.exit_code == 1
    output = result.stdout + str(result.stderr or "")
    assert "leakage" in output or "'val' split" in output


def test_evaluate_sweep_and_box_threshold_are_mutually_exclusive(clean_02_zeroshot_runs):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "bdd100k",
            "--split",
            "train",
            "--sweep",
            "--box-threshold",
            "0.3",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in (result.stdout + str(result.stderr or ""))


def test_evaluate_box_threshold_override_is_labelled_cli_override(clean_02_zeroshot_runs):
    """`chosen_on` must honestly reflect where the threshold actually came from — a manual
    `--box-threshold` is neither the config default nor a train-slice sweep."""
    detect_result = runner.invoke(
        app, ["detect", "-c", "configs/default.yaml", "--dataset", "mini_bdd"]
    )
    assert detect_result.exit_code == 0, detect_result.stdout

    eval_result = runner.invoke(
        app,
        [
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--box-threshold",
            "0.1",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout

    runs_dir = resolve("runs/02-zeroshot")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((latest / "metrics.json").read_text())
    assert metrics["chosen_on"] == "cli_override"
    assert metrics["box_threshold"] == 0.1


def test_evaluate_chosen_on_train_replays_a_sweep_winner_honestly(clean_02_zeroshot_runs):
    """`--chosen-on train` is how the SLURM job (task 7) tells the val evaluation run that its
    `--box-threshold` value was genuinely picked by an earlier train-split sweep, not invented
    on the spot — `metrics.json` must record that provenance, not fall back to `cli_override`."""
    detect_result = runner.invoke(
        app, ["detect", "-c", "configs/default.yaml", "--dataset", "mini_bdd"]
    )
    assert detect_result.exit_code == 0, detect_result.stdout

    eval_result = runner.invoke(
        app,
        [
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--box-threshold",
            "0.2",
            "--chosen-on",
            "train",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout

    runs_dir = resolve("runs/02-zeroshot")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((latest / "metrics.json").read_text())
    assert metrics["chosen_on"] == "train"
    assert metrics["box_threshold"] == 0.2


def test_evaluate_chosen_on_rejects_values_other_than_train(clean_02_zeroshot_runs):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--box-threshold",
            "0.2",
            "--chosen-on",
            "val",
        ],
    )
    assert result.exit_code == 1
    assert "only accepts 'train'" in (result.stdout + str(result.stderr or ""))


def test_evaluate_chosen_on_requires_box_threshold(clean_02_zeroshot_runs):
    result = runner.invoke(
        app,
        ["evaluate", "-c", "configs/default.yaml", "--dataset", "mini_bdd", "--chosen-on", "train"],
    )
    assert result.exit_code == 1
    assert "requires --box-threshold" in (result.stdout + str(result.stderr or ""))
