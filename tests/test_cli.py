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
    for command in ["info", "data", "detect", "evaluate", "train", *PENDING]:
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
    monkeypatch.setattr("gdp.cli._latest_predictions_path", lambda run_spec="02-zeroshot": None)
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


@pytest.fixture
def clean_03_finetune_runs():
    """A fine-tuning CLI command writes a fresh timestamped dir; clean up after."""
    runs_dir = resolve("runs/03-finetune")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


def test_train_detector_writes_checkpoint_and_log(clean_03_finetune_runs, tmp_path):
    """A short, non-overfit run: checkpoint + processor + trainer_state.pt + train_log.jsonl."""
    log_every_1 = tmp_path / "log_every_1.yaml"
    log_every_1.write_text("training:\n  log_every: 1\n")

    result = runner.invoke(
        app,
        [
            "train",
            "detector",
            "-c",
            "configs/default.yaml",
            "-c",
            str(log_every_1),
            "--dataset",
            "mini_bdd",
            "--limit",
            "2",
            "--max-steps",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/03-finetune")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    ckpt_dirs = sorted(p for p in latest.iterdir() if p.name.startswith("checkpoint-"))
    assert ckpt_dirs, f"no checkpoint-<step> dir under {latest}"
    assert (ckpt_dirs[-1] / "trainer_state.pt").is_file()
    assert (ckpt_dirs[-1] / "config.json").is_file()
    assert (latest / "train_log.jsonl").is_file()
    log_lines = (latest / "train_log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1


def test_train_detector_overfit_gate_fails_loudly_when_target_unmet(clean_03_finetune_runs):
    """Design decision 5's gate: too few steps must not reach `overfit_loss_target`, and the CLI
    must exit non-zero and say so — never silently report a passing gate."""
    result = runner.invoke(
        app,
        [
            "train",
            "detector",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--overfit",
            "2",
            "--max-steps",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "overfit gate FAILED" in (result.stdout + str(result.stderr or ""))


def test_train_detector_overfit_gate_passes_and_exits_zero(clean_03_finetune_runs, tmp_path):
    """The gate's other branch: clearing `overfit_loss_target` must exit 0 and say so, not just
    fail loudly — a gate that can only ever report failure isn't a gate. A generous target isolates
    the exit-code plumbing from the fixture's real convergence behaviour (design decision 5: the
    laptop's job is proving the plumbing, not genuine convergence on 4 images)."""
    lenient_target = tmp_path / "lenient_overfit_target.yaml"
    lenient_target.write_text("training:\n  overfit_loss_target: 1000000.0\n  log_every: 1\n")

    result = runner.invoke(
        app,
        [
            "train",
            "detector",
            "-c",
            "configs/default.yaml",
            "-c",
            str(lenient_target),
            "--dataset",
            "mini_bdd",
            "--overfit",
            "2",
            "--max-steps",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "overfit gate passed" in result.stdout

    runs_dir = resolve("runs/03-finetune")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    assert any(p.name.startswith("checkpoint-") for p in latest.iterdir())


def test_detect_checkpoint_and_run_spec_write_under_the_named_spec_dir(clean_03_finetune_runs):
    """`--checkpoint`/`--run-spec` must reuse spec 02's exact detect code path (H8) — a local
    checkpoint dir loads fine as `detector.model_id`, and output lands under the named run-spec."""
    train_result = runner.invoke(
        app,
        [
            "train",
            "detector",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--limit",
            "2",
            "--max-steps",
            "1",
        ],
    )
    assert train_result.exit_code == 0, train_result.stdout
    runs_dir = resolve("runs/03-finetune")
    latest_train = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    ckpt_dirs = sorted(p for p in latest_train.iterdir() if p.name.startswith("checkpoint-"))
    checkpoint = ckpt_dirs[-1]

    detect_result = runner.invoke(
        app,
        [
            "detect",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--checkpoint",
            str(checkpoint),
            "--run-spec",
            "03-finetune",
        ],
    )
    assert detect_result.exit_code == 0, detect_result.stdout

    after = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    latest_detect = after[-1]
    assert latest_detect != latest_train
    assert (latest_detect / "predictions.json").is_file()


def test_compare_writes_comparison_json(clean_03_finetune_runs, tmp_path):
    zeroshot_metrics = tmp_path / "zeroshot_metrics.json"
    finetuned_metrics = tmp_path / "finetuned_metrics.json"
    base = {
        "dataset": "mini_bdd",
        "split": "val",
        "num_images": 4,
        "box_threshold": 0.25,
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "map": 0.10,
        "map50": 0.20,
        "per_class_ap": {"car": 0.10},
    }
    zeroshot_metrics.write_text(json.dumps(base))
    finetuned_metrics.write_text(json.dumps({**base, "map": 0.15, "map50": 0.25}))

    result = runner.invoke(
        app,
        ["compare", "--zeroshot", str(zeroshot_metrics), "--finetuned", str(finetuned_metrics)],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/03-finetune")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    comparison = json.loads((latest / "comparison.json").read_text())
    assert comparison["map_zeroshot"] == 0.10
    assert comparison["map_finetuned"] == 0.15
    assert comparison["map_delta"] == pytest.approx(0.05)


def test_compare_refuses_mismatched_splits(clean_03_finetune_runs, tmp_path):
    zeroshot_metrics = tmp_path / "zeroshot_metrics.json"
    finetuned_metrics = tmp_path / "finetuned_metrics.json"
    base = {
        "dataset": "mini_bdd",
        "split": "val",
        "num_images": 4,
        "box_threshold": 0.25,
        "map": 0.10,
        "map50": 0.20,
        "per_class_ap": {"car": 0.10},
    }
    zeroshot_metrics.write_text(json.dumps(base))
    finetuned_metrics.write_text(json.dumps({**base, "num_images": 999}))

    result = runner.invoke(
        app,
        ["compare", "--zeroshot", str(zeroshot_metrics), "--finetuned", str(finetuned_metrics)],
    )
    assert result.exit_code == 1
    assert "refusing to compare mismatched runs" in (result.stdout + str(result.stderr or ""))


def test_probe_openvocab_writes_demo_stamped_json(clean_03_finetune_runs):
    result = runner.invoke(
        app,
        [
            "probe-openvocab",
            "-c",
            "configs/default.yaml",
            "--images-root",
            "tests/fixtures/mini_bdd",
            "--limit",
            "2",
            "--box-threshold",
            "0.15",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/03-finetune")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    probe = json.loads((latest / "probe.json").read_text())
    assert probe["is_demo"] is True
    assert probe["is_metric"] is False
    assert "construction cone" in probe["phrases"]
    forbidden_keys = {"accuracy", "map", "precision", "recall"}
    assert not forbidden_keys & probe.keys()


def test_compare_dump_pairs_saves_a_miss_to_hit_crop(clean_03_finetune_runs, tmp_path):
    """Ground truth image_id 0 has a pedestrian (category_id 0) at bbox [12, 203, 96, 37] —
    zero-shot predicts nothing there, fine-tuned catches it exactly."""
    zeroshot_metrics = tmp_path / "zeroshot_metrics.json"
    finetuned_metrics = tmp_path / "finetuned_metrics.json"
    base = {
        "dataset": "mini_bdd",
        "split": "val",
        "num_images": 4,
        "box_threshold": 0.25,
        "map": 0.10,
        "map50": 0.20,
        "per_class_ap": {"pedestrian": 0.10},
    }
    zeroshot_metrics.write_text(json.dumps(base))
    finetuned_metrics.write_text(json.dumps({**base, "map": 0.20, "map50": 0.30}))

    zeroshot_predictions = tmp_path / "zeroshot_predictions.json"
    finetuned_predictions = tmp_path / "finetuned_predictions.json"
    zeroshot_predictions.write_text(json.dumps([]))
    finetuned_predictions.write_text(
        json.dumps([{"image_id": 0, "category_id": 0, "bbox": [12, 203, 96, 37], "score": 0.9}])
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "--zeroshot",
            str(zeroshot_metrics),
            "--finetuned",
            str(finetuned_metrics),
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--zeroshot-predictions",
            str(zeroshot_predictions),
            "--finetuned-predictions",
            str(finetuned_predictions),
            "--dump-pairs",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/03-finetune")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    pairs_dir = latest / "pairs"
    assert pairs_dir.is_dir()
    saved = list(pairs_dir.iterdir())
    assert len(saved) == 1
    assert "pedestrian" in saved[0].name
