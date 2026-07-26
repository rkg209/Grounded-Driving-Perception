"""The CLI is an honest roadmap: implemented commands work, unimplemented ones name their spec.

Memory hygiene (CLAUDE.md §4): every test here that invokes `detect`, `train detector`, or
`probe-openvocab` through `CliRunner` reaches a real `.from_pretrained("grounding-dino-tiny")`
inside the command body — the CLI owns the load, so there is no fixture to scope it to. Each such
test is marked `model_heavy` individually (the file's other ~18 tests are pure argument-validation
and JSON plumbing, and must keep running on a bare `uv run pytest`). Unmarked, this one file loads
the detector 11 times in a single process and hangs a 16GB M4 — the exact failure [SEQ-0034] was
written about. Run them deliberately, and alone:

    uv run pytest tests/test_cli.py -m model_heavy
"""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from gdp.cli import app
from gdp.paths import resolve

runner = CliRunner()

# command → the spec that will implement it
PENDING: dict[str, str] = {}


def test_help_lists_the_whole_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "info",
        "data",
        "detect",
        "evaluate",
        "train",
        "deploy",
        "vqa",
        "demo",
        *PENDING,
    ]:
        assert command in result.stdout


def test_deploy_help_lists_its_subcommands():
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    for subcommand in ["export", "quantize", "bench", "evaluate-variants"]:
        assert subcommand in result.stdout


def test_train_help_lists_detector_and_vlm():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "detector" in result.stdout
    assert "vlm" in result.stdout


def test_train_vlm_help_is_implemented():
    """Spec 07 (H2, cluster-only): the command exists and documents itself, but actually running
    it needs the real 3B model + real DriveLM data — never invoked for real in a laptop test."""
    result = runner.invoke(app, ["train", "vlm", "--help"])
    assert result.exit_code == 0
    assert "--train-jsonl" in result.stdout
    assert "--overfit" in result.stdout
    assert "--resume" in result.stdout


def test_vqa_help_lists_generate():
    result = runner.invoke(app, ["vqa", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout


def test_vqa_generate_help_is_implemented():
    result = runner.invoke(app, ["vqa", "generate", "--help"])
    assert result.exit_code == 0
    assert "--adapter" in result.stdout


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


def test_data_prepare_help_lists_prepare_drivelm():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "prepare-drivelm" in result.stdout


@pytest.fixture
def clean_06_data_runs():
    """`gdp data prepare-drivelm` writes a fresh timestamped dir per invocation; clean up after."""
    runs_dir = resolve("runs/06-data")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


def test_data_prepare_drivelm_writes_jsonl_and_stats_end_to_end(clean_06_data_runs):
    """Acceptance 1-3: `gdp data prepare-drivelm --dataset mini_drivelm` exits 0 and writes
    train/val JSONL whose line counts match `stats_{train,val}.json`'s `qa_count`, with zero
    scene-token overlap between the two splits."""
    result = runner.invoke(
        app,
        ["data", "prepare-drivelm", "-c", "configs/default.yaml", "--dataset", "mini_drivelm"],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/06-data")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    train_lines = (latest / "train.jsonl").read_text().splitlines()
    val_lines = (latest / "val.jsonl").read_text().splitlines()

    train_stats = json.loads((latest / "stats_train.json").read_text())
    val_stats = json.loads((latest / "stats_val.json").read_text())
    assert len(train_lines) == train_stats["qa_count"]
    assert len(val_lines) == val_stats["qa_count"]
    for field in ("official_split", "split_provenance", "caveat", "is_synthetic"):
        assert field in train_stats
        assert field in val_stats

    train_tokens = {json.loads(line)["scene_token"] for line in train_lines}
    val_tokens = {json.loads(line)["scene_token"] for line in val_lines}
    assert train_tokens.isdisjoint(val_tokens)


def test_data_prepare_drivelm_subsample_path(clean_06_data_runs):
    result = runner.invoke(
        app,
        [
            "data",
            "prepare-drivelm",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_drivelm",
            "--train-fraction",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/06-data")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    assert (latest / "subsample.json").is_file()
    subsample = json.loads((latest / "subsample.json").read_text())
    assert subsample["unit"] == "scene"
    assert subsample["fraction"] == 0.5


def test_data_prepare_drivelm_rejects_val_with_train_fraction(clean_06_data_runs):
    result = runner.invoke(
        app,
        [
            "data",
            "prepare-drivelm",
            "--dataset",
            "mini_drivelm",
            "--split",
            "val",
            "--train-fraction",
            "0.5",
        ],
    )
    assert result.exit_code == 1
    assert "never subsampled" in (result.stdout + str(result.stderr or ""))


def test_data_prepare_drivelm_rejects_unknown_dataset(clean_06_data_runs):
    result = runner.invoke(app, ["data", "prepare-drivelm", "--dataset", "nope"])
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


# Ground truth for mini_bdd image_id 0: a pedestrian (category 0) at xywh [12, 203, 96, 37], plus
# a rider and a car. Images 1-3 hold the other 7 boxes. Predicting that one pedestrian perfectly
# and nothing else is therefore a *perfect* score on image 0 alone, and a poor one across all four
# — which is exactly what makes it a probe for whether `--limit`'s image set is being respected.
_PERFECT_ON_IMAGE_0 = [{"image_id": 0, "category_id": 0, "bbox": [12, 203, 96, 37], "score": 0.9}]


def _write_predictions_with_meta(tmp_path, image_ids: list[int] | None):
    """A predictions.json (+ optional sidecar) as `gdp detect` would have written it."""
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(_PERFECT_ON_IMAGE_0))
    if image_ids is not None:
        (tmp_path / "predictions_meta.json").write_text(
            json.dumps({"image_ids": image_ids, "num_images": len(image_ids)})
        )
    return predictions_path


def _evaluate(predictions_path):
    return runner.invoke(
        app,
        [
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--predictions",
            str(predictions_path),
            "--box-threshold",
            "0.5",
        ],
    )


def test_evaluate_scores_only_the_images_detect_actually_ran_on(clean_02_zeroshot_runs, tmp_path):
    """The `--limit` trap: predictions from a 1-image run must be scored against that 1 image's
    ground truth, not the whole split's. Scored against all 4 mini_bdd images, the 7 boxes in
    images 1-3 become phantom false negatives — mAP collapses and `--sweep` would chase a
    threshold that is too low, with no error shown anywhere."""
    result = _evaluate(_write_predictions_with_meta(tmp_path, image_ids=[0]))
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/02-zeroshot")
    metrics = json.loads(
        (max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime) / "metrics.json").read_text()
    )
    assert metrics["num_images"] == 1, "num_images must report the evaluated set, not the split"
    # Image 0 holds 3 GT boxes; the one predicted pedestrian is the only true positive available.
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2, "only image 0's other 2 boxes are missed — not images 1-3's seven"
    assert metrics["per_class_ap"]["pedestrian"] == pytest.approx(1.0)


def test_evaluate_without_a_sidecar_assumes_the_full_split_and_says_so(
    clean_02_zeroshot_runs, tmp_path
):
    """A hand-written predictions file has no provenance. Falling back to the full split is the
    only honest default, but it must be stated out loud, not assumed in silence."""
    result = _evaluate(_write_predictions_with_meta(tmp_path, image_ids=None))
    assert result.exit_code == 0, result.stdout
    assert "assuming these predictions cover the whole split" in str(result.stderr or "")

    runs_dir = resolve("runs/02-zeroshot")
    metrics = json.loads(
        (max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime) / "metrics.json").read_text()
    )
    assert metrics["num_images"] == 4
    assert metrics["fn"] == 9, "all 10 GT boxes across 4 images, minus the 1 hit"


def test_evaluate_rejects_predictions_from_a_different_split(clean_02_zeroshot_runs, tmp_path):
    """A sidecar naming image ids the annotations file has never heard of means the predictions
    and the ground truth are not describing the same data. Refuse, rather than score a subset."""
    result = _evaluate(_write_predictions_with_meta(tmp_path, image_ids=[0, 4242]))
    assert result.exit_code == 1
    assert "disagree about which split this is" in (result.stdout + str(result.stderr or ""))


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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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


@pytest.mark.model_heavy
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
